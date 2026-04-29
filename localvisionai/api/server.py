"""
LocalVisionAI REST + WebSocket API server.

Endpoints:
    GET  /health                  health check
    GET  /api/backends            list supported backends and example models
    POST /api/jobs                create and start a pipeline job
    GET  /api/jobs                list all jobs (summary)
    GET  /api/jobs/{job_id}       job details + accumulated results
    DELETE /api/jobs/{job_id}     cancel a running job

    WS   /ws/{job_id}             stream live results for a job

WebSocket message schema:
    {"type": "result",   "data": {timestamp, description, latency_ms, ...}}
    {"type": "status",   "status": "running"|"completed"|"failed"|"cancelled"}
    {"type": "complete", "job_id": "..."}
    {"type": "error",    "message": "..."}
"""

from __future__ import annotations

import asyncio
import datetime
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LocalVisionAI API",
    version="0.1.0",
    description="Local AI-powered video understanding pipeline — REST + WebSocket API.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Dev only; tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the built frontend if it exists
import os
from pathlib import Path

_FRONTEND_DIST = Path(__file__).parent.parent.parent / "frontend" / "dist"


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class WebcamFrameRequest(BaseModel):
    image: str      # base64-encoded JPEG
    timestamp: float


class JobCreateRequest(BaseModel):
    backend: str = "ollama"
    model_id: str = "gemma3"
    source_type: str = "file"   # file | webcam | rtsp | url | screen | audio
    source_path: Optional[str] = None
    device_index: int = 0
    rtsp_url: Optional[str] = None
    fps: float = Field(1.0, gt=0, le=30)
    sampler: str = "uniform"
    prompt: str = "Describe what is happening in this frame in one sentence."
    system_prompt: Optional[str] = None
    context_mode: str = "none"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    max_tokens: int = Field(512, ge=1)
    output_formats: list[str] = ["json"]
    output_dir: str = "./output/"
    # Audio settings (only used when source_type='audio' or audio=True)
    audio: bool = False
    audio_mode: str = "auto"     # auto | native
    audio_window: float = 3.0


# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------

@dataclass
class JobState:
    job_id: str
    backend: str
    model_id: str
    source: str
    status: str = "queued"          # queued | running | completed | failed | cancelled
    results: list[dict] = field(default_factory=list)
    error: Optional[str] = None
    started_at: str = ""
    completed_at: Optional[str] = None
    task: Optional[asyncio.Task] = None
    ws_handler: object = None       # WebSocketOutput instance
    browser_source: object = None   # BrowserCaptureSource for webcam jobs

    def summary(self) -> dict:
        return {
            "job_id": self.job_id,
            "backend": self.backend,
            "model_id": self.model_id,
            "source": self.source,
            "status": self.status,
            "result_count": len(self.results),
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    def detail(self) -> dict:
        return {**self.summary(), "results": self.results}


# In-memory job store
_jobs: dict[str, JobState] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _build_pipeline_config(req: JobCreateRequest):
    """Build a PipelineConfig from an API request."""
    from localvisionai.config import PipelineConfig
    cli_args = {
        "source": req.source_type,
        "video": req.source_path,
        "device": req.device_index,
        "rtsp_url": req.rtsp_url,
        "backend": req.backend,
        "model": req.model_id,
        "load_in_4bit": False,
        "api_key": req.api_key,
        "api_base": req.api_base,
        "max_tokens": req.max_tokens,
        "fps": req.fps,
        "sampler": req.sampler,
        "prompt": req.prompt,
        "system_prompt": req.system_prompt,
        "context_mode": req.context_mode,
        "output_formats": req.output_formats,
        "output_dir": req.output_dir,
        "config_file": None,
    }
    # Pass audio settings when explicitly requested or when source is audio-only
    if req.audio or req.source_type == "audio":
        cli_args["audio"] = True
        cli_args["audio_mode"] = req.audio_mode
        cli_args["audio_window"] = req.audio_window
    return PipelineConfig.from_cli(cli_args)


async def _run_job(job: JobState, req: JobCreateRequest) -> None:
    """Background task: run the pipeline and update job state."""
    from localvisionai.pipeline import Pipeline
    from localvisionai.outputs.websocket_output import WebSocketOutput

    job.status = "running"
    job.started_at = _now_iso()

    ws_handler = WebSocketOutput()
    job.ws_handler = ws_handler

    # For webcam jobs, create a BrowserCaptureSource so the browser retains
    # exclusive camera access (avoids OpenCV/getUserMedia conflict on Windows).
    browser_source = None
    if req.source_type == "webcam":
        from localvisionai.inputs.browser_source import BrowserCaptureSource
        browser_source = BrowserCaptureSource()
        job.browser_source = browser_source

    # Broadcast status change to any already-connected WS clients
    await _broadcast(job.job_id, {"type": "status", "status": "running"})

    try:
        config = _build_pipeline_config(req)
        pipeline = Pipeline(config)
        pipeline.job_id = job.job_id

        # Accumulate results into job state via a capture handler
        captured: list[dict] = []

        class _CaptureHandler:
            async def open(self, _job_id): pass
            async def handle(self, result):
                d = result.to_dict()
                captured.append(d)
                job.results.append(d)
            async def close(self): pass

        await pipeline.run(
            extra_handlers=[ws_handler, _CaptureHandler()],
            source=browser_source,  # None for non-webcam jobs (build_source runs normally)
        )

        job.status = "completed"

    except asyncio.CancelledError:
        job.status = "cancelled"
        await _broadcast(job.job_id, {"type": "status", "status": "cancelled"})
        raise

    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
        await _broadcast(job.job_id, {"type": "error", "message": str(exc)})

    finally:
        job.completed_at = _now_iso()
        await _broadcast(job.job_id, {
            "type": "complete",
            "job_id": job.job_id,
            "status": job.status,
        })


# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------

# job_id → set of connected WebSocket objects
_ws_connections: dict[str, set[WebSocket]] = {}


async def _broadcast(job_id: str, message: dict) -> None:
    """Send a JSON message to all WebSocket clients watching a job."""
    connections = _ws_connections.get(job_id, set())
    dead: set[WebSocket] = set()
    for ws in list(connections):
        try:
            await ws.send_json(message)
        except Exception:
            dead.add(ws)
    connections -= dead


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


@app.get("/api/backends")
async def list_backends():
    """Return all supported backends with their example models and auth requirements."""
    return {
        "backends": [
            {
                "id": "ollama",
                "label": "Ollama",
                "type": "local",
                "requires_api_key": False,
                "default_model": "gemma3",
                "example_models": ["gemma3", "qwen2-vl", "llava", "llava-llama3", "moondream"],
                "install": "pip install localvisionai[ollama]",
                "note": "Ollama must be running: ollama serve",
            },
            {
                "id": "openai",
                "label": "OpenAI",
                "type": "cloud",
                "requires_api_key": True,
                "key_env_var": "OPENAI_API_KEY",
                "default_model": "gpt-4o",
                "example_models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
                "install": "pip install localvisionai[openai]",
            },
            {
                "id": "anthropic",
                "label": "Anthropic (Claude)",
                "type": "cloud",
                "requires_api_key": True,
                "key_env_var": "ANTHROPIC_API_KEY",
                "default_model": "claude-sonnet-4-6",
                "example_models": ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5"],
                "install": "pip install localvisionai[anthropic]",
            },
            {
                "id": "gemini",
                "label": "Google Gemini",
                "type": "cloud",
                "requires_api_key": True,
                "key_env_var": "GOOGLE_API_KEY",
                "default_model": "gemini-2.0-flash",
                "example_models": ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
                "install": "pip install localvisionai[gemini]",
            },
            {
                "id": "lmstudio",
                "label": "LM Studio",
                "type": "local",
                "requires_api_key": False,
                "default_model": "local-model",
                "example_models": ["llava-v1.6-mistral-7b", "qwen2-vl-7b", "internvl2-8b"],
                "install": "pip install localvisionai[openai]",
                "note": "LM Studio server must be running on localhost:1234",
            },
            {
                "id": "transformers",
                "label": "HuggingFace Transformers",
                "type": "local",
                "requires_api_key": False,
                "default_model": "Qwen/Qwen2-VL-7B-Instruct",
                "example_models": [
                    "Qwen/Qwen2-VL-7B-Instruct",
                    "llava-hf/llava-1.5-7b-hf",
                    "google/paligemma-3b-pt-224",
                ],
                "install": "pip install localvisionai[transformers]",
            },
        ]
    }


@app.post("/api/jobs", status_code=201)
async def create_job(req: JobCreateRequest):
    """Create and immediately start a new pipeline job."""
    job_id = f"j_{uuid.uuid4().hex[:8]}"
    source_label = req.source_path or req.rtsp_url or req.source_type

    job = JobState(
        job_id=job_id,
        backend=req.backend,
        model_id=req.model_id,
        source=source_label or req.source_type,
    )
    _jobs[job_id] = job
    _ws_connections[job_id] = set()

    # Early validation before building config
    if req.source_type == "file":
        if not req.source_path:
            raise HTTPException(status_code=422, detail="source_path is required when source_type is 'file'.")
        clean_path = req.source_path.strip().strip('"').strip("'")
        if not Path(clean_path).is_file():
            del _jobs[job_id]
            del _ws_connections[job_id]
            raise HTTPException(
                status_code=422,
                detail=f"Video file not found: {clean_path}. Enter an absolute path to an existing file (no surrounding quotes).",
            )
        req.source_path = clean_path
    if req.source_type == "audio":
        if not req.source_path:
            raise HTTPException(status_code=422, detail="source_path is required when source_type is 'audio'.")
        clean_path = req.source_path.strip().strip('"').strip("'")
        if not Path(clean_path).is_file():
            del _jobs[job_id]
            del _ws_connections[job_id]
            raise HTTPException(
                status_code=422,
                detail=f"Audio file not found: {clean_path}. Enter an absolute path to an existing audio/video file.",
            )
        req.source_path = clean_path
    if req.source_type == "rtsp" and not req.rtsp_url:
        raise HTTPException(status_code=422, detail="rtsp_url is required when source_type is 'rtsp'.")

    # Validate full config before starting
    try:
        _build_pipeline_config(req)
    except Exception as e:
        del _jobs[job_id]
        del _ws_connections[job_id]
        raise HTTPException(status_code=422, detail=str(e))

    # Launch pipeline as background task
    task = asyncio.create_task(_run_job(job, req))
    job.task = task

    return job.summary()


@app.get("/api/jobs")
async def list_jobs():
    """Return summary of all jobs, newest first."""
    return {"jobs": [j.summary() for j in reversed(list(_jobs.values()))]}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    """Return full job detail including all accumulated results."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job.detail()


@app.post("/api/jobs/{job_id}/frame", status_code=202)
async def push_webcam_frame(job_id: str, req: WebcamFrameRequest):
    """
    Receive a browser-captured webcam frame and push it into the pipeline.

    Called by the frontend once per configured FPS interval while a webcam job
    is running. The frame is decoded from base64 JPEG and forwarded to the
    BrowserCaptureSource that feeds the pipeline.
    """
    import base64
    from io import BytesIO
    from PIL import Image as PilImage

    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job.status not in ("queued", "running"):
        return {"ok": False, "reason": job.status}

    browser_source = job.browser_source
    if browser_source is None:
        # Job not yet initialised (model still loading) — silently drop the frame.
        return {"ok": False, "reason": "not_ready"}

    try:
        img_bytes = base64.b64decode(req.image)
        img = PilImage.open(BytesIO(img_bytes)).convert("RGB")
        await browser_source.push_frame(img, req.timestamp)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid frame data: {e}")

    return {"ok": True}


@app.delete("/api/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running job."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job.status not in ("queued", "running"):
        raise HTTPException(status_code=409, detail=f"Job '{job_id}' is already {job.status}.")

    if job.task and not job.task.done():
        job.task.cancel()

    # Signal browser-capture source to stop streaming
    if job.browser_source is not None:
        job.browser_source.stop()

    job.status = "cancelled"
    job.completed_at = _now_iso()
    return {"job_id": job_id, "status": "cancelled"}


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws/{job_id}")
async def job_websocket(websocket: WebSocket, job_id: str):
    """
    Stream live inference results for a job.

    The client receives JSON messages:
        {"type": "result",   "data": {...}}     on each new frame
        {"type": "status",   "status": "..."}   on state transitions
        {"type": "complete", "job_id": "..."}   when the job finishes
        {"type": "error",    "message": "..."}  on failure
    """
    job = _jobs.get(job_id)
    if job is None:
        await websocket.close(code=4004, reason=f"Job '{job_id}' not found.")
        return

    await websocket.accept()
    _ws_connections.setdefault(job_id, set()).add(websocket)

    # Send all results buffered so far (client joining mid-job or after completion)
    for result in job.results:
        await websocket.send_json({"type": "result", "data": result})

    # If already done, send final status and close
    if job.status in ("completed", "failed", "cancelled"):
        if job.status == "failed" and job.error:
            await websocket.send_json({"type": "error", "message": job.error})
        await websocket.send_json({"type": "complete", "job_id": job_id, "status": job.status})
        _ws_connections[job_id].discard(websocket)
        await websocket.close()
        return

    # Subscribe to the live handler queue
    ws_handler = job.ws_handler
    if ws_handler is None:
        # Handler not yet created (job is still queued), poll briefly
        for _ in range(20):
            await asyncio.sleep(0.1)
            if job.ws_handler is not None:
                ws_handler = job.ws_handler
                break

    if ws_handler is not None:
        q = ws_handler.subscribe()
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    # Send a ping to keep the connection alive
                    try:
                        await websocket.send_json({"type": "ping"})
                    except Exception:
                        break
                    continue

                if msg is None:
                    # End-of-stream sentinel from WebSocketOutput.close()
                    break

                try:
                    await websocket.send_json(msg)
                except Exception:
                    break

        except WebSocketDisconnect:
            pass
        finally:
            ws_handler.unsubscribe(q)

    _ws_connections[job_id].discard(websocket)
    try:
        await websocket.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Serve built frontend (production)
# Mount LAST so all /api/* and /ws/* routes above take priority.
# StaticFiles(html=True) serves index.html for any path not found on disk,
# which is exactly what a React SPA needs.
# ---------------------------------------------------------------------------

if _FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")
