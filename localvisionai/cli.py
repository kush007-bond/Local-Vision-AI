"""
LocalVisionAI CLI — built with Typer.

Available commands:
    localvisionai run    — Run the pipeline on a video source
    localvisionai serve  — Start the REST + WebSocket API server
    localvisionai search — Search a processed timeline database
    localvisionai models — List or pull available models
    localvisionai info   — Show hardware information
"""

from __future__ import annotations

import asyncio
import sys
from typing import List, Optional

import typer
from rich.console import Console

app = typer.Typer(
    name="localvisionai",
    help="Local AI-powered video understanding pipeline.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)

models_app = typer.Typer(help="Manage models (list, pull).")
app.add_typer(models_app, name="models")

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# localvisionai run
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def run(
    video: Optional[str] = typer.Option(None, "--video", "-v", help="Path to video file."),
    source: str = typer.Option("file", "--source", "-s", help="Source type: file|webcam|rtsp|url|screen."),
    device: int = typer.Option(0, "--device", help="Webcam device index (for --source webcam)."),
    rtsp_url: Optional[str] = typer.Option(None, "--rtsp-url", help="RTSP stream URL."),
    backend: str = typer.Option(
        "ollama", "--backend", "-b",
        help="Model backend: ollama|transformers|llamacpp|mlx|openai|anthropic|gemini|lmstudio.",
    ),
    model: str = typer.Option("gemma3", "--model", "-m", help="Model identifier."),
    load_4bit: bool = typer.Option(False, "--4bit", help="Load model in 4-bit quantization (transformers only)."),
    api_key: Optional[str] = typer.Option(
        None, "--api-key",
        help="API key for cloud backends (openai/anthropic/gemini). Defaults to env var.",
        envvar=["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"],
    ),
    api_base: Optional[str] = typer.Option(
        None, "--api-base",
        help="Override base URL for the API (e.g. http://localhost:1234/v1 for LM Studio).",
    ),
    max_tokens: int = typer.Option(512, "--max-tokens", help="Max tokens to generate (cloud backends)."),
    fps: float = typer.Option(1.0, "--fps", help="Frames per second to sample (uniform strategy)."),
    sampler: str = typer.Option("uniform", "--sampler", help="Sampling strategy: uniform|scene|keyframe|adaptive."),
    prompt: str = typer.Option(
        "Describe what is happening in this frame in one sentence.",
        "--prompt", "-p",
        help="Prompt sent to the model for each frame.",
    ),
    system_prompt: Optional[str] = typer.Option(None, "--system-prompt", help="System prompt override."),
    context_mode: str = typer.Option("none", "--context", help="Context mode: none|sliding_window."),
    output_formats: List[str] = typer.Option(["console", "json"], "--output-formats", "-o", help="Output formats."),
    output_dir: str = typer.Option("./output/", "--output-dir", help="Directory for output files."),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="YAML config file path."),
    audio: Optional[bool] = typer.Option(
        None, "--audio/--no-audio",
        help="Enable audio analysis alongside video frames.",
    ),
    audio_mode: Optional[str] = typer.Option(
        None, "--audio-mode",
        help="Audio routing mode: auto|native. Default: auto.",
    ),
    audio_window: Optional[float] = typer.Option(
        None, "--audio-window",
        help="Seconds of audio per frame (default: 3.0).",
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Show latency and token counts."),
) -> None:
    """Run the LocalVisionAI pipeline on a video source."""
    from localvisionai.config import PipelineConfig
    from localvisionai.pipeline import Pipeline

    cli_kwargs = {
        "source": source,
        "video": video,
        "device": device,
        "rtsp_url": rtsp_url,
        "backend": backend,
        "model": model,
        "load_in_4bit": load_4bit,
        "api_key": api_key,
        "api_base": api_base,
        "max_tokens": max_tokens,
        "fps": fps,
        "sampler": sampler,
        "prompt": prompt,
        "system_prompt": system_prompt,
        "context_mode": context_mode,
        "output_formats": output_formats,
        "output_dir": output_dir,
        "config_file": config_file,
        "audio": audio,
        "audio_mode": audio_mode,
        "audio_window": audio_window,
    }

    try:
        config = PipelineConfig.from_cli(cli_kwargs)
    except Exception as e:
        console.print(f"[bold red]Configuration error:[/bold red] {e}")
        raise typer.Exit(1)

    try:
        asyncio.run(Pipeline(config).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user.[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Pipeline error:[/bold red] {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        raise typer.Exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# localvisionai analyze
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def analyze(
    video: str = typer.Argument(..., help="Path to the video file to analyse."),
    backend: str = typer.Option(
        "ollama", "--backend", "-b",
        help="Model backend: ollama|transformers|openai|anthropic|gemini|lmstudio.",
    ),
    model: str = typer.Option("gemma3", "--model", "-m", help="Model identifier."),
    api_key: Optional[str] = typer.Option(
        None, "--api-key",
        help="API key for cloud backends (openai/anthropic/gemini).",
        envvar=["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"],
    ),
    api_base: Optional[str] = typer.Option(
        None, "--api-base",
        help="Override base URL for the API (e.g. LM Studio endpoint).",
    ),
    max_tokens: int = typer.Option(1024, "--max-tokens", help="Max tokens to generate per inference call."),
    fps: float = typer.Option(1.0, "--fps", help="Frames per second to sample."),
    sampler: str = typer.Option(
        "uniform", "--sampler",
        help="Sampling strategy: uniform|keyframe.",
    ),
    prompt: str = typer.Option(
        "Describe what is happening in this frame in one sentence.",
        "--prompt", "-p",
        help="Prompt used for per-frame analysis.",
    ),
    output_dir: str = typer.Option(
        "./output/", "--output-dir",
        help="Directory where the analysis cache and thumbnail are saved.",
    ),
    audio: bool = typer.Option(
        False, "--audio/--no-audio",
        help="Enable audio extraction alongside video frames (sent natively to the model).",
    ),
    no_qa: bool = typer.Option(
        False, "--no-qa",
        help="Skip the interactive Q&A phase after the summary.",
    ),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="YAML config file path."),
) -> None:
    """Analyse a video: extract frames, transcribe audio, generate a summary, then answer questions.

    Results are cached to <output_dir>/<video_stem>_analysis.json so re-running
    the command skips the expensive analysis pass and goes straight to the summary
    and Q&A phase.
    """
    from localvisionai.config import PipelineConfig
    from localvisionai.analysis.pipeline import AnalysisPipeline

    cli_kwargs: dict = {
        "config_file": config_file,
        "video": video,
        "source": "file",
        "backend": backend,
        "model": model,
        "api_key": api_key,
        "api_base": api_base,
        "max_tokens": max_tokens,
        "fps": fps,
        "sampler": sampler,
        "prompt": prompt,
        "output_formats": ["json"],
        "output_dir": output_dir,
    }

    if audio:
        cli_kwargs["audio"] = True

    try:
        config = PipelineConfig.from_cli(cli_kwargs)
    except Exception as e:
        console.print(f"[bold red]Configuration error:[/bold red] {e}")
        raise typer.Exit(1)

    try:
        asyncio.run(AnalysisPipeline(config).run(skip_qa=no_qa))
    except KeyboardInterrupt:
        console.print("\n[yellow]Analysis interrupted by user.[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Analysis error:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# localvisionai serve
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host (default: localhost only)."),
    port: int = typer.Option(8765, "--port", help="Bind port."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes (dev mode)."),
) -> None:
    """Start the LocalVisionAI REST + WebSocket API server."""
    try:
        import uvicorn
        from localvisionai.api.server import app as fastapi_app
    except ImportError:
        console.print("[bold red]FastAPI/uvicorn not installed.[/bold red]")
        raise typer.Exit(1)

    console.print(f"[bold green]Starting LocalVisionAI API server on http://{host}:{port}[/bold green]")
    console.print(f"  Swagger UI: [link]http://{host}:{port}/docs[/link]")
    uvicorn.run("localvisionai.api.server:app", host=host, port=port, reload=reload)


# ─────────────────────────────────────────────────────────────────────────────
# localvisionai search
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def search(
    db: str = typer.Option(..., "--db", help="Path to the SQLite timeline database."),
    query: str = typer.Option(..., "--query", "-q", help="Natural language search query."),
    limit: int = typer.Option(10, "--limit", help="Maximum number of results to return."),
) -> None:
    """Search a processed video timeline by natural language query."""
    try:
        import sqlite3
        conn = sqlite3.connect(db)
        cur = conn.execute(
            "SELECT timestamp, description FROM frames_fts WHERE description MATCH ? LIMIT ?",
            (query, limit),
        )
        rows = cur.fetchall()
        conn.close()
    except Exception as e:
        console.print(f"[bold red]Search error:[/bold red] {e}")
        raise typer.Exit(1)

    if not rows:
        console.print("[yellow]No results found.[/yellow]")
        return

    from rich.table import Table
    from localvisionai.utils.timing import format_timestamp

    table = Table(title=f'Search results for: "{query}"', show_header=True)
    table.add_column("Timestamp", style="green")
    table.add_column("Description")
    for ts, desc in rows:
        table.add_row(format_timestamp(float(ts)), desc)
    console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
# localvisionai info
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def info() -> None:
    """Show hardware detection results and recommended backend."""
    from localvisionai.utils.hardware import print_hardware_info
    print_hardware_info()


# ─────────────────────────────────────────────────────────────────────────────
# localvisionai models list / pull
# ─────────────────────────────────────────────────────────────────────────────

@models_app.command("list")
def models_list(
    backend: str = typer.Option("ollama", "--backend", "-b", help="Backend to query."),
) -> None:
    """List available models for a backend."""
    if backend == "ollama":
        try:
            import ollama
            import asyncio

            async def _list():
                client = ollama.AsyncClient()
                resp = await client.list()
                return [m.model for m in resp.models]

            models = asyncio.run(_list())
            console.print(f"[bold]Ollama models ({len(models)}):[/bold]")
            for m in sorted(models):
                console.print(f"  • {m}")
        except ImportError:
            console.print("[red]Ollama package not installed.[/red]")
        except Exception as e:
            console.print(f"[red]Error connecting to Ollama: {e}[/red]")
    else:
        console.print(f"[yellow]Model listing not yet supported for backend: {backend}[/yellow]")


@models_app.command("pull")
def models_pull(
    model_name: str = typer.Argument(..., help="Model name to pull (e.g. gemma3, qwen2-vl:7b)."),
) -> None:
    """Pull a model via Ollama."""
    try:
        import subprocess
        console.print(f"[bold]Pulling model: {model_name}[/bold]")
        subprocess.run(["ollama", "pull", model_name], check=True)
        console.print(f"[green]✅ Model '{model_name}' pulled successfully.[/green]")
    except FileNotFoundError:
        console.print("[red]Ollama not found. Install from https://ollama.ai[/red]")
        raise typer.Exit(1)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]ollama pull failed: {e}[/red]")
        raise typer.Exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
