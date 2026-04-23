"""AnalysisPipeline — offline video analysis with caching, summary, and Q&A.

Workflow
--------
1. **Frame analysis** — sample the video at the configured FPS, run per-frame
   inference, save results to a JSON cache file.  The cache survives between
   runs so the expensive pass is only paid once per video.
2. **Audio analysis** — extract the full audio track with ffmpeg and send audio
   chunks natively to the model alongside each frame (requires a multimodal
   adapter that declares ``supports_audio = True``).
3. **Summarisation** — the model is given all frame descriptions and the full
   audio transcript and asked to write a comprehensive summary.
4. **Interactive Q&A** — the user can ask questions; each answer is grounded in
   the cached frame timeline, transcript, and summary.
"""

from __future__ import annotations

import asyncio
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from localvisionai.adapters.base import InferenceResult
from localvisionai.adapters.registry import get_adapter
from localvisionai.analysis.cache import AnalysisCache
from localvisionai.config import PipelineConfig
from localvisionai.inputs import build_source
from localvisionai.prompts.builder import build_prompt
from localvisionai.sampling import build_sampler
from localvisionai.utils.logging import get_logger

logger = get_logger(__name__)
console = Console()

# Maximum number of frame records included in the Q&A context to keep the
# prompt within a reasonable token budget for all backends.
_QA_MAX_FRAMES = 120


class AnalysisPipeline:
    """
    Full offline video analysis pipeline.

    Usage::

        config = PipelineConfig.from_cli({...})
        await AnalysisPipeline(config).run()
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.job_id = f"analysis_{uuid.uuid4().hex[:8]}"

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(self, skip_qa: bool = False) -> None:
        """Run the full analysis → summary → Q&A workflow."""
        video_path = self.config.source.path
        if not video_path:
            raise ValueError("No video path provided. Pass --video <path>.")

        cache = AnalysisCache.for_video(video_path, self.config.output.output_dir)

        console.rule("[bold blue]LocalVisionAI — Video Analysis")
        console.print(f"  [dim]Video  :[/dim] {video_path}")
        console.print(f"  [dim]Backend:[/dim] {self.config.model.backend} / {self.config.model.model_id}")
        console.print(f"  [dim]Cache  :[/dim] {cache.cache_path}")
        console.print()

        adapter = self._build_adapter()
        anchor_frame = None

        try:
            await adapter.load()

            # ── Phase 1: Frame + Audio Analysis ─────────────────────────
            if cache.is_complete:
                console.print("[green]✓[/green] Cached analysis found — skipping re-analysis.")
                anchor_frame = self._load_thumbnail(cache)
            else:
                anchor_frame = await self._run_analysis(adapter, cache)

            # ── Phase 2: Summary ─────────────────────────────────────────
            if cache.summary:
                console.print("[green]✓[/green] Using cached summary.")
                summary = cache.summary
            else:
                summary = await self._generate_summary(adapter, cache, anchor_frame)
                cache.set_summary(summary)
                cache.save()

            self._print_summary(summary, cache)

            # ── Phase 3: Interactive Q&A ──────────────────────────────────
            if not skip_qa:
                await self._run_qa(adapter, cache, anchor_frame)

        finally:
            await adapter.unload()

    # ------------------------------------------------------------------
    # Adapter construction (mirrors Pipeline._build_adapter logic)
    # ------------------------------------------------------------------

    def _build_adapter(self):
        cfg = self.config.model
        if cfg.backend == "ollama":
            kwargs: dict = {"model": cfg.model_id}
        else:
            kwargs = {"model_id": cfg.model_id}
        if cfg.api_key:
            kwargs["api_key"] = cfg.api_key
        if cfg.api_base:
            kwargs["host" if cfg.backend == "ollama" else "api_base"] = cfg.api_base
        if cfg.backend in ("openai", "anthropic", "gemini", "lmstudio"):
            kwargs["max_new_tokens"] = cfg.max_tokens
        if cfg.load_in_4bit:
            kwargs["load_in_4bit"] = True
        return get_adapter(cfg.backend, **kwargs)

    # ------------------------------------------------------------------
    # Phase 1: Analysis
    # ------------------------------------------------------------------

    async def _run_analysis(self, adapter, cache: AnalysisCache):
        """Run frame inference and (optionally) audio transcription.

        Returns the first sampled frame as a PIL Image for later use as the
        Q&A/summary anchor.
        """
        from PIL import Image

        cache.set_metadata(
            video_path=self.config.source.path,
            model_id=adapter.model_id,
            backend=adapter.backend_name,
            audio_enabled=self.config.audio.enabled,
        )

        # ── Audio extraction ────────────────────────────────────────────
        audio_segmenter = None

        if self.config.audio.enabled:
            audio_segmenter = await self._extract_audio()

        supports_native_audio = bool(getattr(adapter, "supports_audio", False))

        # ── Frame analysis ──────────────────────────────────────────────
        source = build_source(self.config.source)
        sampler = build_sampler(self.config.sampling)
        frame_count_hint = self._estimate_frame_count()
        anchor_frame: Optional[Image.Image] = None

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task(
                "Analysing frames...",
                total=frame_count_hint,
            )

            async with source:
                async for frame, ts in source.stream():
                    if not sampler.should_process(frame, ts):
                        continue

                    # First frame → save as thumbnail for future sessions
                    if anchor_frame is None:
                        anchor_frame = frame.copy()
                        anchor_frame.save(str(cache.thumbnail_path), "JPEG", quality=85)

                    # Per-frame audio window — sent natively if adapter supports it
                    audio_chunk = None
                    audio_mode_label: Optional[str] = None

                    if audio_segmenter is not None:
                        audio_chunk = audio_segmenter.get_chunk(
                            ts, self.config.audio.window_seconds
                        )
                        if audio_chunk.is_empty:
                            audio_chunk = None

                    user_prompt, system_prompt = build_prompt(
                        self.config.prompt, transcript=None
                    )

                    tokens: list[str] = []
                    t0 = time.perf_counter() * 1000
                    if audio_chunk is not None and supports_native_audio:
                        audio_mode_label = "native"
                        async for token in adapter.infer_with_audio(
                            frame, audio_chunk, user_prompt, system_prompt
                        ):
                            tokens.append(token)
                    else:
                        async for token in adapter.infer(frame, user_prompt, system_prompt):
                            tokens.append(token)
                    elapsed = time.perf_counter() * 1000 - t0

                    result = InferenceResult(
                        timestamp=ts,
                        description="".join(tokens).strip(),
                        model_id=adapter.model_id,
                        backend=adapter.backend_name,
                        latency_ms=elapsed,
                        token_count=len(tokens),
                        audio_transcript=None,
                        audio_mode=audio_mode_label,
                    )
                    cache.add_frame(result)
                    progress.advance(task)

                    # Incremental save every 10 frames (checkpoint on crash)
                    if len(cache.frames) % 10 == 0:
                        cache.save()

        cache.mark_complete()
        cache.save()

        n = len(cache.frames)
        console.print(f"[green]✓[/green] Analysis complete — {n} frame{'s' if n != 1 else ''} processed.")
        console.print(f"  Saved to: {cache.cache_path}")
        if anchor_frame is None:
            anchor_frame = self._blank_frame()
        return anchor_frame

    # ------------------------------------------------------------------
    # Audio helpers
    # ------------------------------------------------------------------

    async def _extract_audio(self):
        """Extract full audio track and return an AudioSegmenter for native delivery.

        Returns (AudioSegmenter | None).
        """
        from localvisionai.audio import AudioSegmenter, FfmpegAudioExtractor

        with console.status("[cyan]Extracting audio track via ffmpeg...[/cyan]"):
            extractor = FfmpegAudioExtractor(self.config)
            samples = await asyncio.to_thread(extractor.extract)

        if samples is None or len(samples) == 0:
            console.print("[yellow]⚠[/yellow]  No audio track found — skipping audio analysis.")
            return None

        segmenter = AudioSegmenter(
            samples,
            sample_rate=self.config.audio.sample_rate,
            channels=self.config.audio.channels,
        )
        duration = segmenter.duration
        console.print(
            f"[green]✓[/green] Audio extracted — {duration:.1f}s at "
            f"{self.config.audio.sample_rate} Hz"
        )
        return segmenter

    # ------------------------------------------------------------------
    # Phase 2: Summary
    # ------------------------------------------------------------------

    async def _generate_summary(self, adapter, cache: AnalysisCache, anchor_frame) -> str:
        """Build a comprehensive summary by asking the model to synthesise all findings."""
        console.print()

        with console.status("[cyan]Generating summary...[/cyan]"):
            prompt = self._build_summary_prompt(cache)
            tokens: list[str] = []
            async for token in adapter.infer(anchor_frame, prompt, system_prompt=None):
                tokens.append(token)

        return "".join(tokens).strip()

    def _build_summary_prompt(self, cache: AnalysisCache) -> str:
        meta = cache.metadata
        n = meta.get("total_frames_analyzed", len(cache.frames))
        lines = [
            f"You have analysed a video ({meta.get('video_name', 'unknown')}) "
            f"by examining {n} sampled frames and (optionally) transcribing the audio.",
            "",
        ]

        if cache.full_audio_transcript:
            lines += [
                "FULL AUDIO TRANSCRIPT:",
                cache.full_audio_transcript,
                "",
            ]

        lines.append("FRAME-BY-FRAME DESCRIPTIONS (timestamp → description):")
        for rec in cache.frames:
            ts_str = _fmt_ts(rec.timestamp)
            entry = f"  [{ts_str}] {rec.description}"
            if rec.audio_transcript:
                entry += f'  (audio: "{rec.audio_transcript}")'
            lines.append(entry)

        lines += [
            "",
            "Based on all of the above, write a comprehensive summary of the video.",
            "Cover: the main topic, key events in chronological order, important spoken content,",
            "notable visual details, and an overall conclusion.",
            "Be specific and reference timestamps where helpful.",
        ]

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Phase 3: Interactive Q&A
    # ------------------------------------------------------------------

    async def _run_qa(self, adapter, cache: AnalysisCache, anchor_frame) -> None:
        """Interactive question-answering loop grounded in the cached analysis."""
        console.print()
        console.rule("[bold cyan]Q&A Mode")
        console.print(
            "  Ask anything about the video. "
            "Type [bold]exit[/bold], [bold]quit[/bold], or press Ctrl-C to stop.\n"
        )

        qa_context = _build_qa_context(cache)

        while True:
            try:
                question = await asyncio.to_thread(
                    _prompt_user, "[bold cyan]>[/bold cyan] "
                )
            except (EOFError, KeyboardInterrupt):
                break

            question = question.strip()
            if not question:
                continue
            if question.lower() in ("exit", "quit", "q", "bye", "done"):
                console.print("[dim]Goodbye![/dim]")
                break

            full_prompt = (
                f"{qa_context}\n\n"
                f"USER QUESTION: {question}\n\n"
                "Answer the question based on the video analysis context above. "
                "Be specific and reference timestamps when relevant."
            )

            console.print()
            tokens: list[str] = []
            async for token in adapter.infer(anchor_frame, full_prompt, system_prompt=None):
                tokens.append(token)
                sys.stdout.write(token)
                sys.stdout.flush()
            sys.stdout.write("\n\n")
            sys.stdout.flush()

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _print_summary(summary: str, cache: AnalysisCache) -> None:
        meta = cache.metadata
        title = f"Summary — {meta.get('video_name', 'video')}"
        console.print()
        console.print(Panel(summary, title=title, border_style="cyan"))
        console.print()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _estimate_frame_count(self) -> Optional[int]:
        """Best-effort estimate of how many frames will be sampled (for the progress bar)."""
        try:
            import json as _json
            import subprocess
            path = self.config.source.path
            result = subprocess.run(
                [
                    "ffprobe", "-v", "quiet",
                    "-print_format", "json",
                    "-show_streams", str(path),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            data = _json.loads(result.stdout)
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    dur = float(stream.get("duration", 0) or 0)
                    if dur > 0:
                        return max(1, int(dur * self.config.sampling.fps))
        except Exception:
            pass
        return None

    @staticmethod
    def _load_thumbnail(cache: AnalysisCache):
        """Load the saved thumbnail as a PIL Image, falling back to a blank frame."""
        if cache.thumbnail_path.exists():
            try:
                from PIL import Image
                return Image.open(cache.thumbnail_path).convert("RGB")
            except Exception:
                pass
        return AnalysisPipeline._blank_frame()

    @staticmethod
    def _blank_frame():
        """Return a small grey placeholder PIL Image."""
        from PIL import Image
        return Image.new("RGB", (64, 64), color=(200, 200, 200))


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _fmt_ts(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _build_qa_context(cache: AnalysisCache, max_frames: int = _QA_MAX_FRAMES) -> str:
    """Build the context block injected into every Q&A prompt.

    For long videos the frame timeline is uniformly sub-sampled to keep the
    prompt within a reasonable token budget.
    """
    lines = ["=== VIDEO ANALYSIS CONTEXT ===", ""]

    meta = cache.metadata
    if meta.get("video_name"):
        lines += [f"Video: {meta['video_name']}", ""]

    if cache.summary:
        lines += ["SUMMARY:", cache.summary, ""]

    if cache.full_audio_transcript:
        lines += [
            "FULL AUDIO TRANSCRIPT:",
            cache.full_audio_transcript,
            "",
        ]

    frames = cache.frames
    if frames:
        if len(frames) > max_frames:
            step = len(frames) / max_frames
            frames = [frames[int(i * step)] for i in range(max_frames)]
            lines.append(
                f"FRAME TIMELINE (uniformly sampled to {max_frames} of "
                f"{len(cache.frames)} frames):"
            )
        else:
            lines.append("FRAME TIMELINE:")

        for rec in frames:
            ts = _fmt_ts(rec.timestamp)
            entry = f"  [{ts}] {rec.description}"
            if rec.audio_transcript:
                entry += f'  | audio: "{rec.audio_transcript}"'
            lines.append(entry)

    return "\n".join(lines)


def _prompt_user(rich_markup: str) -> str:
    """Print a Rich-formatted prompt and read a line from stdin."""
    from rich.console import Console as _Console
    _Console().print(rich_markup, end="")
    return input()
