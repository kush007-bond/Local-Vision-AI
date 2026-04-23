"""
Pipeline — the central orchestrator of LocalVisionAI.

Implements the async producer-consumer pattern:
  - Producer: reads frames from the input source, passes them through the sampler
  - Consumer: pulls frames from the bounded queue, runs model inference, fans out to handlers

The bounded asyncio.Queue provides natural backpressure — when the model is slower
than the input, old frames are dropped according to the configured drop policy.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Optional

from localvisionai.adapters.base import InferenceResult
from localvisionai.adapters.registry import get_adapter
from localvisionai.config import PipelineConfig
from localvisionai.exceptions import ModelInferenceError
from localvisionai.inputs import build_source
from localvisionai.outputs import build_handlers
from localvisionai.outputs.json_output import JSONOutput
from localvisionai.prompts.builder import build_prompt
from localvisionai.prompts.memory import ContextWindow
from localvisionai.sampling import build_sampler
from localvisionai.utils.logging import get_logger, setup_logging
from localvisionai.utils.timing import LatencyTracker

# Audio is imported lazily inside run() so pipelines that don't enable it
# never pay the numpy/ffmpeg import cost.

logger = get_logger(__name__)

_SENTINEL = None  # Signals end of queue to consumer


class Pipeline:
    """
    The main LocalVisionAI pipeline.

    Usage:
        config = PipelineConfig.from_yaml("configs/default.yaml")
        config.source.path = "video.mp4"
        await Pipeline(config).run()
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.job_id = f"j_{uuid.uuid4().hex[:8]}"
        # Audio components — populated in run() if audio is enabled
        self._audio_segmenter = None
        self._use_native_audio = False
        # Sampler reference — stored so _maybe_adjust_fps can call record_latency
        self._sampler = None

    async def run(self, extra_handlers: Optional[list] = None, source=None) -> None:
        """Run the full pipeline end-to-end.

        Args:
            extra_handlers: Additional AbstractOutputHandler instances injected
                            at runtime (e.g. WebSocketOutput from the API server).
                            These are appended after the config-based handlers.
            source: Pre-built AbstractVideoSource. When provided, build_source()
                    is skipped. Used for BrowserCaptureSource (webcam jobs) to
                    avoid OpenCV/browser camera conflicts.
        """
        setup_logging()
        logger.info(f"Pipeline starting — job_id={self.job_id}")

        # Build components (skip if caller provided a pre-built source)
        if source is None:
            source = build_source(self.config.source)

        # ── Audio-only mode auto-configuration ────────────────────────────
        # When source type is 'audio', there is no video stream — force audio
        # transcription on and wire the correct window size into the source.
        if self.config.source.type == "audio":
            if not self.config.audio.enabled:
                from localvisionai.config import AudioConfig
                # Use model_copy to safely create an updated config copy
                self.config = self.config.model_copy(
                    update={
                        "audio": AudioConfig(
                            enabled=True,
                            mode="native",
                            window_seconds=self.config.audio.window_seconds,
                            sample_rate=self.config.audio.sample_rate,
                            channels=self.config.audio.channels,
                        )
                    }
                )
            # Rebuild source with the correct window_seconds from config
            from localvisionai.inputs.audio_source import AudioOnlySource
            source = AudioOnlySource(
                path=self.config.source.path,
                window_seconds=self.config.audio.window_seconds,
            )
        # ──────────────────────────────────────────────────────────────────


        sampler = build_sampler(self.config.sampling)
        self._sampler = sampler  # keep reference for adaptive FPS

        # Build adapter from registry
        if self.config.model.backend == "ollama":
            adapter_kwargs: dict = {"model": self.config.model.model_id}
        else:
            adapter_kwargs = {"model_id": self.config.model.model_id}

        if self.config.model.load_in_4bit:
            adapter_kwargs["load_in_4bit"] = True

        # Remote / cloud provider settings
        if self.config.model.api_key:
            adapter_kwargs["api_key"] = self.config.model.api_key
        if self.config.model.api_base:
            if self.config.model.backend == "ollama":
                # OllamaAdapter uses 'host' instead of 'api_base'
                adapter_kwargs["host"] = self.config.model.api_base
            else:
                adapter_kwargs["api_base"] = self.config.model.api_base
        if self.config.model.backend in ("openai", "anthropic", "gemini", "lmstudio"):
            adapter_kwargs["max_new_tokens"] = self.config.model.max_tokens

        adapter = get_adapter(self.config.model.backend, **adapter_kwargs)

        # Build output handlers
        handlers = build_handlers(self.config.output, self.job_id)
        if extra_handlers:
            handlers.extend(extra_handlers)

        # Set JSON metadata if JSONOutput is present
        for h in handlers:
            if isinstance(h, JSONOutput):
                h.set_metadata(
                    model_id=self.config.model.model_id,
                    backend=self.config.model.backend,
                    source=str(self.config.source.path or self.config.source.type),
                )

        # Open all handlers
        for h in handlers:
            await h.open(self.job_id)

        try:
            # Load the model
            logger.info(f"Loading model: {self.config.model.model_id} ({self.config.model.backend})")
            await adapter.load()
            logger.info("Model loaded. Starting pipeline.")

            # Audio setup — extract track, build segmenter, load Whisper if needed
            if self.config.audio.enabled:
                await self._setup_audio(adapter)

            # Open the source
            async with source:
                queue: asyncio.Queue = asyncio.Queue(maxsize=self.config.pipeline.queue_size)
                context = ContextWindow(max_tokens=self.config.prompt.context_tokens)
                latency_tracker = LatencyTracker()

                await asyncio.gather(
                    self._producer(source, sampler, queue),
                    self._consumer(adapter, queue, handlers, context, latency_tracker),
                )

        finally:
            # Close output handlers
            for h in handlers:
                await h.close()

            await adapter.unload()
            logger.info(f"Pipeline complete — job_id={self.job_id}")

    # --------------------------------------------------------------------------
    # Audio setup
    # --------------------------------------------------------------------------

    async def _setup_audio(self, adapter) -> None:
        """Extract the audio track and build a segmenter for native model delivery."""
        from localvisionai.audio import AudioSegmenter, FfmpegAudioExtractor

        logger.info("Audio enabled — extracting track via ffmpeg.")
        extractor = FfmpegAudioExtractor(self.config)
        samples = await asyncio.to_thread(extractor.extract)

        self._audio_segmenter = AudioSegmenter(
            samples,
            sample_rate=self.config.audio.sample_rate,
            channels=self.config.audio.channels,
        )

        self._use_native_audio = bool(getattr(adapter, "supports_audio", False))
        if self._use_native_audio:
            logger.info(
                f"Audio mode=native — adapter '{adapter.backend_name}' "
                "will receive raw audio alongside each frame."
            )
        else:
            logger.warning(
                f"Adapter '{adapter.backend_name}' does not declare native audio "
                "support — audio chunks will be skipped."
            )

    # --------------------------------------------------------------------------
    # Producer
    # --------------------------------------------------------------------------

    async def _producer(self, source, sampler, queue: asyncio.Queue) -> None:
        """Read frames from source, apply sampler, push to queue."""
        frames_read = 0
        frames_queued = 0
        frames_dropped = 0

        try:
            async for frame, ts in source.stream():
                frames_read += 1

                if not sampler.should_process(frame, ts):
                    continue

                # Apply drop policy if queue is full
                if queue.full():
                    if self.config.pipeline.drop_policy == "oldest":
                        try:
                            queue.get_nowait()  # Discard oldest
                            frames_dropped += 1
                        except asyncio.QueueEmpty:
                            pass
                    elif self.config.pipeline.drop_policy == "newest":
                        frames_dropped += 1
                        continue  # Discard this (newest) frame
                    # "none" → block until space available
                    elif self.config.pipeline.drop_policy == "none":
                        pass  # Falls through to queue.put() which will block

                await queue.put((frame, ts))
                frames_queued += 1

        except Exception as e:
            logger.error(f"Producer error: {e}", exc_info=True)
        finally:
            await queue.put(_SENTINEL)
            logger.info(
                f"Producer done — read={frames_read} queued={frames_queued} dropped={frames_dropped}"
            )

    # --------------------------------------------------------------------------
    # Consumer
    # --------------------------------------------------------------------------

    async def _consumer(
        self,
        adapter,
        queue: asyncio.Queue,
        handlers: list,
        context: ContextWindow,
        latency_tracker: LatencyTracker,
    ) -> None:
        """Pull frames from queue, run inference, fan out to handlers."""
        frames_processed = 0
        frames_failed = 0

        while True:
            # ── Sliding-window freeze fix ─────────────────────────────────
            # Use a timeout on queue.get() so that if the producer finishes
            # and the sentinel was already consumed (or missed), the consumer
            # does not block forever. A 60-second ceiling is generous enough
            # for even the slowest CPU-only model.
            try:
                item = await asyncio.wait_for(queue.get(), timeout=60.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "Consumer timed out waiting for next frame (60s). "
                    "Assuming producer is done — exiting consumer."
                )
                break
            # ─────────────────────────────────────────────────────────────

            if item is _SENTINEL:
                queue.task_done()
                break

            frame, ts = item

            for attempt in range(max(1, self.config.pipeline.max_retries)):
                try:
                    # Resolve optional audio chunk for this frame
                    audio_chunk = None
                    audio_mode_label: Optional[str] = None
                    if self._audio_segmenter is not None:
                        audio_chunk = self._audio_segmenter.get_chunk(
                            ts, self.config.audio.window_seconds
                        )

                    user_prompt, system_prompt = build_prompt(
                        self.config.prompt,
                        context_summary=context.get_summary(),
                        transcript=None,
                    )

                    tokens: list[str] = []
                    start_ms = time.perf_counter() * 1000

                    if (
                        audio_chunk is not None
                        and not audio_chunk.is_empty
                        and self._use_native_audio
                    ):
                        audio_mode_label = "native"
                        async for token in adapter.infer_with_audio(
                            frame, audio_chunk, user_prompt, system_prompt
                        ):
                            tokens.append(token)
                    else:
                        async for token in adapter.infer(frame, user_prompt, system_prompt):
                            tokens.append(token)

                    elapsed_ms = time.perf_counter() * 1000 - start_ms
                    description = "".join(tokens).strip()

                    result = InferenceResult(
                        timestamp=ts,
                        description=description,
                        model_id=adapter.model_id,
                        backend=adapter.backend_name,
                        latency_ms=elapsed_ms,
                        token_count=len(tokens),
                        raw_tokens=tokens,
                        audio_transcript=None,
                        audio_mode=audio_mode_label,
                    )

                    # Update context window
                    if self.config.prompt.context_mode == "sliding_window":
                        context.update(description)

                    # Fan out to all handlers
                    for handler in handlers:
                        await handler.handle(result)

                    frames_processed += 1

                    # Auto-adjust FPS if CPU-only and too slow
                    self._maybe_adjust_fps(elapsed_ms / 1000.0)

                    break  # Success — exit retry loop

                except ModelInferenceError as e:
                    if attempt + 1 < self.config.pipeline.max_retries and self.config.pipeline.retry_on_error:
                        logger.warning(f"Inference failed at {ts:.2f}s (attempt {attempt+1}): {e}. Retrying...")
                        await asyncio.sleep(0.5)
                    else:
                        logger.warning(f"Inference failed at {ts:.2f}s — skipping frame. Error: {e}")
                        frames_failed += 1
                        break

                except Exception as e:
                    logger.error(f"Unexpected error at {ts:.2f}s: {e}", exc_info=True)
                    frames_failed += 1
                    break

            queue.task_done()

        logger.info(
            f"Consumer done — processed={frames_processed} failed={frames_failed}"
        )

    def _maybe_adjust_fps(self, actual_latency_s: float) -> None:
        """
        Auto-adjust sampler FPS if the active sampler supports it.

        Currently only AdaptiveSampler implements ``record_latency``; all
        other sampler types are left untouched.
        """
        if self._sampler is None:
            return
        record = getattr(self._sampler, "record_latency", None)
        if callable(record):
            record(actual_latency_s)
