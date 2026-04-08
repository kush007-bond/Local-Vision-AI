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

    async def run(self, extra_handlers: Optional[list] = None) -> None:
        """Run the full pipeline end-to-end.

        Args:
            extra_handlers: Additional AbstractOutputHandler instances injected
                            at runtime (e.g. WebSocketOutput from the API server).
                            These are appended after the config-based handlers.
        """
        setup_logging()
        logger.info(f"Pipeline starting — job_id={self.job_id}")

        # Build components
        source = build_source(self.config.source)
        sampler = build_sampler(self.config.sampling)

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
            item = await queue.get()

            if item is _SENTINEL:
                queue.task_done()
                break

            frame, ts = item

            for attempt in range(max(1, self.config.pipeline.max_retries)):
                try:
                    user_prompt, system_prompt = build_prompt(
                        self.config.prompt,
                        context_summary=context.get_summary(),
                    )

                    tokens: list[str] = []
                    start_ms = time.perf_counter() * 1000

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
        Auto-adjust sampler FPS if model is slower than the target rate.
        Only applies to UniformSampler on CPU-only machines.
        """
        from localvisionai.sampling.uniform_sampler import UniformSampler
        # Will be bound after build_sampler() is called; accessed via runtime attribute
        # This is set after sampler construction in run()
        # For full implementation, the sampler reference would be stored on self
        pass
