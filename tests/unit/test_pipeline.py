"""Unit tests for the Pipeline (with fully mocked adapter and source)."""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from localvisionai.adapters.base import InferenceResult
from localvisionai.config import PipelineConfig, SourceConfig, ModelConfig


def make_frame() -> Image.Image:
    return Image.new("RGB", (224, 224), color=(64, 128, 192))


async def _token_gen(tokens: list[str]) -> AsyncGenerator[str, None]:
    for t in tokens:
        yield t


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineConfig:

    def test_default_config_valid(self):
        config = PipelineConfig()
        assert config.model.backend == "ollama"
        assert config.model.model_id == "gemma3"
        assert config.sampling.fps == 1.0

    def test_from_cli_file_source(self):
        config = PipelineConfig.from_cli({
            "source": "file",
            "video": "test.mp4",
            "backend": "ollama",
            "model": "gemma3",
            "fps": 2.0,
            "sampler": "uniform",
            "prompt": "Describe.",
            "output_formats": ["console"],
            "output_dir": "./out/",
            "config_file": None,
        })
        assert config.source.type == "file"
        assert config.source.path == "test.mp4"
        assert config.sampling.fps == 2.0

    def test_from_cli_webcam_source(self):
        config = PipelineConfig.from_cli({
            "source": "webcam",
            "video": None,
            "device": 1,
            "backend": "ollama",
            "model": "gemma3",
            "fps": 1.0,
            "sampler": "uniform",
            "prompt": "Describe.",
            "output_formats": ["console"],
            "output_dir": "./out/",
            "config_file": None,
        })
        assert config.source.type == "webcam"
        assert config.source.device_index == 1

    def test_file_source_without_path_raises(self):
        with pytest.raises(Exception):
            SourceConfig(type="file", path=None)

    def test_invalid_backend_raises(self):
        with pytest.raises(Exception):
            ModelConfig(backend="invalid_backend", model_id="test")


# ─────────────────────────────────────────────────────────────────────────────
# Context Window
# ─────────────────────────────────────────────────────────────────────────────

class TestContextWindow:

    def test_empty_window_returns_none(self):
        from localvisionai.prompts.memory import ContextWindow
        ctx = ContextWindow(max_tokens=100)
        assert ctx.get_summary() is None

    def test_single_entry_returned(self):
        from localvisionai.prompts.memory import ContextWindow
        ctx = ContextWindow(max_tokens=100)
        ctx.update("A person walks in.")
        assert ctx.get_summary() == "A person walks in."

    def test_multiple_entries_joined(self):
        from localvisionai.prompts.memory import ContextWindow
        ctx = ContextWindow(max_tokens=200)
        ctx.update("Scene A.")
        ctx.update("Scene B.")
        summary = ctx.get_summary()
        assert "Scene A." in summary
        assert "Scene B." in summary

    def test_evicts_oldest_on_overflow(self):
        from localvisionai.prompts.memory import ContextWindow
        # Very small budget — only fits a few words
        ctx = ContextWindow(max_tokens=5)
        ctx.update("one two three four five")  # 5 tokens, fills budget
        ctx.update("six seven eight nine ten")  # Should evict the first entry
        summary = ctx.get_summary()
        # First entry should have been evicted
        assert "one" not in summary or "six" in summary

    def test_reset_clears_entries(self):
        from localvisionai.prompts.memory import ContextWindow
        ctx = ContextWindow(max_tokens=100)
        ctx.update("Something happened.")
        ctx.reset()
        assert ctx.get_summary() is None
        assert len(ctx) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline (mocked)
# ─────────────────────────────────────────────────────────────────────────────

class TestPipeline:

    @pytest.mark.asyncio
    async def test_pipeline_calls_handler_for_each_frame(self):
        """Pipeline should call handle() on all handlers once per processed frame."""
        from localvisionai.pipeline import Pipeline

        config = PipelineConfig()
        config.source.type = "file"
        config.source.path = "fake.mp4"
        config.output.formats = ["json"]
        config.output.output_dir = "/tmp/test_pipeline/"

        pipeline = Pipeline(config)

        # Mock source that yields 3 frames
        mock_source = AsyncMock()
        mock_source.__aenter__ = AsyncMock(return_value=mock_source)
        mock_source.__aexit__ = AsyncMock(return_value=False)
        frames = [(make_frame(), float(i)) for i in range(3)]

        async def _stream():
            for frame, ts in frames:
                yield frame, ts

        mock_source.stream.return_value = _stream()

        # Mock adapter
        mock_adapter = AsyncMock()
        mock_adapter.model_id = "gemma3"
        mock_adapter.backend_name = "ollama"
        mock_adapter.preferred_resolution = (448, 448)
        mock_adapter.supports_multi_frame = False

        async def _infer(frame, prompt, system_prompt=None):
            yield "A test description."

        mock_adapter.infer.side_effect = _infer

        # Mock handler
        mock_handler = AsyncMock()

        with patch("localvisionai.pipeline.build_source", return_value=mock_source), \
             patch("localvisionai.pipeline.get_adapter", return_value=mock_adapter), \
             patch("localvisionai.pipeline.build_handlers", return_value=[mock_handler]):
            await pipeline.run()

        # Should have been called 3 times (once per sampled frame at 1 FPS with 3 second timestamps)
        assert mock_handler.handle.call_count == 3
