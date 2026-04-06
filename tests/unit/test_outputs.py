"""Unit tests for output handlers."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile

import pytest

from localvisionai.adapters.base import InferenceResult


def make_result(ts: float = 0.0, description: str = "A test frame.") -> InferenceResult:
    return InferenceResult(
        timestamp=ts,
        description=description,
        model_id="gemma3",
        backend="ollama",
        latency_ms=500.0,
        token_count=4,
        raw_tokens=["A ", "test ", "frame", "."],
    )


# ─────────────────────────────────────────────────────────────────────────────
# ConsoleOutput
# ─────────────────────────────────────────────────────────────────────────────

class TestConsoleOutput:

    @pytest.mark.asyncio
    async def test_handle_does_not_raise(self):
        from localvisionai.outputs.console_output import ConsoleOutput
        handler = ConsoleOutput()
        await handler.open("test-job-id")
        await handler.handle(make_result())  # Should not raise
        await handler.close()

    @pytest.mark.asyncio
    async def test_verbose_mode(self):
        from localvisionai.outputs.console_output import ConsoleOutput
        handler = ConsoleOutput(verbose=True)
        await handler.open("test-job-id")
        await handler.handle(make_result())
        await handler.close()


# ─────────────────────────────────────────────────────────────────────────────
# JSONOutput
# ─────────────────────────────────────────────────────────────────────────────

class TestJSONOutput:

    @pytest.mark.asyncio
    async def test_writes_valid_json_on_close(self):
        from localvisionai.outputs.json_output import JSONOutput

        with tempfile.TemporaryDirectory() as tmpdir:
            handler = JSONOutput(output_dir=tmpdir)
            await handler.open("job-test-123")
            handler.set_metadata("gemma3", "ollama", "video.mp4")

            for i in range(3):
                await handler.handle(make_result(ts=float(i), description=f"Frame {i}"))

            await handler.close()

            output_file = os.path.join(tmpdir, "job-test-123.json")
            assert os.path.exists(output_file)

            with open(output_file, "r") as f:
                data = json.load(f)

            assert data["job_id"] == "job-test-123"
            assert data["model"] == "gemma3"
            assert len(data["frames"]) == 3
            assert data["frames"][0]["timestamp"] == 0.0
            assert data["frames"][2]["description"] == "Frame 2"

    @pytest.mark.asyncio
    async def test_creates_output_dir_if_missing(self):
        from localvisionai.outputs.json_output import JSONOutput

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = os.path.join(tmpdir, "a", "b", "c")
            handler = JSONOutput(output_dir=nested_dir)
            await handler.open("job-nested")
            await handler.handle(make_result())
            await handler.close()

            assert os.path.exists(nested_dir)
            assert os.path.exists(os.path.join(nested_dir, "job-nested.json"))

    @pytest.mark.asyncio
    async def test_frame_dict_has_required_keys(self):
        from localvisionai.outputs.json_output import JSONOutput

        with tempfile.TemporaryDirectory() as tmpdir:
            handler = JSONOutput(output_dir=tmpdir)
            await handler.open("job-keys")
            handler.set_metadata("gemma3", "ollama", "video.mp4")
            await handler.handle(make_result())
            await handler.close()

            with open(os.path.join(tmpdir, "job-keys.json")) as f:
                data = json.load(f)

            frame = data["frames"][0]
            assert "timestamp" in frame
            assert "description" in frame
            assert "latency_ms" in frame
            assert "token_count" in frame
