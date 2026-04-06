"""Unit tests for model adapters (with mocked backends)."""

from __future__ import annotations

import pytest
from PIL import Image
from unittest.mock import AsyncMock, MagicMock, patch

from localvisionai.adapters.base import AbstractModelAdapter, InferenceResult
from localvisionai.adapters.registry import get_adapter, REGISTRY


def make_frame() -> Image.Image:
    return Image.new("RGB", (224, 224), color=(128, 128, 128))


# ─────────────────────────────────────────────────────────────────────────────
# InferenceResult
# ─────────────────────────────────────────────────────────────────────────────

class TestInferenceResult:

    def test_to_dict_has_required_keys(self):
        result = InferenceResult(
            timestamp=5.0,
            description="A person walks in.",
            model_id="gemma3",
            backend="ollama",
            latency_ms=1240.0,
            token_count=5,
            raw_tokens=["A ", "person ", "walks ", "in", "."],
        )
        d = result.to_dict()
        assert "timestamp" in d
        assert "description" in d
        assert "model_id" in d
        assert "backend" in d
        assert "latency_ms" in d
        assert "token_count" in d
        # raw_tokens should NOT be in the dict (too verbose for export)
        assert "raw_tokens" not in d

    def test_to_dict_rounds_latency(self):
        result = InferenceResult(
            timestamp=0.0,
            description="Test.",
            model_id="test",
            backend="ollama",
            latency_ms=1234.5678,
            token_count=1,
        )
        assert result.to_dict()["latency_ms"] == 1234.6


# ─────────────────────────────────────────────────────────────────────────────
# Adapter Registry
# ─────────────────────────────────────────────────────────────────────────────

class TestAdapterRegistry:

    def test_get_ollama_adapter(self):
        adapter = get_adapter("ollama", model="gemma3")
        assert adapter is not None
        assert adapter.backend_name == "ollama"
        assert adapter.model_id == "gemma3"

    def test_get_transformers_adapter(self):
        adapter = get_adapter("transformers", model_id="Qwen/Qwen2-VL-7B-Instruct")
        assert adapter is not None
        assert adapter.backend_name == "transformers"

    def test_unknown_backend_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_adapter("unknown_backend", model="test")


# ─────────────────────────────────────────────────────────────────────────────
# OllamaAdapter (mocked)
# ─────────────────────────────────────────────────────────────────────────────

class TestOllamaAdapter:

    @pytest.mark.asyncio
    async def test_load_raises_when_model_not_found(self):
        from localvisionai.adapters.ollama_adapter import OllamaAdapter
        from localvisionai.exceptions import ModelNotFoundError

        adapter = OllamaAdapter(model="nonexistent-model-xyz")

        # Mock the Ollama client to return an empty model list
        mock_client = AsyncMock()
        mock_model = MagicMock()
        mock_model.model = "other-model"
        mock_client.list.return_value = MagicMock(models=[mock_model])
        adapter._client = mock_client

        with patch("localvisionai.adapters.ollama_adapter.ollama") as mock_ollama:
            mock_ollama.AsyncClient.return_value = mock_client

            with pytest.raises(ModelNotFoundError, match="not found in Ollama"):
                await adapter.load()

    @pytest.mark.asyncio
    async def test_infer_yields_tokens(self):
        from localvisionai.adapters.ollama_adapter import OllamaAdapter

        adapter = OllamaAdapter(model="gemma3")

        # Build mock streaming response
        async def mock_chat(**kwargs):
            for word in ["A ", "person ", "walks."]:
                chunk = MagicMock()
                chunk.message.content = word
                yield chunk

        mock_client = AsyncMock()
        mock_client.chat.return_value = mock_chat()
        adapter._client = mock_client

        frame = make_frame()
        tokens = []
        async for token in adapter.infer(frame, "Describe this frame."):
            tokens.append(token)

        assert len(tokens) == 3
        assert "".join(tokens) == "A person walks."

    @pytest.mark.asyncio
    async def test_infer_raises_when_not_loaded(self):
        from localvisionai.adapters.ollama_adapter import OllamaAdapter
        from localvisionai.exceptions import ModelInferenceError

        adapter = OllamaAdapter(model="gemma3")
        # _client is None (not loaded)

        with pytest.raises(ModelInferenceError, match="not loaded"):
            async for _ in adapter.infer(make_frame(), "test"):
                pass

    def test_preferred_resolution_is_tuple(self):
        from localvisionai.adapters.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter(model="gemma3")
        res = adapter.preferred_resolution
        assert isinstance(res, tuple)
        assert len(res) == 2
        assert all(isinstance(x, int) for x in res)
