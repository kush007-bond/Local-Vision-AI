"""
Configuration system for LocalVisionAI.

Priority order (highest wins):
    CLI flags > Environment variables (LVA_ prefix) > YAML config file > Defaults

All config models are Pydantic v2 BaseModel subclasses with full validation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

class SourceConfig(BaseModel):
    type: Literal["file", "webcam", "rtsp", "url", "screen"] = "file"
    path: Optional[str] = None
    device_index: int = 0
    rtsp_url: Optional[str] = None

    @model_validator(mode="after")
    def validate_path_for_file(self) -> "SourceConfig":
        if self.type == "file" and not self.path:
            raise ValueError("source.path is required when source.type is 'file'.")
        if self.type == "rtsp" and not self.rtsp_url:
            raise ValueError("source.rtsp_url is required when source.type is 'rtsp'.")
        return self


class ModelConfig(BaseModel):
    backend: Literal["ollama", "transformers", "llamacpp", "mlx", "vllm"] = "ollama"
    model_id: str = "gemma3"
    multi_frame: bool = False
    multi_frame_count: int = Field(4, ge=2, le=16)
    load_in_4bit: bool = False


class SamplingConfig(BaseModel):
    strategy: Literal["uniform", "scene", "keyframe", "adaptive"] = "uniform"
    fps: float = Field(1.0, gt=0, le=30)
    scene_threshold: float = Field(30.0, gt=0)
    min_interval: float = Field(0.5, ge=0)


class PromptConfig(BaseModel):
    user: str = "Describe what is happening in this frame in one sentence."
    system: Optional[str] = None
    context_mode: Literal["none", "sliding_window"] = "none"
    context_tokens: int = Field(200, ge=0)


class OutputConfig(BaseModel):
    formats: list[Literal["console", "json", "srt", "csv", "sqlite"]] = ["console", "json"]
    output_dir: str = "./output/"


class PipelineRuntimeConfig(BaseModel):
    queue_size: int = Field(8, ge=1)
    drop_policy: Literal["oldest", "newest", "none"] = "oldest"
    retry_on_error: bool = True
    max_retries: int = Field(3, ge=0)


class APIConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(8765, ge=1, le=65535)
    cors_origins: list[str] = []


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class PipelineConfig(BaseModel):
    source: SourceConfig = SourceConfig()
    model: ModelConfig = ModelConfig()
    sampling: SamplingConfig = SamplingConfig()
    prompt: PromptConfig = PromptConfig()
    output: OutputConfig = OutputConfig()
    pipeline: PipelineRuntimeConfig = PipelineRuntimeConfig()
    api: APIConfig = APIConfig()

    # -----------------------------------------------------------------------
    # Factory methods
    # -----------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load config from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """
        Build a config from LVA_* environment variables.

        Variable naming: LVA_<SECTION>_<KEY> (uppercase)
        Example: LVA_MODEL_BACKEND=transformers
        """
        env_map: dict = {}

        def _set(d: dict, keys: list[str], value: str) -> None:
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value

        for key, val in os.environ.items():
            if not key.startswith("LVA_"):
                continue
            parts = key[4:].lower().split("_", 1)
            if len(parts) == 2:
                _set(env_map, parts, val)

        return cls.model_validate(env_map)

    @classmethod
    def from_cli(cls, kwargs: dict) -> "PipelineConfig":
        """
        Build a PipelineConfig from a flat dict of CLI keyword arguments.
        Loads YAML base first if config_file is provided, then overlays CLI values.
        """
        config_file = kwargs.pop("config_file", None)
        base: dict = {}

        if config_file:
            with open(config_file, "r", encoding="utf-8") as f:
                base = yaml.safe_load(f) or {}

        # Map flat CLI args into nested structure
        source_type = kwargs.get("source", "file")
        base.setdefault("source", {})["type"] = source_type
        if kwargs.get("video"):
            base["source"]["path"] = kwargs["video"]
        if kwargs.get("device") is not None:
            base["source"]["device_index"] = kwargs["device"]
        if kwargs.get("rtsp_url"):
            base["source"]["rtsp_url"] = kwargs["rtsp_url"]

        base.setdefault("model", {})
        if kwargs.get("backend"):
            base["model"]["backend"] = kwargs["backend"]
        if kwargs.get("model"):
            base["model"]["model_id"] = kwargs["model"]

        base.setdefault("sampling", {})
        if kwargs.get("sampler"):
            base["sampling"]["strategy"] = kwargs["sampler"]
        if kwargs.get("fps") is not None:
            base["sampling"]["fps"] = kwargs["fps"]

        base.setdefault("prompt", {})
        if kwargs.get("prompt"):
            base["prompt"]["user"] = kwargs["prompt"]

        base.setdefault("output", {})
        if kwargs.get("output_formats"):
            base["output"]["formats"] = kwargs["output_formats"]
        if kwargs.get("output_dir"):
            base["output"]["output_dir"] = kwargs["output_dir"]

        return cls.model_validate(base)
