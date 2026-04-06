"""Custom exception hierarchy for LocalVisionAI.

All exceptions derive from LocalVisionAIError so callers can catch
the entire family with a single except clause if needed.
"""


class LocalVisionAIError(Exception):
    """Base exception for all LocalVisionAI errors."""


class ModelNotFoundError(LocalVisionAIError):
    """Raised when a requested model is not available on the selected backend."""


class ModelInferenceError(LocalVisionAIError):
    """Raised when the model fails to produce output for a given frame."""


class SourceOpenError(LocalVisionAIError):
    """Raised when a video source cannot be opened (file not found, RTSP unreachable, etc.)."""


class SourceReadError(LocalVisionAIError):
    """Raised when a video source drops mid-stream and cannot be recovered."""


class ConfigValidationError(LocalVisionAIError):
    """Raised when supplied configuration fails Pydantic validation."""
