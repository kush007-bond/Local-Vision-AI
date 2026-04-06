"""Image frame utilities — resize, encode, convert."""

from __future__ import annotations

import base64
import io
from typing import Optional, Tuple

from PIL import Image


def resize_frame(
    image: Image.Image,
    target_size: Tuple[int, int],
    resample: int = Image.LANCZOS,
) -> Image.Image:
    """
    Resize a frame to (width, height) using high-quality LANCZOS resampling.
    Maintains the original mode (RGB, RGBA, etc.).
    """
    if image.size == target_size:
        return image
    return image.resize(target_size, resample=resample)


def to_rgb(image: Image.Image) -> Image.Image:
    """Convert any PIL image to RGB (drops alpha channel if present)."""
    if image.mode == "RGB":
        return image
    return image.convert("RGB")


def encode_to_base64(image: Image.Image, format: str = "JPEG", quality: int = 85) -> str:
    """
    Encode a PIL Image to a base64 string suitable for API payloads.

    Args:
        image: Source PIL image (will be converted to RGB if needed).
        format: Output format — 'JPEG' (smaller) or 'PNG' (lossless).
        quality: JPEG quality (1-95). Ignored for PNG.

    Returns:
        Base64-encoded string (no data: URI prefix).
    """
    image = to_rgb(image)
    buf = io.BytesIO()
    save_kwargs: dict = {"format": format}
    if format.upper() == "JPEG":
        save_kwargs["quality"] = quality
        save_kwargs["optimize"] = True
    image.save(buf, **save_kwargs)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def encode_to_bytes(image: Image.Image, format: str = "JPEG", quality: int = 85) -> bytes:
    """
    Encode a PIL Image to raw bytes.
    """
    image = to_rgb(image)
    buf = io.BytesIO()
    save_kwargs: dict = {"format": format}
    if format.upper() == "JPEG":
        save_kwargs["quality"] = quality
    image.save(buf, **save_kwargs)
    return buf.getvalue()


def frame_fingerprint(image: Image.Image) -> bytes:
    """
    Return a compact perceptual fingerprint of a frame (64×64 grayscale bytes).
    Used by deduplication logic to skip near-identical frames.
    """
    return image.convert("L").resize((64, 64), Image.NEAREST).tobytes()
