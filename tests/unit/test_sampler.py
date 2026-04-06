"""Unit tests for frame samplers."""

from __future__ import annotations

import pytest
from PIL import Image

from localvisionai.sampling.uniform_sampler import UniformSampler
from localvisionai.sampling.keyframe_sampler import KeyframeSampler


def make_frame(width: int = 224, height: int = 224) -> Image.Image:
    """Create a minimal RGB frame for testing."""
    return Image.new("RGB", (width, height), color=(100, 150, 200))


# ─────────────────────────────────────────────────────────────────────────────
# UniformSampler
# ─────────────────────────────────────────────────────────────────────────────

class TestUniformSampler:

    def test_first_frame_always_processed(self):
        sampler = UniformSampler(fps=1.0)
        assert sampler.should_process(make_frame(), 0.0) is True

    def test_frame_within_interval_rejected(self):
        sampler = UniformSampler(fps=1.0)
        sampler.should_process(make_frame(), 0.0)  # First accepted
        assert sampler.should_process(make_frame(), 0.5) is False

    def test_frame_at_interval_boundary_accepted(self):
        sampler = UniformSampler(fps=1.0)
        sampler.should_process(make_frame(), 0.0)
        assert sampler.should_process(make_frame(), 1.0) is True

    def test_frame_past_interval_accepted(self):
        sampler = UniformSampler(fps=1.0)
        sampler.should_process(make_frame(), 0.0)
        assert sampler.should_process(make_frame(), 1.5) is True

    def test_half_fps_intervals(self):
        sampler = UniformSampler(fps=0.5)  # One frame every 2 seconds
        assert sampler.should_process(make_frame(), 0.0) is True
        assert sampler.should_process(make_frame(), 1.0) is False
        assert sampler.should_process(make_frame(), 2.0) is True

    def test_two_fps_intervals(self):
        sampler = UniformSampler(fps=2.0)  # Two frames per second
        assert sampler.should_process(make_frame(), 0.0) is True
        assert sampler.should_process(make_frame(), 0.4) is False
        assert sampler.should_process(make_frame(), 0.5) is True
        assert sampler.should_process(make_frame(), 0.9) is False
        assert sampler.should_process(make_frame(), 1.0) is True

    def test_reset_clears_state(self):
        sampler = UniformSampler(fps=1.0)
        sampler.should_process(make_frame(), 5.0)  # Advance state
        sampler.reset()
        # After reset, first frame at any timestamp should be accepted
        assert sampler.should_process(make_frame(), 5.5) is True

    def test_invalid_fps_raises(self):
        with pytest.raises(ValueError):
            UniformSampler(fps=0)
        with pytest.raises(ValueError):
            UniformSampler(fps=-1.0)


# ─────────────────────────────────────────────────────────────────────────────
# KeyframeSampler
# ─────────────────────────────────────────────────────────────────────────────

class TestKeyframeSampler:

    def test_frame_with_keyframe_flag_accepted(self):
        sampler = KeyframeSampler()
        frame = make_frame()
        frame._is_keyframe = True
        assert sampler.should_process(frame, 0.0) is True

    def test_frame_without_keyframe_flag_rejected(self):
        sampler = KeyframeSampler()
        frame = make_frame()
        frame._is_keyframe = False
        assert sampler.should_process(frame, 0.0) is False

    def test_frame_with_no_attribute_defaults_to_true(self):
        """When source doesn't set _is_keyframe, all frames pass (safe fallback)."""
        sampler = KeyframeSampler()
        frame = make_frame()  # No _is_keyframe attribute
        assert sampler.should_process(frame, 0.0) is True

    def test_min_interval_respected(self):
        sampler = KeyframeSampler(min_interval=2.0)
        frame = make_frame()
        frame._is_keyframe = True
        assert sampler.should_process(frame, 0.0) is True
        # Even if next frame is a keyframe, min_interval blocks it
        frame2 = make_frame()
        frame2._is_keyframe = True
        assert sampler.should_process(frame2, 1.0) is False
        assert sampler.should_process(frame2, 2.0) is True

    def test_reset_clears_state(self):
        sampler = KeyframeSampler(min_interval=5.0)
        frame = make_frame()
        frame._is_keyframe = True
        sampler.should_process(frame, 10.0)
        sampler.reset()
        frame2 = make_frame()
        frame2._is_keyframe = True
        assert sampler.should_process(frame2, 10.5) is True
