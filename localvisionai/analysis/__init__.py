"""
Offline video analysis package for LocalVisionAI.

Provides the ``localvisionai analyze`` workflow:
    1. Frame analysis — sample the video, run per-frame inference, cache results.
    2. Audio analysis — transcribe the full audio track via Whisper.
    3. Summarization — ask the model to synthesise all findings.
    4. Q&A — interactive chat grounded in the cached analysis.

Results are persisted to a JSON cache file so the expensive analysis pass
only needs to run once per video.
"""

from .cache import AnalysisCache, FrameRecord
from .pipeline import AnalysisPipeline

__all__ = ["AnalysisCache", "FrameRecord", "AnalysisPipeline"]
