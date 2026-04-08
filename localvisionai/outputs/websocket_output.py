"""WebSocket output handler — broadcasts InferenceResults to subscribed queues.

Used by the API server to stream live results to WebSocket clients without
coupling the pipeline directly to FastAPI or WebSocket machinery.

Pattern:
    handler = WebSocketOutput()
    q = handler.subscribe()           # called per WS client connection
    await Pipeline(config).run(extra_handlers=[handler])
    # consumer loop:
    while (msg := await q.get()) is not None:
        await websocket.send_json(msg)
"""

from __future__ import annotations

import asyncio
from typing import Optional

from localvisionai.adapters.base import InferenceResult
from localvisionai.utils.logging import get_logger
from .base import AbstractOutputHandler

logger = get_logger(__name__)


class WebSocketOutput(AbstractOutputHandler):
    """
    Output handler that pushes serialised InferenceResult dicts to all
    subscriber asyncio.Queues.

    Each WebSocket client connection should call subscribe() to get its own
    queue, and unsubscribe() when the connection closes.

    The sentinel value None is placed on all queues when close() is called,
    signalling end-of-stream to consumers.
    """

    def __init__(self) -> None:
        self._queues: list[asyncio.Queue] = []
        self._job_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    def subscribe(self) -> asyncio.Queue:
        """Register a new subscriber queue and return it."""
        q: asyncio.Queue = asyncio.Queue()
        self._queues.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Remove a subscriber queue (call when WS connection closes)."""
        try:
            self._queues.remove(q)
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # AbstractOutputHandler interface
    # ------------------------------------------------------------------

    async def open(self, job_id: str) -> None:
        self._job_id = job_id
        logger.debug(f"WebSocketOutput open — job_id={job_id}")

    async def handle(self, result: InferenceResult) -> None:
        """Broadcast one result to all currently-subscribed queues."""
        payload = {"type": "result", "data": result.to_dict()}
        for q in list(self._queues):
            await q.put(payload)

    async def close(self) -> None:
        """Signal end-of-stream to all subscribers by enqueuing None."""
        logger.debug(f"WebSocketOutput close — job_id={self._job_id}, subscribers={len(self._queues)}")
        for q in list(self._queues):
            await q.put(None)
        self._queues.clear()
