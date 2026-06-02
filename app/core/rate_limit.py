from __future__ import annotations

import time
from collections import deque
from threading import Lock
from typing import Deque, Dict, Tuple


class InMemorySlidingWindowRateLimiter:
    """
    Best-effort in-memory limiter for single-process deployments.
    For multi-instance production, replace with Redis-based limiter.
    """

    def __init__(self) -> None:
        self._events: Dict[str, Deque[float]] = {}
        self._lock = Lock()

    def allow(self, *, key: str, limit: int, window_sec: int) -> Tuple[bool, int]:
        """
        Returns (allowed, retry_after_seconds).
        """
        if limit <= 0 or window_sec <= 0:
            return True, 0

        now = time.time()
        cutoff = now - float(window_sec)
        with self._lock:
            q = self._events.get(key)
            if q is None:
                q = deque()
                self._events[key] = q

            while q and q[0] < cutoff:
                q.popleft()

            if len(q) >= int(limit):
                retry_after = int(max(1.0, float(window_sec) - (now - q[0]))) if q else int(window_sec)
                return False, retry_after

            q.append(now)
            return True, 0
