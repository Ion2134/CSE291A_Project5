from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Callable


class NonceCache:
    """Bounded record of nonces already accepted. ``check_and_add`` is atomic, so two concurrent
    requests carrying the same nonce can't both pass."""

    def __init__(
        self,
        ttl_s: float = 300.0,
        max_entries: int = 100_000,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self._clock = clock
        self._seen: OrderedDict[str, float] = OrderedDict()  # nonce -> expiry, oldest first
        self._lock = threading.Lock()

    def check_and_add(self, nonce: str) -> bool:
        """Return True if ``nonce`` is new (and remember it), False if it was already seen."""
        now = self._clock()
        with self._lock:
            self._prune(now)
            if nonce in self._seen:
                return False
            self._seen[nonce] = now + self.ttl_s
            while len(self._seen) > self.max_entries:
                self._seen.popitem(last=False)
            return True

    def __len__(self) -> int:
        with self._lock:
            return len(self._seen)

    def _prune(self, now: float) -> None:
        while self._seen:
            nonce, expiry = next(iter(self._seen.items()))
            if expiry > now:
                break
            self._seen.popitem(last=False)
