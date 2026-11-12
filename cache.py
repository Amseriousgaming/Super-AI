"""Small in-process LRU cache with optional time-to-live.

Used for deterministic engine results and external research lookups.
Answers from AI models are never cached.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Hashable, Optional


class TTLCache:
    def __init__(self, maxsize: int = 256, ttl: Optional[float] = None):
        self.maxsize = maxsize
        self.ttl = ttl
        self._data: "OrderedDict[Hashable, tuple[float, Any]]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: Hashable, default: Any = None) -> Any:
        item = self._data.get(key)
        if item is None:
            self.misses += 1
            return default
        stamp, value = item
        if self.ttl is not None and time.monotonic() - stamp > self.ttl:
            del self._data[key]
            self.misses += 1
            return default
        self._data.move_to_end(key)
        self.hits += 1
        return value

    def set(self, key: Hashable, value: Any) -> None:
        if self.maxsize <= 0:
            return
        self._data[key] = (time.monotonic(), value)
        self._data.move_to_end(key)
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)
