
from __future__ import annotations
import asyncio
import logging
from typing import Any

from engines import mathematics, scientific

log = logging.getLogger("psai.engines")

class EngineError(Exception):
    """Raised when an engine operation fails or is rejected by the sandbox."""
    pass

ENGINES = {
    "math": mathematics.OPERATIONS,
    "science": scientific.OPERATIONS,
}

class EngineHub:
    def __init__(self, settings):
        self.settings = settings
        self.healthy = True

    async def start(self):
        pass

    async def stop(self):
        pass

    async def run(self, engine: str, operation: str, args: dict[str, Any]) -> Any:
        if engine not in ENGINES:
            raise EngineError(f"Unknown engine: {engine}")
        ops = ENGINES[engine]
        if operation not in ops:
            raise EngineError(f"Unknown operation '{operation}' for engine '{engine}'")
        op_def = ops[operation]
        try:
            return op_def.fn(args)
        except Exception as exc:
            if isinstance(exc, EngineError):
                raise
            raise EngineError(str(exc)) from exc
          
