"""Computation engines and the hub that runs them safely.

    hub = EngineHub(settings)
    await hub.start()
    await hub.run("math", "differentiate", {"expression": "x^3", "variable": "x"})

Add an engine by creating a module with an `OPERATIONS` dict and registering it in ENGINES
(and in the worker registry in sandbox.py).
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from Services.cache import TTLCache

from . import mathematics, scientific
from .common import EngineError, OperationSpec, to_jsonable
from .sandbox import SandboxPool

log = logging.getLogger("psai.engines")

ENGINES: dict[str, dict[str, OperationSpec]] = {
    "math": mathematics.OPERATIONS,
    "science": scientific.OPERATIONS,
}

__all__ = ["ENGINES", "EngineError", "EngineHub", "OperationSpec"]


class EngineHub:
    def __init__(self, settings):
        self.settings = settings
        self._pool: Optional[SandboxPool] = None
        self._cache = TTLCache(maxsize=settings.engine_cache_size, ttl=3600)

    async def start(self) -> None:
        if self.settings.engine_sandbox:
            self._pool = SandboxPool(self.settings.engine_workers, self.settings.engine_timeout_seconds,
                                     self.settings.engine_memory_mb)
            await self._pool.start()
            log.info("Engine sandbox started with %d workers", self.settings.engine_workers)
        else:
            log.warning("ENGINE_SANDBOX=false: engines run in-process without a hard timeout. Development only.")

    async def stop(self) -> None:
        if self._pool:
            await self._pool.close()

    @property
    def healthy(self) -> bool:
        return self._pool.healthy if self._pool else True

    def operations(self, engine: str) -> dict[str, OperationSpec]:
        if engine not in ENGINES:
            raise EngineError(f"Unknown engine '{engine}'. Use one of: {', '.join(ENGINES)}.")
        return ENGINES[engine]

    async def run(self, engine: str, operation: str, args: dict) -> dict:
        ops = self.operations(engine)
        if operation not in ops:
            raise EngineError(f"Unknown {engine} operation '{operation}'. Available: {', '.join(ops)}.")
        if not isinstance(args, dict):
            raise EngineError("'args' must be a JSON object.")
        try:
            key = (engine, operation, json.dumps(args, sort_keys=True, default=str))
        except (TypeError, ValueError):
            raise EngineError("'args' must be JSON-serialisable.") from None
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        if self._pool:
            result = await self._pool.run(engine, operation, args)
        else:
            timeout = self.settings.engine_timeout_seconds
            try:
                result = to_jsonable(await asyncio.wait_for(asyncio.to_thread(ops[operation].fn, args), timeout))
            except asyncio.TimeoutError:
                raise EngineError(f"The computation took longer than {timeout:g} seconds.") from None
        self._cache.set(key, result)
        return result
