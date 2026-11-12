"""A small pool of worker PROCESSES for engine calls.

Why processes: symbolic maths can run forever or eat all memory (for example
`9**9**9**9`). A thread cannot be stopped, a process can. Each call has a hard
timeout; when it expires the worker is killed and replaced. On Linux every
worker also gets a memory cap (address-space limit on top of its baseline).
"""
from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Optional

from .common import EngineError, to_jsonable

log = logging.getLogger("psai.sandbox")


def _vm_size_bytes() -> Optional[int]:
    try:
        with open("/proc/self/statm") as fh:
            return int(fh.read().split()[0]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, AttributeError):
        return None


def _worker_main(conn, memory_mb: int) -> None:  # runs in the child process
    from engines import mathematics, scientific  # warm the imports before limits apply

    try:
        import resource

        base = _vm_size_bytes()
        if base:
            limit = base + memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except Exception:  # noqa: BLE001  (non-Linux platforms)
        pass

    registry = {"math": mathematics.OPERATIONS, "science": scientific.OPERATIONS}
    conn.send(("ready", None))
    while True:
        try:
            message = conn.recv()
        except EOFError:
            break
        if message is None:
            break
        engine, operation, args = message
        try:
            result = registry[engine][operation].fn(args)
            conn.send(("ok", to_jsonable(result)))
        except EngineError as exc:
            conn.send(("error", str(exc)))
        except MemoryError:
            conn.send(("error", "The computation needed too much memory."))
        except RecursionError:
            conn.send(("error", "The expression is too deeply nested."))
        except Exception as exc:  # noqa: BLE001
            conn.send(("error", f"{type(exc).__name__}: {exc}"))


@dataclass
class _Worker:
    proc: "mp.process.BaseProcess"
    conn: "mp.connection.Connection"


class SandboxPool:
    def __init__(self, size: int, timeout: float, memory_mb: int):
        self.size, self.timeout, self.memory_mb = size, timeout, memory_mb
        self._ctx = mp.get_context("spawn")
        self._idle: "asyncio.Queue[_Worker]" = asyncio.Queue()
        self._all: list[_Worker] = []
        self._closed = False

    @property
    def healthy(self) -> bool:
        return not self._closed and any(w.proc.is_alive() for w in self._all)

    async def start(self) -> None:
        # Keep BLAS/OpenMP from spawning many threads in every worker.
        for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(var, "1")
        workers = await asyncio.gather(*[self._spawn() for _ in range(self.size)])
        for w in workers:
            self._all.append(w)
            self._idle.put_nowait(w)

    async def _spawn(self) -> _Worker:
        loop = asyncio.get_running_loop()
        parent, child = self._ctx.Pipe()
        proc = self._ctx.Process(target=_worker_main, args=(child, self.memory_mb), daemon=True)
        proc.start()
        child.close()
        ready = await loop.run_in_executor(None, parent.poll, 60)
        if not ready or parent.recv()[0] != "ready":
            proc.kill()
            raise RuntimeError("An engine worker failed to start.")
        return _Worker(proc, parent)

    def _kill(self, worker: _Worker) -> None:
        try:
            worker.proc.kill()
            worker.proc.join(timeout=2)
            worker.conn.close()
        except Exception:  # noqa: BLE001
            pass

    async def _replace(self, worker: _Worker) -> _Worker:
        self._kill(worker)
        if worker in self._all:
            self._all.remove(worker)
        fresh = await self._spawn()
        self._all.append(fresh)
        return fresh

    async def run(self, engine: str, operation: str, args: dict, timeout: Optional[float] = None) -> dict:
        timeout = timeout or self.timeout
        loop = asyncio.get_running_loop()
        try:
            worker = await asyncio.wait_for(self._idle.get(), timeout)
        except asyncio.TimeoutError:
            raise EngineError("All computation workers are busy. Try again shortly.") from None

        replace = False
        try:
            await loop.run_in_executor(None, worker.conn.send, (engine, operation, args))
            if not await loop.run_in_executor(None, worker.conn.poll, timeout):
                replace = True
                raise EngineError(f"The computation took longer than {timeout:g} seconds and was stopped.")
            status, payload = await loop.run_in_executor(None, worker.conn.recv)
        except (EOFError, BrokenPipeError, OSError):
            replace = True
            raise EngineError("The computation was stopped (it probably needed too much memory).") from None
        except BaseException:  # includes cancellation when the client disconnects
            replace = True
            raise
        finally:
            if replace and not self._closed:
                try:
                    worker = await asyncio.shield(self._replace(worker))
                except Exception:  # noqa: BLE001
                    log.exception("Could not replace an engine worker")
                    worker = None
            if worker is not None:
                self._idle.put_nowait(worker)
        if status == "error":
            raise EngineError(payload)
        return payload

    async def close(self) -> None:
        self._closed = True
        for w in list(self._all):
            try:
                w.conn.send(None)
            except Exception:  # noqa: BLE001
                pass
            self._kill(w)
        self._all.clear()
