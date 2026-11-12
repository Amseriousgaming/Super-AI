"""The chat pipeline shared by /api/chat and /api/analyze-image.

    validate images -> resolve conversation -> choose features -> gather history and memory
    -> orchestrator -> persist -> response (JSON or Server-Sent Events)
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from ai.base import AllModelsFailed, NoModelAvailable, ProviderError
from ai.orchestrator import ChatContext, Emitter, HistoryEntry, Orchestrator, OrchestratorResult
from models import ChatRequest, ChatResponse, ImageInfoOut, SourceOut, ToolTraceOut, UsageOut
from services.image_analysis import ImageService, ImageValidationError, PreparedImage, RawImage, task_instruction
from services.memory import MemoryService

log = logging.getLogger("psai.chat")


class ServerBusy(Exception):
    pass


@dataclass
class PreparedChat:
    ctx: ChatContext
    conversation_id: str
    user_id: Optional[str]
    request: ChatRequest
    images: list[PreparedImage] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    task: Optional[str] = None


def effective_features(settings, requested: dict[str, bool]) -> set[str]:
    """Server-offered features that are on by default, plus those the client switched on (or off)."""
    offered = settings.offered_feature_set
    on = {f for f in settings.auto_feature_set if requested.get(f) is not False}
    on |= {f for f, v in requested.items() if v is True}
    return on & offered


def error_for(exc: BaseException) -> Optional[tuple[int, str, dict[str, str]]]:
    """Map known failures to (status, public message, headers). None means 'unexpected'."""
    if isinstance(exc, ImageValidationError):
        return exc.status, str(exc), {}
    if isinstance(exc, ServerBusy):
        return 503, "The server is busy with other requests. Please try again in a moment.", {"Retry-After": "10"}
    if isinstance(exc, asyncio.TimeoutError):
        return 504, "The request took too long. Try again, or ask a shorter or simpler question.", {}
    if isinstance(exc, NoModelAvailable):
        return 503, str(exc), {}
    if isinstance(exc, (AllModelsFailed, ProviderError)):
        kinds = {k for _, k in exc.failures} if isinstance(exc, AllModelsFailed) else {exc.kind}
        if kinds == {"rate_limit"}:
            return 429, ("The AI provider's rate limit was reached. Free tiers have limits; wait a little and try again, "
                         "or configure another provider."), {"Retry-After": "30"}
        if "auth" in kinds:
            return 502, "The AI provider rejected this server's credentials. The administrator must check the API key.", {}
        return 502, "The AI models are temporarily unavailable. Please try again shortly.", {}
    return None


class ChatService:
    def __init__(self, settings, memory: MemoryService, images: ImageService, orchestrator: Orchestrator):
        self.settings, self.memory, self.images, self.orchestrator = settings, memory, images, orchestrator
        self._sem = asyncio.Semaphore(settings.max_concurrent_requests)

    @asynccontextmanager
    async def _slot(self):
        try:
            await asyncio.wait_for(self._sem.acquire(), self.settings.queue_timeout_seconds)
        except asyncio.TimeoutError:
            raise ServerBusy() from None
        try:
            yield
        finally:
            self._sem.release()

    # --- preparation -------------------------------------------------------------------
    async def prepare(self, body: ChatRequest, user_id: Optional[str], *, raw_images: Optional[list[RawImage]] = None,
                      task: Optional[str] = None) -> PreparedChat:
        payloads = body.all_images()
        raw_images = raw_images or []
        if len(payloads) + len(raw_images) > self.settings.max_images:
            raise ImageValidationError(f"At most {self.settings.max_images} images per request.", 413)
        images = await self.images.prepare_payloads(payloads) if payloads else []
        if raw_images:
            images += await self.images.prepare_raw(raw_images)

        warnings: list[str] = []
        task_hint = ""
        if task is not None:
            task_hint, warning = task_instruction(task)
            if warning:
                warnings.append(warning)
            if not images:
                raise ImageValidationError("analyze-image needs at least one image.")

        conversation_id = body.conversation_id or uuid.uuid4().hex[:16]
        features = effective_features(self.settings, body.features)

        history = [HistoryEntry(h.role, h.content, h.has_images)
                   for h in body.history[-self.settings.max_history_items:] if h.content or h.has_images]
        memory_block = ""
        if user_id:
            if not history and self.settings.max_history_items:
                stored = await self.memory.load_history(user_id, conversation_id, self.settings.max_history_items)
                history = [HistoryEntry(role, content) for role, content in stored]
            memory_block = await self.memory.build_context(user_id, body.question, "long_term_memory" in features)

        ctx = ChatContext(
            question=body.question, subject=body.subject, level=body.level, mode=body.mode, history=history,
            images=[i.part() for i in images], requested_model=body.model, features=features, user_id=user_id,
            conversation_id=conversation_id, memory_block=memory_block, task_hint=task_hint,
        )
        return PreparedChat(ctx, conversation_id, user_id, body, images, warnings, task)

    # --- answering ---------------------------------------------------------------------
    async def respond(self, prep: PreparedChat) -> ChatResponse:
        started = time.monotonic()
        async with self._slot():
            result = await asyncio.wait_for(self.orchestrator.run(prep.ctx), self.settings.request_timeout_seconds)
        persisted = await self._persist(prep, result)
        return self.build_response(prep, result, persisted, started)

    async def stream(self, prep: PreparedChat) -> AsyncIterator[str]:
        """Server-Sent Events in the shape script.js expects: `data: {"delta": ...}` ... `data: [DONE]`."""
        started = time.monotonic()
        queue: asyncio.Queue = asyncio.Queue()

        async def sink(event: dict) -> None:
            await queue.put(event)

        async def worker() -> None:
            try:
                async with self._slot():
                    result = await asyncio.wait_for(self.orchestrator.run(prep.ctx, sink), self.settings.request_timeout_seconds)
                persisted = await self._persist(prep, result)
                await queue.put({"type": "final", "response": self.build_response(prep, result, persisted, started)})
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                mapped = error_for(exc)
                if mapped is None:
                    log.exception("Unexpected error while streaming")
                await queue.put({"type": "error", "message": mapped[1] if mapped else "Internal server error."})
            finally:
                await queue.put(None)

        task = asyncio.create_task(worker())
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), 15)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"  # keeps proxies from closing long tool calls
                    continue
                if event is None:
                    break
                kind = event["type"]
                if kind == "delta":
                    yield _sse({"delta": event["text"]})
                elif kind == "tool":
                    yield _sse({"event": "tool", "name": event["name"], "operation": event.get("operation"), "ok": event["ok"]})
                elif kind == "final":
                    r = event["response"]
                    yield _sse({"event": "done", "conversation_id": r.conversation_id, "model": r.model, "provider": r.provider,
                                "routing": r.routing, "warnings": r.warnings, "persisted": r.persisted,
                                "sources": [s.model_dump() for s in r.sources],
                                "tools_used": [t.model_dump() for t in r.tools_used]})
                elif kind == "error":
                    yield _sse({"error": event["message"]})
            yield "data: [DONE]\n\n"
        finally:
            task.cancel()  # the client pressed Stop or disconnected

    # --- helpers -----------------------------------------------------------------------
    async def _persist(self, prep: PreparedChat, result: OrchestratorResult) -> bool:
        if not prep.user_id:
            return False
        try:
            await self.memory.save_turn(
                prep.user_id, prep.conversation_id, prep.request.question, result.text,
                model=result.spec.model, provider=result.spec.provider, subject=prep.request.subject,
                level=prep.request.level, image_count=len(prep.images),
                meta={"mode": prep.request.mode, "tools": [t.name for t in result.traces],
                      "sources": [{"index": s.index, "title": s.title, "url": s.url} for s in result.sources]})
            return True
        except Exception:  # noqa: BLE001 - failing to save must never lose the answer
            log.exception("Could not save conversation turn")
            return False

    def build_response(self, prep: PreparedChat, result: OrchestratorResult, persisted: bool, started: float) -> ChatResponse:
        base = dict(
            answer=result.text, conversation_id=prep.conversation_id, model=result.spec.model, provider=result.spec.provider,
            routing={"designation": result.decision.designation, "reasons": result.decision.reasons,
                     "model_key": result.spec.key},
            tools_used=[ToolTraceOut(name=t.name, operation=t.operation, ok=t.ok, duration_ms=t.duration_ms, summary=t.summary)
                        for t in result.traces],
            sources=[SourceOut(index=s.index, title=s.title, url=s.url, snippet=s.snippet, kind=s.kind, note=s.note,
                               retrieved_at=s.retrieved_at) for s in result.sources],
            warnings=prep.warnings + result.warnings,
            usage=UsageOut(input_tokens=result.usage.input_tokens, output_tokens=result.usage.output_tokens),
            latency_ms=int((time.monotonic() - started) * 1000), persisted=persisted,
        )
        if prep.task is None:
            return ChatResponse(**base)
        from models import ImageAnalysisResponse  # local import keeps the common path light
        return ImageAnalysisResponse(
            **base, task=prep.task,
            images=[ImageInfoOut(name=i.name, width=i.width, height=i.height, bytes=i.size_bytes, sha256=i.sha256)
                    for i in prep.images])


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"
