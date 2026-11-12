"""The orchestrator turns a prepared chat request into an answer.

    route -> build prompt -> call model -> run tool calls -> call model again ... -> final answer

* Tools (SymPy, NumPy/SciPy, research, memory) are offered to the model, which decides when to use them.
* If a model fails (rate limit, outage, timeout) the next candidate from the router is tried.
* After a final answer, references the model wrote that were not retrieved by a tool are flagged.
* Streaming and non-streaming share the same code path; `emit` is only used for streaming.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from config import Settings

from .base import (
    AllModelsFailed, Completion, GenerationParams, ImagePart, Message, ModelSpec, ProviderError, Source, Usage,
)
from .prompts import build_system_prompt
from .router import RouteInput, Router, RoutingDecision
from .tools import ToolContext, ToolRegistry, ToolTrace

log = logging.getLogger("psai.orchestrator")

Emit = Callable[[dict], Awaitable[None]]


@dataclass
class HistoryEntry:
    role: str          # "user" | "assistant"
    content: str
    has_images: bool = False


@dataclass
class ChatContext:
    question: str
    subject: str
    level: str
    mode: str
    history: list[HistoryEntry]
    images: list[ImagePart]
    requested_model: str
    features: set[str]
    user_id: Optional[str]
    conversation_id: str
    memory_block: str = ""
    task_hint: str = ""


@dataclass
class OrchestratorResult:
    text: str
    spec: ModelSpec
    usage: Usage
    traces: list[ToolTrace]
    sources: list[Source]
    warnings: list[str]
    decision: RoutingDecision


class Emitter:
    """Wraps the streaming callback and remembers whether any text was already sent."""

    def __init__(self, sink: Emit):
        self._sink = sink
        self.sent_text = False

    async def __call__(self, event: dict) -> None:
        if event.get("type") == "delta":
            self.sent_text = True
        await self._sink(event)


_URL_RE = re.compile(r"https?://[^\s)>\]\"'`]+", re.I)
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s)>\]\"'`]+", re.I)


def audit_references(text: str, sources: list[Source], question: str) -> list[str]:
    """Links or DOIs in the answer that no tool returned and the user did not supply."""
    known = (" ".join((s.url or "") + " " + s.title for s in sources) + " " + question).lower()
    found = {m.rstrip(".,;:") for m in _URL_RE.findall(text)} | {m.rstrip(".,;:") for m in _DOI_RE.findall(text)}
    return sorted(r for r in found if r.lower() not in known)[:5]


class Orchestrator:
    def __init__(self, settings: Settings, registry, router: Router, tools: ToolRegistry):
        self.settings, self.registry, self.router, self.tools = settings, registry, router, tools

    # -- message assembly -------------------------------------------------------
    def _messages(self, ctx: ChatContext, system: str) -> list[Message]:
        budget = self.settings.max_context_chars - len(ctx.question) - len(system)
        kept: list[HistoryEntry] = []
        for entry in reversed(ctx.history):
            budget -= len(entry.content)
            if budget < 0:
                break
            kept.append(entry)
        kept.reverse()
        while kept and kept[0].role != "user":  # providers expect the conversation to start with the user
            kept.pop(0)

        messages = [Message("system", system)]
        for e in kept:
            note = "[The user attached an image to this message.]\n" if e.has_images else ""
            messages.append(Message(e.role, note + e.content))
        question = ctx.question
        if ctx.images and not question.strip():
            question = "Please look at the attached image."
        messages.append(Message("user", question, images=list(ctx.images)))
        return messages

    # -- public entry point -----------------------------------------------------
    async def run(self, ctx: ChatContext, emit: Optional[Emit] = None) -> OrchestratorResult:
        decision = self.router.decide(RouteInput(ctx.question, ctx.mode, ctx.level, bool(ctx.images), ctx.requested_model))
        emitter = Emitter(emit) if emit else None
        warnings = list(decision.warnings)
        failures: list[tuple[str, str]] = []

        for position, spec in enumerate(decision.candidates):
            tool_ctx = ToolContext(ctx.user_id, ctx.conversation_id, set(ctx.features))
            try:
                result = await self._run_with_model(spec, ctx, decision, tool_ctx, emitter, warnings)
                return result
            except ProviderError as exc:
                failures.append((spec.key, exc.kind))
                log.warning("Model %s failed (%s)", spec.key, exc.kind)
                if emitter and emitter.sent_text:
                    raise  # cannot switch models after part of an answer reached the client
                if position + 1 < len(decision.candidates):
                    warnings.append(f"{spec.key} failed ({exc.kind}); another model was used.")
        raise AllModelsFailed(failures)

    # -- one model, with the tool loop -------------------------------------------
    async def _run_with_model(self, spec: ModelSpec, ctx: ChatContext, decision: RoutingDecision,
                              tool_ctx: ToolContext, emitter: Optional[Emitter], warnings: list[str]) -> OrchestratorResult:
        provider = self.registry.provider_for(spec)
        tools = self.tools.specs_for(tool_ctx) if spec.tools else []
        system = build_system_prompt(subject=ctx.subject, level=ctx.level, mode=ctx.mode,
                                     tool_names=[t.name for t in tools], memory_block=ctx.memory_block,
                                     task_hint=ctx.task_hint, has_images=bool(ctx.images))
        messages = self._messages(ctx, system)
        reasoning = decision.designation == "reasoning"
        params = GenerationParams(
            max_tokens=self.settings.max_output_tokens_reasoning if reasoning else self.settings.max_output_tokens,
            temperature=self.settings.temperature,
            reasoning_effort=decision.reasoning_effort,
        )
        usage, traces = Usage(), []

        for round_no in range(self.settings.max_tool_rounds + 1):
            offered = tools if round_no < self.settings.max_tool_rounds else []
            completion = await self._call(provider, spec, messages, offered, params, emitter)
            usage += completion.usage
            if not completion.tool_calls or not offered:
                text = completion.text.strip()
                if not text:
                    raise ProviderError(f"{spec.key} returned an empty answer", kind="unavailable", provider=spec.provider)
                note = self._audit_note(text, tool_ctx.sources, ctx.question)
                if note:
                    text += note
                    if emitter:
                        await emitter({"type": "delta", "text": note})
                return OrchestratorResult(text, spec, usage, traces, tool_ctx.sources, warnings, decision)

            messages.append(Message("assistant", completion.text, tool_calls=completion.tool_calls))
            outcomes = await asyncio.gather(*(self.tools.execute(c, tool_ctx) for c in completion.tool_calls))
            for call, outcome in zip(completion.tool_calls, outcomes):
                messages.append(Message("tool", outcome.content, tool_call_id=call.id, name=call.name))
                traces.append(outcome.trace)
                if emitter:
                    await emitter({"type": "tool", "name": call.name, "operation": outcome.trace.operation,
                                   "ok": outcome.trace.ok})
        raise ProviderError("Tool loop did not finish", kind="unknown", provider=spec.provider)  # unreachable

    async def _call(self, provider, spec, messages, tools, params, emitter: Optional[Emitter]) -> Completion:
        attempts = 1 + self.settings.provider_retries
        for attempt in range(attempts):
            try:
                if emitter is None:
                    return await provider.complete(spec, messages, tools, params)
                completion: Optional[Completion] = None
                async for event in provider.stream(spec, messages, tools, params):
                    if event.kind == "text":
                        await emitter({"type": "delta", "text": event.text})
                    else:
                        completion = event.completion
                if completion is None:
                    raise ProviderError("The model stream ended unexpectedly", kind="unavailable", provider=spec.provider)
                return completion
            except ProviderError as exc:
                last = attempt == attempts - 1
                if last or not exc.retryable or (emitter is not None and emitter.sent_text):
                    raise
                await asyncio.sleep(min(exc.retry_after or float(attempt + 1), 5.0))
        raise ProviderError("unreachable", kind="unknown")  # pragma: no cover

    @staticmethod
    def _audit_note(text: str, sources: list[Source], question: str) -> str:
        unverified = audit_references(text, sources, question)
        if not unverified:
            return ""
        shown = ", ".join(unverified[:3])
        return ("\n\n---\n*Note: this answer mentions links or identifiers that were not checked against any retrieved "
                f"source ({shown}). Verify them before relying on or citing them.*")
