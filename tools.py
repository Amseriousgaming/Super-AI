"""Tools the AI model may call, and the registry that runs them.

    math_compute      -> engines.mathematics (SymPy), sandboxed
    science_compute   -> engines.scientific (NumPy/SciPy), sandboxed
    research          -> services.knowledge (Wikipedia, arXiv, Crossref, ...)
    remember          -> services.memory (long-term memory)

Which tools are offered depends on the effective feature flags of the request.
Add a tool by adding a spec in `specs_for` and a branch in `execute`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from engines import ENGINES, EngineError

from .base import Source, ToolCall, ToolSpec

log = logging.getLogger("psai.tools")
MAX_TOOL_RESULT_CHARS = 6000


@dataclass
class ToolTrace:
    name: str
    operation: Optional[str]
    ok: bool
    duration_ms: int
    summary: Optional[str] = None


@dataclass
class ToolOutcome:
    content: str
    trace: ToolTrace


@dataclass
class ToolContext:
    user_id: Optional[str]
    conversation_id: str
    features: set[str]
    sources: list[Source] = field(default_factory=list)  # filled by the research tool


def _describe(engine: str) -> str:
    return "\n".join(f"- {op.signature}: {op.summary}" for op in ENGINES[engine].values())


def _json(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) > MAX_TOOL_RESULT_CHARS:
        text = text[:MAX_TOOL_RESULT_CHARS] + ' ... [truncated]"}'
    return text


class ToolRegistry:
    def __init__(self, engines, knowledge, memory):
        self.engines, self.knowledge, self.memory = engines, knowledge, memory

    def specs_for(self, ctx: ToolContext) -> list[ToolSpec]:
        specs: list[ToolSpec] = []
        if "math_verification" in ctx.features:
            specs.append(ToolSpec(
                "math_compute",
                "Exact symbolic mathematics with SymPy. Call with an operation name and an args object.\nOperations:\n"
                + _describe("math"),
                {"type": "object", "properties": {
                    "operation": {"type": "string", "enum": list(ENGINES["math"])},
                    "args": {"type": "object", "description": "Arguments for the operation, as listed in its signature."}},
                 "required": ["operation", "args"]}))
        if "scientific_calc" in ctx.features:
            specs.append(ToolSpec(
                "science_compute",
                "Numeric and scientific computation (NumPy/SciPy and physics/chemistry helpers). Call with an operation name "
                "and an args object.\nOperations:\n" + _describe("science"),
                {"type": "object", "properties": {
                    "operation": {"type": "string", "enum": list(ENGINES["science"])},
                    "args": {"type": "object", "description": "Arguments for the operation, as listed in its signature."}},
                 "required": ["operation", "args"]}))
        if "web_research" in ctx.features and self.knowledge is not None and self.knowledge.available:
            specs.append(ToolSpec(
                "research",
                "Search real reference sources and return numbered results (title, url, snippet). Sources: "
                + ", ".join(self.knowledge.source_names) + ". Only cite what this tool returns.",
                {"type": "object", "properties": {
                    "query": {"type": "string", "description": "Short search query."},
                    "source": {"type": "string", "description": "Optional: one source name, or 'auto' for all."}},
                 "required": ["query"]}))
        if "long_term_memory" in ctx.features and ctx.user_id and self.memory is not None:
            specs.append(ToolSpec(
                "remember",
                "Save a durable fact about the user for future conversations.",
                {"type": "object", "properties": {
                    "fact": {"type": "string", "description": "One short, self-contained sentence."},
                    "kind": {"type": "string", "enum": ["fact", "preference", "goal"]}},
                 "required": ["fact"]}))
        return specs

    async def execute(self, call: ToolCall, ctx: ToolContext) -> ToolOutcome:
        started = time.monotonic()
        operation: Optional[str] = None
        try:
            if "_error" in call.arguments:
                raise EngineError(call.arguments["_error"])
            if call.name in {"math_compute", "science_compute"}:
                engine = "math" if call.name == "math_compute" else "science"
                operation = str(call.arguments.get("operation", ""))
                result = await self.engines.run(engine, operation, call.arguments.get("args") or {})
                content, summary = _json({"ok": True, "result": result}), str(result.get("result", ""))[:120] or None
            elif call.name == "research":
                query = str(call.arguments.get("query", "")).strip()
                if not query:
                    raise EngineError("'query' is required.")
                found = await self.knowledge.search(query, str(call.arguments.get("source", "auto")), start_index=len(ctx.sources) + 1)
                ctx.sources.extend(found)
                content = _json({"ok": True, "results": [
                    {"index": s.index, "title": s.title, "url": s.url, "snippet": s.snippet, "kind": s.kind, "note": s.note}
                    for s in found] or "No results. Say that nothing was found rather than guessing."})
                operation, summary = "search", f"{len(found)} result(s)"
            elif call.name == "remember":
                fact = str(call.arguments.get("fact", "")).strip()
                if not fact or len(fact) > 500:
                    raise EngineError("'fact' must be 1 to 500 characters.")
                await self.memory.add_memory(ctx.user_id, fact, kind=str(call.arguments.get("kind", "fact")),
                                             source_conversation=ctx.conversation_id)
                content, operation, summary = _json({"ok": True, "saved": True}), "save", None
            else:
                raise EngineError(f"Unknown tool '{call.name}'.")
            ok = True
        except EngineError as exc:
            content, summary, ok = _json({"ok": False, "error": str(exc)}), str(exc)[:120], False
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a tool failure must never crash the chat
            log.exception("Tool %s failed", call.name)
            content, summary, ok = _json({"ok": False, "error": f"{type(exc).__name__}: tool failed"}), "tool failed", False
        return ToolOutcome(content, ToolTrace(call.name, operation, ok, int((time.monotonic() - started) * 1000), summary))
