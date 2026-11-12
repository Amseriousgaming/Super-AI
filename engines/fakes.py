"""Test doubles. These exist ONLY for tests, to exercise the real pipeline without calling a paid API.
Production code never imports this module."""
from __future__ import annotations

import asyncio
from typing import Optional

from ai.base import AIProvider, Completion, ModelSpec, ProviderError, ToolCall, Usage


def make_settings(**overrides):
    from config import Settings

    base = dict(engine_sandbox=False, provider_retries=0, database_path=":memory:")
    base.update(overrides)
    return Settings(_env_file=None, **base)


def run(coro):
    return asyncio.run(coro)


def spec(provider: str, model: str, designations=("general",), vision: bool = False, tools: bool = True) -> ModelSpec:
    return ModelSpec(f"{provider}:{model}", provider, model, frozenset(designations), vision, tools)


class ScriptedProvider(AIProvider):
    """Replays a script. Steps: ("text", "...") or ("tool", "math_compute", {...args}).
    `fail_with` makes every call raise that ProviderError instead."""

    def __init__(self, name: str = "fake", script: Optional[list[tuple]] = None, specs: Optional[list[ModelSpec]] = None,
                 fail_with: Optional[ProviderError] = None):
        self.name = name
        self.script = list(script or [("text", "ok")])
        self.specs = specs or [spec(name, "m1")]
        self.fail_with = fail_with
        self.calls: list[dict] = []

    async def complete(self, spec, messages, tools, params) -> Completion:
        self.calls.append({"model": spec.key, "messages": list(messages), "tools": [t.name for t in tools]})
        if self.fail_with:
            raise self.fail_with
        step = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if step[0] == "tool":
            call = ToolCall(id=f"call{len(self.calls)}", name=step[1], arguments=step[2])
            return Completion("", [call], Usage(10, 5), spec.model, self.name, "tool_calls")
        return Completion(step[1], [], Usage(10, 5), spec.model, self.name, "stop")
