"""Concrete AI providers.

  * OpenAICompatibleProvider: anything that speaks `/chat/completions`
    (OpenAI, Google Gemini's OpenAI endpoint, Groq, OpenRouter, Ollama, vLLM, LM Studio, ...).
  * AnthropicProvider: the Anthropic Messages API.

Add another vendor by subclassing `AIProvider`, then register it:

    register_provider_kind("myvendor", MyVendorProvider)

and give it a preset in config.PRESETS with `"kind": "myvendor"`.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, AsyncIterator, Optional

import httpx

from config import ProviderConfig, Settings

from .base import (
    AIProvider, Completion, GenerationParams, Message, ModelSpec, ProviderError, StreamEvent, ToolCall,
    ToolSpec, Usage,
)

log = logging.getLogger("psai.providers")


# --- shared helpers -----------------------------------------------------------
def specs_from_config(cfg: ProviderConfig) -> list[ModelSpec]:
    """One ModelSpec per distinct model id; designations of duplicates are merged."""
    merged: dict[str, dict[str, Any]] = {}
    for designation, model in cfg.models.items():
        entry = merged.setdefault(model, {"designations": set(), "vision": cfg.vision_all})
        entry["designations"].add(designation)
        if designation == "vision":
            entry["vision"] = True
    return [
        ModelSpec(key=f"{cfg.name}:{model}", provider=cfg.name, model=model,
                  designations=frozenset(e["designations"]), vision=e["vision"], tools=cfg.tools)
        for model, e in merged.items()
    ]


def _http_error(provider: str, status: int, body: str, headers: httpx.Headers) -> ProviderError:
    detail = ""
    try:
        data = json.loads(body)
        err = data.get("error", data) if isinstance(data, dict) else {}
        detail = err.get("message", "") if isinstance(err, dict) else str(err)
    except ValueError:
        detail = body
    detail = re.sub(r"\s+", " ", detail or "")[:300]
    retry_after: Optional[float] = None
    try:
        retry_after = float(headers.get("retry-after", ""))
    except ValueError:
        pass
    if status in (401, 403):
        kind = "auth"
    elif status == 429:
        kind = "rate_limit"
    elif status in (408, 504):
        kind = "timeout"
    elif status >= 500 or status == 529:
        kind = "unavailable"
    else:
        kind = "bad_request"
    log.warning("%s returned HTTP %s (%s): %s", provider, status, kind, detail)
    return ProviderError(f"{provider} returned HTTP {status}", kind=kind, status=status,
                         retry_after=retry_after, provider=provider)


def _transport_error(provider: str, exc: Exception) -> ProviderError:
    if isinstance(exc, httpx.TimeoutException):
        return ProviderError(f"{provider} timed out", kind="timeout", provider=provider)
    return ProviderError(f"{provider} connection failed ({type(exc).__name__})", kind="unavailable", provider=provider)


def _client(cfg: ProviderConfig, settings: Settings, headers: dict[str, str]) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=cfg.base_url.rstrip("/"),
        headers=headers,
        timeout=httpx.Timeout(connect=settings.provider_connect_timeout, read=settings.provider_timeout_seconds,
                              write=30.0, pool=10.0),
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
    )


def _json_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
        return value if isinstance(value, dict) else {"_error": "tool arguments must be a JSON object"}
    except ValueError:
        return {"_error": "tool arguments were not valid JSON"}


_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


class _ThinkFilter:
    """Drops <think>...</think> blocks that some open models emit, even when split across chunks."""

    def __init__(self) -> None:
        self.inside = False
        self.buf = ""

    def feed(self, chunk: str) -> str:
        self.buf += chunk
        out = ""
        while self.buf:
            if self.inside:
                end = self.buf.find("</think>")
                if end < 0:
                    self.buf = self.buf[-8:]
                    return out
                self.buf, self.inside = self.buf[end + 8:], False
            else:
                start = self.buf.find("<think>")
                if start >= 0:
                    out += self.buf[:start]
                    self.buf, self.inside = self.buf[start + 7:], True
                    continue
                keep = 0
                for n in range(min(6, len(self.buf)), 0, -1):  # hold back a partial "<think"
                    if "<think>".startswith(self.buf[-n:]):
                        keep = n
                        break
                out += self.buf[: len(self.buf) - keep]
                self.buf = self.buf[len(self.buf) - keep:]
                return out
        return out

    def flush(self) -> str:
        rest, self.buf = ("" if self.inside else self.buf), ""
        return rest


# --- OpenAI-compatible --------------------------------------------------------
class OpenAICompatibleProvider(AIProvider):
    def __init__(self, cfg: ProviderConfig, settings: Settings):
        self.name, self.cfg = cfg.name, cfg
        headers = {"Content-Type": "application/json", **cfg.extra_headers}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        self.client = _client(cfg, settings, headers)
        self.specs = specs_from_config(cfg)

    # request building
    @staticmethod
    def _message(m: Message) -> dict[str, Any]:
        if m.role == "tool":
            return {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content}
        if m.role == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": m.content or ""}
            if m.tool_calls:
                msg["content"] = m.content or None
                msg["tool_calls"] = [{"id": c.id, "type": "function",
                                      "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                                     for c in m.tool_calls]
            return msg
        if m.images:
            parts: list[dict[str, Any]] = [{"type": "text", "text": m.content}] if m.content else []
            parts += [{"type": "image_url", "image_url": {"url": f"data:{i.mime_type};base64,{i.data_b64}"}}
                      for i in m.images]
            return {"role": m.role, "content": parts}
        return {"role": m.role, "content": m.content}

    def _body(self, spec: ModelSpec, messages: list[Message], tools: list[ToolSpec], params: GenerationParams,
              stream: bool) -> dict[str, Any]:
        reasoning = "reasoning" in spec.designations and self.cfg.reasoning_effort
        body: dict[str, Any] = {"model": spec.model, "messages": [self._message(m) for m in messages],
                                self.cfg.max_tokens_param: params.max_tokens}
        if params.temperature is not None and not reasoning:
            body["temperature"] = params.temperature
        if reasoning and params.reasoning_effort:
            body["reasoning_effort"] = params.reasoning_effort
        if tools and spec.tools:
            body["tools"] = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                               "parameters": t.parameters}} for t in tools]
            body["tool_choice"] = "auto"
        if stream:
            body["stream"] = True
        return body

    # non-streaming
    async def complete(self, spec, messages, tools, params) -> Completion:
        try:
            resp = await self.client.post("/chat/completions", json=self._body(spec, messages, tools, params, False))
        except httpx.HTTPError as exc:
            raise _transport_error(self.name, exc) from exc
        if resp.status_code >= 400:
            raise _http_error(self.name, resp.status_code, resp.text, resp.headers)
        try:
            data = resp.json()
        except ValueError as exc:
            raise ProviderError(f"{self.name} returned invalid JSON", kind="unavailable", provider=self.name) from exc
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        text = msg.get("content") or ""
        if isinstance(text, list):
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
        calls = [ToolCall(id=tc.get("id") or uuid.uuid4().hex[:12], name=(tc.get("function") or {}).get("name", ""),
                          arguments=_json_args((tc.get("function") or {}).get("arguments")))
                 for tc in msg.get("tool_calls") or []]
        usage = data.get("usage") or {}
        return Completion(text=_THINK_RE.sub("", text).strip(), tool_calls=calls,
                          usage=Usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)),
                          model=spec.model, provider=self.name, finish_reason=choice.get("finish_reason") or "")

    # streaming
    async def stream(self, spec, messages, tools, params) -> AsyncIterator[StreamEvent]:
        parts: list[str] = []
        slots: dict[int, dict[str, str]] = {}
        finish, usage, think = "", Usage(), _ThinkFilter()
        try:
            async with self.client.stream("POST", "/chat/completions",
                                          json=self._body(spec, messages, tools, params, True)) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", "replace")
                    raise _http_error(self.name, resp.status_code, body, resp.headers)
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except ValueError:
                        continue
                    if chunk.get("usage"):
                        usage = Usage(chunk["usage"].get("prompt_tokens", 0), chunk["usage"].get("completion_tokens", 0))
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        piece = think.feed(delta["content"]) if delta.get("content") else ""
                        if piece:
                            parts.append(piece)
                            yield StreamEvent("text", text=piece)
                        for tc in delta.get("tool_calls") or []:
                            slot = slots.setdefault(tc.get("index", 0), {"id": "", "name": "", "args": ""})
                            slot["id"] = tc.get("id") or slot["id"]
                            fn = tc.get("function") or {}
                            slot["name"] = slot["name"] or fn.get("name", "")
                            slot["args"] += fn.get("arguments") or ""
                        finish = choice.get("finish_reason") or finish
        except httpx.HTTPError as exc:
            raise _transport_error(self.name, exc) from exc
        tail = think.flush()
        if tail:
            parts.append(tail)
            yield StreamEvent("text", text=tail)
        calls = [ToolCall(id=s["id"] or uuid.uuid4().hex[:12], name=s["name"], arguments=_json_args(s["args"]))
                 for _, s in sorted(slots.items())]
        yield StreamEvent("done", completion=Completion(text="".join(parts).strip(), tool_calls=calls, usage=usage,
                                                       model=spec.model, provider=self.name, finish_reason=finish))

    async def aclose(self) -> None:
        await self.client.aclose()


# --- Anthropic ----------------------------------------------------------------
class AnthropicProvider(AIProvider):
    API_VERSION = "2023-06-01"

    def __init__(self, cfg: ProviderConfig, settings: Settings):
        self.name, self.cfg = cfg.name, cfg
        self.client = _client(cfg, settings, {"x-api-key": cfg.api_key or "", "anthropic-version": self.API_VERSION,
                                              "content-type": "application/json", **cfg.extra_headers})
        self.specs = specs_from_config(cfg)

    @staticmethod
    def _convert(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                continue
            if m.role == "tool":
                block = {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}
                if out and out[-1]["role"] == "user" and out[-1].get("_tool"):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block], "_tool": True})
            elif m.role == "assistant":
                blocks: list[dict[str, Any]] = [{"type": "text", "text": m.content}] if m.content else []
                blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments} for c in m.tool_calls]
                out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "."}]})
            else:
                if m.images:
                    blocks = [{"type": "image", "source": {"type": "base64", "media_type": i.mime_type,
                                                           "data": i.data_b64}} for i in m.images]
                    blocks.append({"type": "text", "text": m.content or "Please look at the image."})
                    out.append({"role": "user", "content": blocks})
                else:
                    out.append({"role": "user", "content": m.content})
        for item in out:
            item.pop("_tool", None)
        return system, out

    def _body(self, spec, messages, tools, params, stream: bool) -> dict[str, Any]:
        system, msgs = self._convert(messages)
        body: dict[str, Any] = {"model": spec.model, "max_tokens": params.max_tokens, "messages": msgs}
        if system:
            body["system"] = system
        if params.temperature is not None:
            body["temperature"] = params.temperature
        if tools and spec.tools:
            body["tools"] = [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools]
        if stream:
            body["stream"] = True
        return body

    async def complete(self, spec, messages, tools, params) -> Completion:
        try:
            resp = await self.client.post("/v1/messages", json=self._body(spec, messages, tools, params, False))
        except httpx.HTTPError as exc:
            raise _transport_error(self.name, exc) from exc
        if resp.status_code >= 400:
            raise _http_error(self.name, resp.status_code, resp.text, resp.headers)
        try:
            data = resp.json()
        except ValueError as exc:
            raise ProviderError(f"{self.name} returned invalid JSON", kind="unavailable", provider=self.name) from exc
        text, calls = [], []
        for block in data.get("content") or []:
            if block.get("type") == "text":
                text.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                calls.append(ToolCall(id=block.get("id") or uuid.uuid4().hex[:12], name=block.get("name", ""),
                                      arguments=_json_args(block.get("input"))))
        u = data.get("usage") or {}
        return Completion(text="".join(text).strip(), tool_calls=calls,
                          usage=Usage(u.get("input_tokens", 0), u.get("output_tokens", 0)),
                          model=spec.model, provider=self.name, finish_reason=data.get("stop_reason") or "")

    async def stream(self, spec, messages, tools, params) -> AsyncIterator[StreamEvent]:
        parts: list[str] = []
        blocks: dict[int, dict[str, str]] = {}
        usage, finish = Usage(), ""
        try:
            async with self.client.stream("POST", "/v1/messages",
                                          json=self._body(spec, messages, tools, params, True)) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", "replace")
                    raise _http_error(self.name, resp.status_code, body, resp.headers)
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    try:
                        ev = json.loads(line[5:].strip())
                    except ValueError:
                        continue
                    kind = ev.get("type")
                    if kind == "message_start":
                        usage.input_tokens = (ev.get("message", {}).get("usage") or {}).get("input_tokens", 0)
                    elif kind == "content_block_start":
                        cb = ev.get("content_block") or {}
                        if cb.get("type") == "tool_use":
                            blocks[ev.get("index", 0)] = {"id": cb.get("id", ""), "name": cb.get("name", ""), "args": ""}
                    elif kind == "content_block_delta":
                        d = ev.get("delta") or {}
                        if d.get("type") == "text_delta" and d.get("text"):
                            parts.append(d["text"])
                            yield StreamEvent("text", text=d["text"])
                        elif d.get("type") == "input_json_delta":
                            blocks.setdefault(ev.get("index", 0), {"id": "", "name": "", "args": ""})["args"] += \
                                d.get("partial_json", "")
                    elif kind == "message_delta":
                        finish = (ev.get("delta") or {}).get("stop_reason") or finish
                        usage.output_tokens = (ev.get("usage") or {}).get("output_tokens", usage.output_tokens)
                    elif kind == "error":
                        err = ev.get("error") or {}
                        raise ProviderError(f"{self.name} stream error", provider=self.name,
                                            kind="unavailable" if err.get("type") == "overloaded_error" else "unknown")
        except httpx.HTTPError as exc:
            raise _transport_error(self.name, exc) from exc
        calls = [ToolCall(id=b["id"] or uuid.uuid4().hex[:12], name=b["name"], arguments=_json_args(b["args"]))
                 for _, b in sorted(blocks.items())]
        yield StreamEvent("done", completion=Completion(text="".join(parts).strip(), tool_calls=calls, usage=usage,
                                                       model=spec.model, provider=self.name, finish_reason=finish))

    async def aclose(self) -> None:
        await self.client.aclose()


# --- registry -----------------------------------------------------------------
PROVIDER_KINDS: dict[str, type[AIProvider]] = {
    "openai_compat": OpenAICompatibleProvider,
    "anthropic": AnthropicProvider,
}


def register_provider_kind(kind: str, cls: type[AIProvider]) -> None:
    """Make a new provider implementation available to config presets."""
    PROVIDER_KINDS[kind] = cls


def build_providers(settings: Settings) -> list[AIProvider]:
    providers: list[AIProvider] = []
    for cfg in settings.provider_configs():
        cls = PROVIDER_KINDS.get(cfg.kind)
        if cls is None:
            log.error("Unknown provider kind %r for %s; skipped", cfg.kind, cfg.name)
            continue
        providers.append(cls(cfg, settings))
        log.info("Provider enabled: %s (%s)", cfg.name, ", ".join(sorted(set(cfg.models.values()))))
    return providers
