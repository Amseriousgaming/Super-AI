"""Provider-neutral data types and the interface every AI provider implements.

Nothing in this module knows about a specific vendor. A new provider only has to
subclass `AIProvider` (see providers.py) and describe its models with `ModelSpec`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal, Optional

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ImagePart:
    mime_type: str
    data_b64: str


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    role: Role
    content: str = ""
    images: list[ImagePart] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)  # assistant messages
    tool_call_id: Optional[str] = None                        # tool messages
    name: Optional[str] = None


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __iadd__(self, other: "Usage") -> "Usage":
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        return self


@dataclass
class Completion:
    text: str
    tool_calls: list[ToolCall]
    usage: Usage
    model: str
    provider: str
    finish_reason: str = ""


@dataclass
class StreamEvent:
    kind: Literal["text", "done"]
    text: str = ""
    completion: Optional[Completion] = None


@dataclass
class GenerationParams:
    max_tokens: int = 4096
    temperature: Optional[float] = None
    reasoning_effort: Optional[str] = None  # "low" | "medium" | "high"


@dataclass(frozen=True)
class ModelSpec:
    key: str                           # "provider:model", the id the frontend uses
    provider: str
    model: str
    designations: frozenset            # subset of {general, fast, reasoning, vision}
    vision: bool = False
    tools: bool = True


@dataclass
class Source:
    """A reference returned by a retrieval source. Only real lookups create these."""

    index: int
    title: str
    url: Optional[str]
    snippet: str
    kind: str                          # encyclopedia | preprint | journal | ...
    note: Optional[str] = None
    retrieved_at: float = 0.0


class ProviderError(Exception):
    """A model call failed. `kind` drives retry and fallback decisions."""

    def __init__(self, message: str, kind: str = "unknown", status: Optional[int] = None,
                 retry_after: Optional[float] = None, provider: Optional[str] = None):
        super().__init__(message)
        self.kind, self.status, self.retry_after, self.provider = kind, status, retry_after, provider

    @property
    def retryable(self) -> bool:
        return self.kind in {"rate_limit", "unavailable", "timeout"}


class NoModelAvailable(Exception):
    """No configured model can serve this request (for example no vision model for an image)."""


class AllModelsFailed(Exception):
    def __init__(self, failures: list[tuple[str, str]]):
        super().__init__("; ".join(f"{m}: {k}" for m, k in failures))
        self.failures = failures  # [(model_key, error_kind)]


class AIProvider(ABC):
    """Interface for an AI vendor. One instance owns one reusable HTTP client."""

    name: str
    specs: list[ModelSpec]

    @abstractmethod
    async def complete(self, spec: ModelSpec, messages: list[Message], tools: list[ToolSpec],
                       params: GenerationParams) -> Completion:
        ...

    async def stream(self, spec: ModelSpec, messages: list[Message], tools: list[ToolSpec],
                     params: GenerationParams) -> AsyncIterator[StreamEvent]:
        """Default: no real streaming. Providers override this with true token streaming."""
        completion = await self.complete(spec, messages, tools, params)
        if completion.text:
            yield StreamEvent("text", text=completion.text)
        yield StreamEvent("done", completion=completion)

    async def aclose(self) -> None:
        return None
