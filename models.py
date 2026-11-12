"""Pydantic schemas for requests and responses.

The chat request mirrors what the existing frontend (script.js) sends. Subjects,
levels and modes are free-form labels, so nothing is limited to a fixed list.
"""
from __future__ import annotations

import re
from typing import Any, ClassVar, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from config import get_settings

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,64}$")
_LABEL_RE = re.compile(r"^[\w .+&/-]{1,64}$")


def _label(value: str, field_name: str) -> str:
    value = (value or "auto").strip().lower() or "auto"
    if not _LABEL_RE.match(value):
        raise ValueError(f"{field_name} must be 1-64 plain characters")
    return value


# --- requests -----------------------------------------------------------------
class ImagePayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field("image", max_length=200)
    mime_type: str = Field("image/jpeg", max_length=100)
    data: str = Field(..., min_length=16, description="Base64 image data (a data: URL prefix is tolerated).")
    width: Optional[int] = Field(None, ge=1, le=100_000)
    height: Optional[int] = Field(None, ge=1, le=100_000)


class HistoryItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["user", "assistant"]
    content: str = Field("", max_length=100_000)
    has_images: bool = False


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    require_content: ClassVar[bool] = True

    question: str = Field("", description="The user's question.")
    subject: str = "auto"
    level: str = "auto"
    mode: str = "detailed"
    model: str = Field("auto", max_length=128)
    image: Optional[ImagePayload] = None
    images: list[ImagePayload] = Field(default_factory=list, max_length=16)
    history: list[HistoryItem] = Field(default_factory=list, max_length=500)
    conversation_id: Optional[str] = None
    features: dict[str, bool] = Field(default_factory=dict)
    stream: bool = False
    client: Optional[dict[str, Any]] = None

    @field_validator("subject")
    @classmethod
    def _subject(cls, v: str) -> str:
        return _label(v, "subject")

    @field_validator("level")
    @classmethod
    def _level(cls, v: str) -> str:
        return _label(v, "level")

    @field_validator("mode")
    @classmethod
    def _mode(cls, v: str) -> str:
        return _label(v, "mode")

    @field_validator("conversation_id")
    @classmethod
    def _conversation_id(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v == "":
            return None
        if not _ID_RE.match(v):
            raise ValueError("conversation_id must be 6-64 characters: letters, digits, '_' or '-'")
        return v

    @field_validator("features")
    @classmethod
    def _features(cls, v: dict[str, bool]) -> dict[str, bool]:
        if len(v) > 32:
            raise ValueError("too many feature flags")
        return {str(k)[:48]: bool(val) for k, val in v.items()}

    @field_validator("question")
    @classmethod
    def _question(cls, v: str) -> str:
        limit = get_settings().max_question_chars
        if len(v) > limit:
            raise ValueError(f"question is longer than {limit} characters")
        return v.strip()

    @model_validator(mode="after")
    def _needs_content(self) -> "ChatRequest":
        if self.require_content and not self.question and not (self.image or self.images):
            raise ValueError("Send a question, an image, or both.")
        return self

    def all_images(self) -> list[ImagePayload]:
        """`image` is a convenience copy of `images[0]`; avoid sending the same picture twice."""
        items = list(self.images)
        if self.image and not any(i.data == self.image.data for i in items):
            items.insert(0, self.image)
        return items


class AnalyzeImageRequest(ChatRequest):
    require_content: ClassVar[bool] = False

    task: str = "auto"
    source: str = "upload"  # upload | screenshot | screen_stream (reserved for continuous screen input)

    @field_validator("task", "source")
    @classmethod
    def _plain(cls, v: str) -> str:
        return _label(v, "task/source")


class ComputeRequest(BaseModel):
    engine: Literal["math", "science"]
    operation: str = Field(..., max_length=64)
    args: dict[str, Any] = Field(default_factory=dict)


class PreferencesBody(BaseModel):
    preferences: dict[str, Any]

    @field_validator("preferences")
    @classmethod
    def _check(cls, v: dict[str, Any]) -> dict[str, Any]:
        if len(v) > 50:
            raise ValueError("at most 50 preferences")
        for key, val in v.items():
            if not re.fullmatch(r"[A-Za-z0-9_.-]{1,48}", key):
                raise ValueError(f"invalid preference key '{key}'")
            if val is not None and not isinstance(val, (str, int, float, bool)):
                raise ValueError("preference values must be strings, numbers, booleans or null (null deletes)")
            if isinstance(val, str) and len(val) > 1000:
                raise ValueError("preference values are limited to 1000 characters")
        return v


# --- responses ----------------------------------------------------------------
class SourceOut(BaseModel):
    index: int
    title: str
    url: Optional[str] = None
    snippet: Optional[str] = None
    kind: str
    note: Optional[str] = None
    retrieved_at: float


class ToolTraceOut(BaseModel):
    name: str
    operation: Optional[str] = None
    ok: bool
    duration_ms: int
    summary: Optional[str] = None


class UsageOut(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    model: str
    provider: str
    routing: dict[str, Any]
    tools_used: list[ToolTraceOut] = Field(default_factory=list)
    sources: list[SourceOut] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    usage: UsageOut = Field(default_factory=UsageOut)
    latency_ms: int
    persisted: bool = False


class ImageInfoOut(BaseModel):
    name: str
    width: int
    height: int
    bytes: int
    sha256: str


class ImageAnalysisResponse(ChatResponse):
    task: str
    images: list[ImageInfoOut]


class ConversationSummary(BaseModel):
    id: str
    title: str
    subject: Optional[str] = None
    level: Optional[str] = None
    created_at: float
    updated_at: float
    message_count: int


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    model: Optional[str] = None
    provider: Optional[str] = None
    created_at: float
    meta: dict[str, Any] = Field(default_factory=dict)


class ConversationDetail(ConversationSummary):
    messages: list[MessageOut]


class HistoryResponse(BaseModel):
    conversations: list[ConversationSummary]
    total: int
    limit: int
    offset: int


class DeleteResponse(BaseModel):
    deleted: bool
    conversation_id: str


class PreferencesResponse(BaseModel):
    preferences: dict[str, Any]


class MemoryOut(BaseModel):
    id: int
    kind: str
    content: str
    created_at: float


class MemoriesResponse(BaseModel):
    memories: list[MemoryOut]


class ModelInfo(BaseModel):
    key: str
    provider: str
    model: str
    designations: list[str]
    vision: bool
    tools: bool


class ModelsResponse(BaseModel):
    default: str = "auto"
    models: list[ModelInfo]
    features: dict[str, bool]  # feature id -> available on this server


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    time: float
    database: bool
    engines: bool
    providers: list[dict[str, Any]]
    research_sources: list[str]
    features: list[str]
    notes: list[str] = Field(default_factory=list)


class ComputeResponse(BaseModel):
    ok: bool
    result: dict[str, Any]
