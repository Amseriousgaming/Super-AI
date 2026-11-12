"""Central configuration for Personal Super AI.

Everything is read from environment variables (or a local `.env` file).
Secrets such as API keys are only ever read here, held in memory, and never
logged, returned by an endpoint, or written to the database.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Optional

from dotenv import load_dotenv
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")  # makes .env visible to the provider scanner below

APP_VERSION = "1.0.0"
log = logging.getLogger("psai.config")


def _csv(value: str) -> list[str]:
    return [p.strip() for p in value.split(",") if p.strip()]


def _bool(value: Optional[str], default: bool) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# AI provider configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProviderConfig:
    """One configured AI provider. Built from environment variables."""

    name: str
    kind: str                          # key into providers.PROVIDER_KINDS
    base_url: str
    api_key: Optional[str]
    models: dict[str, str]             # designation -> model id
    vision_all: bool = False           # every model of this provider accepts images
    tools: bool = True                 # models support function/tool calling
    max_tokens_param: str = "max_tokens"
    reasoning_effort: bool = False     # send `reasoning_effort` to reasoning models
    extra_headers: dict[str, str] = field(default_factory=dict)

    def __repr__(self) -> str:  # never leak the key through logs or tracebacks
        return f"ProviderConfig(name={self.name!r}, kind={self.kind!r}, models={self.models!r})"


# Designations: general (default), fast (cheap/quick), reasoning (hard problems),
# vision (guaranteed image input). A provider only needs one of them.
PRESETS: dict[str, dict] = {
    "anthropic": {
        "kind": "anthropic",
        "base_url": "https://api.anthropic.com",
        "vision": True,
        "defaults": {"general": "claude-sonnet-5-5", "fast": "claude-haiku-5-5", "reasoning": "claude-opus-5-5"},
    },
    "openai": {
        "kind": "openai_compat",
        "base_url": "https://api.openai.com/v1",
        "vision": True,
        "max_tokens_param": "max_completion_tokens",
        "reasoning_effort": True,
    },
    "gemini": {
        "kind": "openai_compat",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "vision": True,
    },
    "groq": {"kind": "openai_compat", "base_url": "https://api.groq.com/openai/v1", "vision": False},
    "openrouter": {"kind": "openai_compat", "base_url": "https://openrouter.ai/api/v1", "vision": False},
    "ollama": {"kind": "openai_compat", "base_url": "http://localhost:11434/v1", "vision": False, "needs_key": False},
    "custom": {"kind": "openai_compat", "base_url": "", "vision": False, "needs_key": False},
}

_MODEL_SUFFIXES = (
    ("general", "MODEL"),
    ("fast", "FAST_MODEL"),
    ("reasoning", "REASONING_MODEL"),
    ("vision", "VISION_MODEL"),
)


def load_provider_configs(env: Optional[Mapping[str, str]] = None) -> list[ProviderConfig]:
    """Scan the environment for `<PROVIDER>_API_KEY`, `<PROVIDER>_MODEL`, ... entries."""
    env = os.environ if env is None else env
    configs: list[ProviderConfig] = []
    for name, preset in PRESETS.items():
        p = name.upper()
        key = (env.get(f"{p}_API_KEY") or "").strip() or None
        needs_key = preset.get("needs_key", True)
        base_url = (env.get(f"{p}_BASE_URL") or preset["base_url"]).strip()

        models = dict(preset.get("defaults", {}))
        if not key and needs_key:
            models = {}
        for designation, suffix in _MODEL_SUFFIXES:
            value = (env.get(f"{p}_{suffix}") or "").strip()
            if value:
                models[designation] = value

        if needs_key and not key:
            continue
        if not models:
            if key:
                log.warning("%s_API_KEY is set but no %s_MODEL was given; provider skipped.", p, p)
            continue
        if not base_url:
            log.warning("%s needs %s_BASE_URL; provider skipped.", p, p)
            continue

        extra = {}
        if name == "openrouter":
            if env.get("OPENROUTER_REFERER"):
                extra["HTTP-Referer"] = env["OPENROUTER_REFERER"]
            extra["X-Title"] = "Personal Super AI"

        configs.append(
            ProviderConfig(
                name=name,
                kind=preset["kind"],
                base_url=base_url,
                api_key=key,
                models=models,
                vision_all=_bool(env.get(f"{p}_VISION"), preset.get("vision", False)),
                tools=_bool(env.get(f"{p}_TOOLS"), True),
                max_tokens_param=preset.get("max_tokens_param", "max_tokens"),
                reasoning_effort=preset.get("reasoning_effort", False),
                extra_headers=extra,
            )
        )
    return configs


# ---------------------------------------------------------------------------
# General settings
# ---------------------------------------------------------------------------
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"), env_file_encoding="utf-8", extra="ignore")

    # Application
    environment: str = "development"
    log_level: str = "INFO"
    database_path: str = "data/personal_super_ai.db"

    # Security
    cors_origins: str = "http://localhost:8000,http://127.0.0.1:8000,http://localhost:7860"
    api_auth_token: Optional[SecretStr] = None
    allow_anonymous_history: bool = False
    max_request_bytes: int = Field(40 * 1024 * 1024, ge=1024)
    rate_limit_per_minute: int = Field(0, ge=0)       # 0 = off (no artificial question limit)
    enable_compute_api: bool = False

    # Input limits
    max_question_chars: int = Field(20000, ge=100)
    max_history_items: int = Field(200, ge=0)
    max_context_chars: int = Field(80000, ge=1000)
    max_images: int = Field(4, ge=1, le=16)
    image_max_bytes: int = Field(12 * 1024 * 1024, ge=1024)
    image_max_side: int = Field(2048, ge=256, le=8192)
    image_max_pixels: int = Field(60_000_000, ge=1_000_000)

    # Model calls
    request_timeout_seconds: float = Field(240, ge=5)
    provider_timeout_seconds: float = Field(180, ge=5)
    provider_connect_timeout: float = Field(10, ge=1)
    provider_retries: int = Field(1, ge=0, le=3)
    max_tool_rounds: int = Field(6, ge=0, le=20)
    max_output_tokens: int = Field(4096, ge=256)
    max_output_tokens_reasoning: int = Field(8192, ge=256)
    temperature: Optional[float] = Field(None, ge=0, le=2)
    provider_priority: str = "anthropic,openai,gemini,groq,openrouter,ollama,custom"
    max_concurrent_requests: int = Field(8, ge=1)
    queue_timeout_seconds: float = Field(30, ge=1)

    # Engines (SymPy / NumPy / SciPy)
    engine_sandbox: bool = True
    engine_workers: int = Field(2, ge=1, le=16)
    engine_timeout_seconds: float = Field(20, ge=1)
    engine_memory_mb: int = Field(1536, ge=128)
    engine_cache_size: int = Field(512, ge=0)

    # Features. "offered" = the server can do it. "auto" = on without the client asking.
    offered_features: str = "math_verification,scientific_calc,web_research,long_term_memory"
    auto_features: str = "math_verification,scientific_calc"

    # Research / retrieval
    research_sources: str = "wikipedia,arxiv,crossref"
    contact_email: Optional[str] = None
    knowledge_timeout_seconds: float = Field(10, ge=1)
    knowledge_cache_ttl_seconds: int = Field(3600, ge=0)

    # Long-term memory
    max_memories_per_user: int = Field(1000, ge=10)

    @model_validator(mode="before")
    @classmethod
    def _drop_blank(cls, values):
        if isinstance(values, dict):
            return {k: v for k, v in values.items() if not (isinstance(v, str) and v.strip() == "")}
        return values

    # --- derived values -----------------------------------------------------
    @property
    def cors_origin_list(self) -> list[str]:
        return _csv(self.cors_origins)

    @property
    def offered_feature_set(self) -> set[str]:
        return set(_csv(self.offered_features))

    @property
    def auto_feature_set(self) -> set[str]:
        return set(_csv(self.auto_features))

    @property
    def research_source_list(self) -> list[str]:
        return [s.lower() for s in _csv(self.research_sources)]

    @property
    def provider_priority_list(self) -> list[str]:
        return [s.lower() for s in _csv(self.provider_priority)]

    @property
    def database_file(self) -> Path:
        if self.database_path == ":memory:":
            return Path(":memory:")
        p = Path(self.database_path)
        return p if p.is_absolute() else BASE_DIR / p

    @property
    def auth_token_value(self) -> Optional[str]:
        return self.api_auth_token.get_secret_value() if self.api_auth_token else None

    def provider_configs(self) -> list[ProviderConfig]:
        return load_provider_configs()


@lru_cache
def get_settings() -> Settings:
    return Settings()
