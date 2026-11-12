"""Personal Super AI - FastAPI application.

Run (from the backend/ folder):
    uvicorn app:app --host 0.0.0.0 --port 8000

This module wires the services together and exposes the HTTP API. The logic lives in:
    ai/            providers, routing, prompts, tools, orchestration
    engines/       SymPy / NumPy / SciPy computation (sandboxed)
    services/      memory, image analysis, knowledge retrieval, chat pipeline
"""
from __future__ import annotations

import functools
import hmac
import logging
import re
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Path as ApiPath, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from ai.base import AIProvider
from ai.orchestrator import Orchestrator
from ai.providers import build_providers
from ai.router import ModelRegistry, Router
from ai.tools import ToolRegistry
from config import APP_VERSION, Settings, get_settings
from database import Database
from engines import EngineError, EngineHub
from models import (
    AnalyzeImageRequest, ChatRequest, ChatResponse, ComputeRequest, ComputeResponse, ConversationDetail,
    DeleteResponse, HealthResponse, HistoryResponse, ImageAnalysisResponse, MemoriesResponse, ModelInfo, ModelsResponse,
    PreferencesBody, PreferencesResponse,
)
from services.chat import ChatService, error_for
from services.image_analysis import ImageService, RawImage
from services.knowledge import KnowledgeService
from services.memory import MemoryService

log = logging.getLogger("psai.app")

_CLIENT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
_CONVERSATION_PATTERN = r"^[A-Za-z0-9_-]{6,64}$"
KNOWN_FEATURES = ["math_verification", "scientific_calc", "web_research", "long_term_memory",
                  "pdf_understanding", "screen_understanding"]


# --- wiring ---------------------------------------------------------------------------
class RateLimiter:
    """Optional per-client sliding window (RATE_LIMIT_PER_MINUTE). Off by default."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._hits: dict[str, deque] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        if self.per_minute <= 0:
            return True
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] > 60:
            hits.popleft()
        if len(hits) >= self.per_minute:
            return False
        hits.append(now)
        if len(self._hits) > 20000:  # bound memory
            self._hits.clear()
        return True


@dataclass
class Container:
    settings: Settings
    db: Database
    memory: MemoryService
    engines: EngineHub
    knowledge: Optional[KnowledgeService]
    registry: ModelRegistry
    chat: ChatService
    providers: list[AIProvider]
    limiter: RateLimiter


def get_container(request: Request) -> Container:
    return request.app.state.container


async def access_guard(request: Request, authorization: Optional[str] = Header(None)) -> None:
    """Optional bearer token (API_AUTH_TOKEN). A static web page cannot keep a secret, so this is meant for
    your own servers or apps; browsers are protected by CORS and, if enabled, the rate limit."""
    token = get_container(request).settings.auth_token_value
    if token:
        supplied = (authorization or "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(supplied.encode(), token.encode()):
            raise HTTPException(401, "Missing or invalid access token.", headers={"WWW-Authenticate": "Bearer"})


async def rate_guard(request: Request) -> None:
    c = get_container(request)
    key = request.client.host if request.client else "unknown"
    if not c.limiter.allow(key):
        raise HTTPException(429, "Too many requests from this address. Please slow down.", headers={"Retry-After": "60"})


def client_id(request: Request, x_client_id: Optional[str] = Header(None)) -> Optional[str]:
    """Identifies the browser/device for stored history and memory. The frontend sends a random X-Client-Id."""
    if x_client_id:
        if not _CLIENT_ID_RE.match(x_client_id):
            raise HTTPException(400, "X-Client-Id must be 8-64 characters: letters, digits, '_' or '-'.")
        return x_client_id
    return "anonymous" if get_container(request).settings.allow_anonymous_history else None


def require_client_id(user_id: Optional[str] = Depends(client_id)) -> str:
    if not user_id:
        raise HTTPException(400, "Send an X-Client-Id header to use stored history, preferences and memory.")
    return user_id


def api_errors(fn):
    """Turn expected failures into JSON errors from INSIDE the route, so CORS headers are still added."""

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except (HTTPException, RequestValidationError):
            raise
        except EngineError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:  # noqa: BLE001
            mapped = error_for(exc)
            if mapped is None:
                log.exception("Unhandled error in %s", fn.__name__)
                return JSONResponse({"error": "Internal server error."}, status_code=500)
            status, message, headers = mapped
            return JSONResponse({"error": message}, status_code=status, headers=headers)

    return wrapper


SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}


def create_app(settings: Optional[Settings] = None, providers: Optional[list[AIProvider]] = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    production = settings.environment.lower() == "production"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db = Database(settings.database_file)
        await db.connect()
        used_providers = providers if providers is not None else build_providers(settings)
        engines = EngineHub(settings)
        await engines.start()
        knowledge = KnowledgeService(settings) if settings.research_source_list else None
        memory = MemoryService(db, settings)
        registry = ModelRegistry(used_providers, settings.provider_priority_list)
        orchestrator = Orchestrator(settings, registry, Router(registry), ToolRegistry(engines, knowledge, memory))
        chat = ChatService(settings, memory, ImageService(settings), orchestrator)
        app.state.container = Container(settings, db, memory, engines, knowledge, registry, chat, used_providers,
                                        RateLimiter(settings.rate_limit_per_minute))
        if not registry.specs:
            log.warning("No AI provider is configured. Add an API key and model to backend/.env (see .env.example).")
        if settings.cors_origins.strip() == "*":
            log.warning("CORS_ORIGINS=* allows any website to call this API. Restrict it in production.")
        try:
            yield
        finally:
            for p in used_providers:
                await p.aclose()
            if knowledge:
                await knowledge.aclose()
            await engines.stop()
            await db.close()

    app = FastAPI(title="Personal Super AI", version=APP_VERSION, lifespan=lifespan,
                  docs_url=None if production else "/docs", redoc_url=None, openapi_url=None if production else "/openapi.json")
    app.state.settings = settings

    # --- middleware (the last one added is the outermost) --------------------------------
    @app.middleware("http")
    async def request_guard(request: Request, call_next):
        request_id = uuid.uuid4().hex[:12]
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > settings.max_request_bytes:
            return JSONResponse({"error": "The request is too large."}, status_code=413)
        started = time.monotonic()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        log.info("%s %s -> %s in %dms [%s]", request.method, request.url.path, response.status_code,
                 (time.monotonic() - started) * 1000, request_id)  # paths only: never bodies, keys or ids
        return response

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.cors_origins.strip() == "*" else settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-Client-Id", "Accept"],
        expose_headers=["X-Request-ID", "Retry-After"],
        max_age=600,
    )

    # --- error shapes the frontend understands ({"error": "..."}) -------------------------
    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        problems = [f"{'.'.join(str(p) for p in e.get('loc', []) if p != 'body')}: {e.get('msg', 'invalid')}" for e in exc.errors()]
        return JSONResponse({"error": "Invalid request: " + "; ".join(problems[:5])}, status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def http_handler(request: Request, exc: StarletteHTTPException):
        return JSONResponse({"error": str(exc.detail)}, status_code=exc.status_code, headers=getattr(exc, "headers", None))

    guards = [Depends(access_guard), Depends(rate_guard)]

    # --- system ---------------------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    async def root():
        return FileResponse("index.html")

    @app.get("/api/health", response_model=HealthResponse)
    async def health(c: Container = Depends(get_container)):
        db_ok, engines_ok = await c.db.ping(), c.engines.healthy
        providers = [{"name": p.name, "models": [s.key for s in p.specs], "vision": any(s.vision for s in p.specs)}
                     for p in c.providers]
        notes = []
        if not providers:
            notes.append("No AI provider configured: chat requests will fail until one is added.")
        return HealthResponse(
            status="ok" if (db_ok and engines_ok and providers) else "degraded", version=APP_VERSION, time=time.time(),
            database=db_ok, engines=engines_ok, providers=providers,
            research_sources=c.knowledge.source_names if c.knowledge else [],
            features=sorted(c.settings.offered_feature_set), notes=notes)

    @app.get("/api/models", response_model=ModelsResponse, dependencies=guards)
    async def models(c: Container = Depends(get_container)):
        offered = c.settings.offered_feature_set
        features = {f: (f in offered) for f in KNOWN_FEATURES}
        features["web_research"] = features["web_research"] and bool(c.knowledge and c.knowledge.available)
        features["pdf_understanding"] = features["screen_understanding"] = False  # not implemented yet
        return ModelsResponse(
            models=[ModelInfo(key=s.key, provider=s.provider, model=s.model, designations=sorted(s.designations),
                              vision=s.vision, tools=s.tools) for s in c.registry.specs],
            features=features)

    # --- chat -----------------------------------------------------------------------------
    @app.post("/api/chat", response_model=ChatResponse, dependencies=guards)
    @api_errors
    async def chat(body: ChatRequest, c: Container = Depends(get_container), user_id: Optional[str] = Depends(client_id)):
        prep = await c.chat.prepare(body, user_id)
        if body.stream:
            return StreamingResponse(c.chat.stream(prep), media_type="text/event-stream", headers=SSE_HEADERS)
        return await c.chat.respond(prep)

    @app.post("/api/analyze-image", response_model=ImageAnalysisResponse, dependencies=guards)
    @api_errors
    async def analyze_image(request: Request, c: Container = Depends(get_container),
                            user_id: Optional[str] = Depends(client_id)):
        """Accepts multipart/form-data (field `file` or `files`, plus question/subject/level/mode/model/task/source)
        or JSON with base64 `images`. Same options as /api/chat."""
        content_type = request.headers.get("content-type", "").lower()
        raw_images: list[RawImage] = []
        try:
            if content_type.startswith("multipart/form-data"):
                form = await request.form(max_files=c.settings.max_images, max_fields=30)
                for key in ("file", "files", "image", "images"):
                    for item in form.getlist(key):
                        if isinstance(item, UploadFile):
                            raw_images.append(RawImage(item.filename or "image", await item.read(c.settings.image_max_bytes + 1)))
                fields = {k: form.get(k) for k in ("question", "subject", "level", "mode", "model", "task", "source",
                                                   "conversation_id") if isinstance(form.get(k), str)}
                fields["stream"] = str(form.get("stream", "")).lower() in {"1", "true", "yes"}
                body = AnalyzeImageRequest(**fields)
            else:
                try:
                    data = await request.json()
                except ValueError:
                    raise HTTPException(400, "The body must be JSON or multipart/form-data.") from None
                body = AnalyzeImageRequest.model_validate(data)
        except ValidationError as exc:
            raise RequestValidationError(exc.errors()) from exc

        prep = await c.chat.prepare(body, user_id, raw_images=raw_images, task=body.task)
        if body.stream:
            return StreamingResponse(c.chat.stream(prep), media_type="text/event-stream", headers=SSE_HEADERS)
        return await c.chat.respond(prep)

    # --- history ----------------------------------------------------------------------------
    @app.get("/api/history", response_model=HistoryResponse, dependencies=guards)
    async def list_history(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0),
                           user_id: str = Depends(require_client_id), c: Container = Depends(get_container)):
        rows, total = await c.memory.list_conversations(user_id, limit, offset)
        return HistoryResponse(conversations=rows, total=total, limit=limit, offset=offset)

    @app.get("/api/history/{conversation_id}", response_model=ConversationDetail, dependencies=guards)
    async def get_history(conversation_id: str = ApiPath(..., pattern=_CONVERSATION_PATTERN),
                          user_id: str = Depends(require_client_id), c: Container = Depends(get_container)):
        conversation = await c.memory.get_conversation(user_id, conversation_id)
        if conversation is None:
            raise HTTPException(404, "Conversation not found.")
        return conversation

    @app.delete("/api/history/{conversation_id}", response_model=DeleteResponse, dependencies=guards)
    async def delete_history(conversation_id: str = ApiPath(..., pattern=_CONVERSATION_PATTERN),
                             user_id: str = Depends(require_client_id), c: Container = Depends(get_container)):
        if not await c.memory.delete_conversation(user_id, conversation_id):
            raise HTTPException(404, "Conversation not found.")
        return DeleteResponse(deleted=True, conversation_id=conversation_id)

    # --- preferences and long-term memory ----------------------------------------------------
    @app.get("/api/preferences", response_model=PreferencesResponse, dependencies=guards)
    async def get_preferences(user_id: str = Depends(require_client_id), c: Container = Depends(get_container)):
        return PreferencesResponse(preferences=await c.memory.get_preferences(user_id))

    @app.put("/api/preferences", response_model=PreferencesResponse, dependencies=guards)
    async def put_preferences(body: PreferencesBody, user_id: str = Depends(require_client_id),
                              c: Container = Depends(get_container)):
        return PreferencesResponse(preferences=await c.memory.set_preferences(user_id, body.preferences))

    @app.get("/api/memories", response_model=MemoriesResponse, dependencies=guards)
    async def list_memories(user_id: str = Depends(require_client_id), c: Container = Depends(get_container)):
        return MemoriesResponse(memories=await c.memory.list_memories(user_id))

    @app.delete("/api/memories/{memory_id}", dependencies=guards)
    async def delete_memory(memory_id: int = ApiPath(..., ge=1), user_id: str = Depends(require_client_id),
                            c: Container = Depends(get_container)):
        if not await c.memory.delete_memory(user_id, memory_id):
            raise HTTPException(404, "Memory not found.")
        return {"deleted": True, "id": memory_id}

    # --- direct computation (disabled unless ENABLE_COMPUTE_API=true) --------------------------
    @app.post("/api/compute", response_model=ComputeResponse, dependencies=guards)
    @api_errors
    async def compute(body: ComputeRequest, c: Container = Depends(get_container)):
        if not c.settings.enable_compute_api:
            raise HTTPException(404, "The compute API is disabled. Set ENABLE_COMPUTE_API=true to enable it.")
        return ComputeResponse(ok=True, result=await c.engines.run(body.engine, body.operation, body.args))

    # --- static frontend files (serves your HTML, CSS, JS) ---
    app.mount("/", StaticFiles(directory=".", html=True), name="static")

    return app


app = create_app()

if __name__ == "__main__":  # python app.py  (development convenience)
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
