"""The FastAPI app.

    uvicorn api.main:app --reload            # development
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Boots fast, on purpose
----------------------
`lifespan` runs *before* uvicorn accepts connections, so anything slow put here
delays the port opening. The slow things — the Ollama model load, and
`build_agent()` reflecting the Postgres schema — therefore run in a background
thread instead. The server is listening in well under a second, and `/ready`
reports 503 until warming finishes.

That is also why readiness is a separate endpoint from health: "up" and "able
to serve" are genuinely different states here, and collapsing them would mean
either a slow boot or lying about being ready.
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load .env before anything reads config — src.config resolves env vars at
# import time via field(default_factory=...), so a late load silently yields a
# Config full of blanks.
for env_path in (PROJECT_ROOT / ".env", PROJECT_ROOT / "api" / ".env"):
    if env_path.exists():
        load_dotenv(env_path)

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from api import settings  # noqa: E402
from api.logging_config import configure_logging, request_id_var  # noqa: E402
from api.routes import router  # noqa: E402
from api.state import STATE  # noqa: E402
from api.observability import (  # noqa: E402
    configure_log_export,
    configure_metrics,
    configure_tracing,
    instrument_app,
)

configure_logging()
# Before the app exists, so the instrumentors below attach to the provider
# this installs. A no-op unless an OTLP endpoint is configured.
configure_tracing()
configure_metrics()
# After configure_logging(), which sets handlers and propagate=False on
# uvicorn/rag — this appends to those same loggers rather than only to root.
configure_log_export()
logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "starting",
        extra={
            "service": settings.SERVICE_NAME,
            "version": settings.SERVICE_VERSION,
            "agent_enabled": settings.ENABLE_AGENT,
            "max_concurrent_generations": settings.MAX_CONCURRENT_GENERATIONS,
        },
    )
    if settings.WARMUP:
        # Background, not awaited: see the module docstring.
        STATE.warm_in_background()
    else:
        logger.info("warmup disabled — first request will pay for it")
    yield
    logger.info("shutting down")


app = FastAPI(
    title="Enterprise RAG API",
    version=settings.SERVICE_VERSION,
    description=(
        "Hybrid retrieval over Qdrant with a local Ollama generator, plus a "
        "LangGraph agent. /query needs Qdrant and Ollama; /agent also needs Postgres "
        "and degrades independently."
    ),
    lifespan=lifespan,
)

# Server spans for every request bar /health, plus client spans for the
# Qdrant and Ollama calls. Must follow configure_tracing().
instrument_app(app)

if settings.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Give every request an id, put it in the log context, return it in a header.

    This is the thread that ties a user's complaint to a log line to, later, a
    Tempo trace. An inbound X-Request-ID is honoured so the id survives a proxy
    or a caller that already has one.

    The ContextVar is set here in the event loop, and FastAPI copies the context
    into the threadpool when it runs a sync endpoint — so handler logs carry the
    id without it being passed down by hand.
    """
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = request_id
    token = request_id_var.set(request_id)
    started = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "unhandled error",
            extra={"method": request.method, "path": request.url.path},
        )
        response = JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": request_id},
        )
    finally:
        request_id_var.reset(token)

    duration_ms = round((time.perf_counter() - started) * 1000, 1)
    response.headers["X-Request-ID"] = request_id

    # One structured access line per request. /health is excluded because a
    # container healthcheck every few seconds would otherwise be most of what
    # Loki stores.
    if request.url.path != "/health":
        logger.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": duration_ms,
                "request_id": request_id,
            },
        )
    return response


app.include_router(router)


@app.get("/", include_in_schema=False)
def root():
    return {
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "docs": "/docs",
        "endpoints": ["/health", "/ready", "/query"] + (["/agent"] if settings.ENABLE_AGENT else []),
    }
