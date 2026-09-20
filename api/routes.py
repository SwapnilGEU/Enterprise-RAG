"""Endpoints.

Every handler here is a plain `def`, never `async def`, and that is the single
most important line in this file.

Everything the pipeline does is blocking: Qdrant over HTTP, Ollama over HTTP,
psycopg2. In an `async def` handler those calls block the event loop, so one
slow generation stalls *every* concurrent request including /health — the
service looks completely dead while doing one useful thing. A `def` handler is
run by FastAPI in a threadpool instead, which is exactly right for blocking
work. The cost is that handlers must be thread-safe, which is what
`api/state.py` is for.
"""

import logging
import time

from fastapi import APIRouter, HTTPException, Request

from api import metrics, settings
from api.schemas import (
    AgentRequest,
    AgentResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    ReadyResponse,
    Source,
)
from api.state import STATE

logger = logging.getLogger("api.routes")
router = APIRouter()


# --- liveness and readiness -------------------------------------------------


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Is the process alive? Consults nothing. Always fast, always 200."""
    return HealthResponse(
        service=settings.SERVICE_NAME,
        version=settings.SERVICE_VERSION,
        uptime_seconds=round(time.time() - STATE.started_at, 1),
    )


@router.get("/ready", response_model=ReadyResponse, tags=["ops"])
def ready(response: Request) -> ReadyResponse:
    """Can this instance serve? 503 until every required component is up.

    Returning 503 rather than 200-with-a-flag is deliberate: that is what a load
    balancer, a Docker healthcheck and later a Kubernetes readiness probe all
    understand without configuration.
    """
    # Retry whatever is currently down before answering. Warmup runs once, and
    # under Compose the API starts alongside Postgres and usually wins the race
    # — so without this the agent fails at boot and stays dead until someone
    # restarts the container, even though the database came up seconds later.
    # Rate limited and lock-guarded inside, and a no-op when everything is
    # healthy, so polling this endpoint stays cheap.
    STATE.recheck_failed()

    payload = ReadyResponse(
        ready=STATE.is_ready(),
        warming=STATE.warming,
        components=STATE.components(),
        missing_settings=STATE.missing_settings,
    )
    if not payload.ready:
        raise HTTPException(status_code=503, detail=payload.model_dump())
    return payload


# --- RAG --------------------------------------------------------------------


@router.post("/query", response_model=QueryResponse, tags=["rag"])
def query(body: QueryRequest, request: Request) -> QueryResponse:
    """Retrieve, then answer, with citations.

    Needs Qdrant and Ollama. Does not touch Postgres or the agent, so it keeps
    serving when those are down — which is the whole reason readiness is
    per-component.
    """
    request_id = request.state.request_id
    started = time.perf_counter()

    if not STATE.qdrant.ready or not STATE.ollama.ready:
        # Logged, not merely returned: a 503 that exists only as a status code
        # leaves its reason in a response body nobody kept.
        detail = _unready_detail(["qdrant", "ollama"])
        logger.warning("refused /query — dependency down: %s", detail,
                       extra={"event": "refused", "reason": "dependency_down",
                              "detail": detail})
        raise HTTPException(status_code=503, detail=detail)

    if not STATE.acquire_generation_slot("/query"):
        # An honest refusal beats an invisible queue: the caller can back off,
        # and the number of these is the signal that the generator is the
        # bottleneck rather than the service.
        logger.warning(
            "refused /query — no generation slot within %ss",
            settings.GENERATION_QUEUE_TIMEOUT,
            extra={"event": "refused", "reason": "queue_timeout",
                   "timeout_s": settings.GENERATION_QUEUE_TIMEOUT,
                   "max_concurrent": settings.MAX_CONCURRENT_GENERATIONS},
        )
        raise HTTPException(
            status_code=503,
            detail=(
                f"No generation slot within {settings.GENERATION_QUEUE_TIMEOUT}s "
                f"({settings.MAX_CONCURRENT_GENERATIONS} concurrent max). Retry shortly."
            ),
        )

    try:
        from src.generation import generate_answer

        result = generate_answer(body.question, top_k=body.top_k)
    except Exception as exc:  # noqa: BLE001
        logger.exception("query failed", extra={"question": body.question[:120]})
        raise HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}") from exc
    finally:
        STATE.release_generation_slot("/query")

    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    sources = _to_sources(result.get("sources") or [], include_text=body.include_context)

    # A degraded answer is an HTTP 200 and zero sources is a fast HTTP 200, so
    # neither is visible in an error-rate panel. This is where they become
    # countable.
    metrics.answer(
        "/query",
        degraded=bool(result.get("degraded")),
        failure_stage=result.get("failure_stage"),
        n_sources=len(sources),
    )

    top = sources[0].document if sources else None
    logger.info(
        "answered /query in %sms from %d source(s)%s",
        latency_ms, len(sources), " [DEGRADED]" if result.get("degraded") else "",
        extra={
            "event": "query_answered",
            "latency_ms": latency_ms,
            "n_sources": len(sources),
            "degraded": bool(result.get("degraded")),
            "failure_stage": result.get("failure_stage"),
            "top_document": top,
            # Truncated to the same bound the failure path already uses. This
            # is what makes a Loki search useful: without it you can find that
            # a request was slow but not what was asked.
            "question": body.question[:120],
        },
    )

    return QueryResponse(
        answer=result.get("answer", ""),
        sources=sources,
        degraded=bool(result.get("degraded")),
        failure_stage=result.get("failure_stage"),
        request_id=request_id,
        latency_ms=latency_ms,
    )


# --- agent ------------------------------------------------------------------


@router.post("/agent", response_model=AgentResponse, tags=["agent"])
def agent(body: AgentRequest, request: Request) -> AgentResponse:
    """Route the question through the LangGraph agent.

    Additionally needs Postgres — `build_agent()` constructs the SQL toolkit at
    compile time — so this is the endpoint that 503s while /query stays up.
    """
    if not settings.ENABLE_AGENT:
        raise HTTPException(
            status_code=404,
            detail="Agent endpoint disabled (API_ENABLE_AGENT=false).",
        )

    request_id = request.state.request_id
    started = time.perf_counter()

    if not STATE.agent.ready:
        logger.warning("refused /agent — agent unavailable: %s",
                       STATE.agent.error or "still warming",
                       extra={"event": "refused", "reason": "agent_unavailable",
                              "detail": STATE.agent.error})
        raise HTTPException(
            status_code=503,
            detail=(
                f"Agent unavailable: {STATE.agent.error or 'still warming'}. "
                "/query is unaffected and may still work."
            ),
        )

    if not STATE.acquire_generation_slot("/agent"):
        logger.warning(
            "refused /agent — no generation slot within %ss",
            settings.GENERATION_QUEUE_TIMEOUT,
            extra={"event": "refused", "reason": "queue_timeout",
                   "timeout_s": settings.GENERATION_QUEUE_TIMEOUT},
        )
        raise HTTPException(
            status_code=503,
            detail=f"No generation slot within {settings.GENERATION_QUEUE_TIMEOUT}s. Retry shortly.",
        )

    try:
        from langchain_core.messages import HumanMessage

        graph = STATE.get_agent()
        state = graph.invoke(
            {
                "session_id": body.session_id,
                "user_query": body.question,
                "start_time": time.time(),
                "messages": [HumanMessage(content=body.question)],
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("agent failed", extra={"question": body.question[:120]})
        raise HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}") from exc
    finally:
        STATE.release_generation_slot("/agent")

    latency_ms = round((time.perf_counter() - started) * 1000, 1)

    # dict.fromkeys dedupes while preserving call order — the same walk
    # evaluation/eval_agent.py and mlflow/agentflow.py use, so the API reports
    # tool usage identically to how it is measured.
    #
    # The automatic knowledge-base pre-search is reported separately rather than
    # mixed in: it happens on every question, so folding it into `tools_used`
    # would put `rag_tool` on every answer and drown the one fact this field
    # exists to convey — what the model actually decided to call.
    from src.agent import is_prefetch_call

    calls = [
        call
        for message in state["messages"]
        if getattr(message, "tool_calls", None)
        for call in message.tool_calls
    ]
    tools_used = list(dict.fromkeys(c["name"] for c in calls if not is_prefetch_call(c)))
    prefetched = any(is_prefetch_call(c) for c in calls)

    logger.info(
        "answered /agent in %sms — tools: %s%s",
        latency_ms, ", ".join(tools_used) or "none",
        " (kb pre-searched)" if prefetched else "",
        extra={
            "event": "agent_answered",
            "latency_ms": latency_ms,
            "tools_used": tools_used,
            "n_tools": len(tools_used),
            "retrieval_prefetched": prefetched,
            "session_id": body.session_id,
            "question": body.question[:120],
        },
    )

    metrics.answer("/agent", degraded=False)
    metrics.tool_calls(tools_used, prefetched)

    return AgentResponse(
        answer=_message_text(state["messages"][-1].content),
        tools_used=tools_used,
        retrieval_prefetched=prefetched,
        request_id=request_id,
        latency_ms=latency_ms,
    )


# --- helpers ----------------------------------------------------------------


def _unready_detail(names: list[str]) -> str:
    parts = []
    for name in names:
        component = getattr(STATE, name)
        if not component.ready:
            parts.append(f"{name}: {component.error or 'still warming'}")
    return "; ".join(parts) or "not ready"


def _to_sources(raw: list[dict], include_text: bool) -> list[Source]:
    from src.payload import format_source

    sources = []
    for meta in raw:
        sources.append(
            Source(
                document=str(meta.get("document", "")),
                section_heading=str(meta.get("section_heading") or meta.get("section_title") or ""),
                page_label=str(meta.get("page_label", "")),
                chunk_id=str(meta.get("chunk_id", "")),
                score=meta.get("score"),
                citation=format_source(meta),
                text=str(meta.get("text", "")) if include_text else None,
            )
        )
    return sources


def _message_text(content) -> str:
    """LangChain message content is usually a str but can be a list of blocks."""
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
        return " ".join(p for p in parts if p).strip()
    return str(content).strip()
