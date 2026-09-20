"""The metric instruments, and the only place that knows their names.

Design notes worth keeping
--------------------------
**Every function here is a no-op until `init()` runs**, and `init()` only runs
when `configure_metrics()` found an OTLP endpoint. So `api/state.py` and
`api/routes.py` can call these unconditionally: no import guard at the call
site, no `if metrics_enabled:`, and the service behaves identically when
telemetry is off.

**Instruments are created in `init()`, not at import.** A meter obtained before
the MeterProvider is installed binds to a no-op proxy, and the instruments go
quietly nowhere — which looks exactly like "the collector isn't receiving".

**Buckets are set by Views in api/observability.py, not here.** The SDK's
default histogram boundaries top out at 10 seconds. A generation on a 4b model
over a 6GB card regularly exceeds that, so without explicit boundaries nearly
everything lands in the +Inf bucket and `histogram_quantile` returns a number
that looks plausible and is fiction.

**What is deliberately NOT measured here:** request rate, latency and error
rate. The FastAPI and httpx instrumentors already emit
`http.server.request.duration` and `http.client.request.duration`, which carry
route, status and the Qdrant/Ollama split for free. Re-deriving them would
create two sources of truth that disagree.

Cardinality rule: attributes are endpoint, tool name, component name and
booleans — all bounded. Never the question, `request_id` or `session_id`.
"""

import logging

logger = logging.getLogger("api.metrics")

_ready = False
_i: dict = {}


def init() -> bool:
    """Create the instruments. Called by configure_metrics() once the provider
    is installed. Idempotent; returns False if metrics stay off."""
    global _ready
    if _ready:
        return True
    try:
        from opentelemetry import metrics
    except ImportError:
        return False

    m = metrics.get_meter("enterprise-rag-api")

    # --- the generation bottleneck ---------------------------------------
    # One 4b model, two slots. Queueing is the NORMAL state under any load, so
    # without this a user's 8s request shows 2s of Ollama and 6s unexplained.
    _i["queue_wait"] = m.create_histogram(
        "rag.generation.queue_wait", unit="s",
        description="Time waiting for one of API_MAX_CONCURRENT_GENERATIONS slots")
    _i["rejected"] = m.create_counter(
        "rag.generation.rejected",
        description="Requests given up on after API_GENERATION_QUEUE_TIMEOUT (503)")
    _i["active"] = m.create_up_down_counter(
        "rag.generation.active",
        description="Generation slots currently held")

    # --- answers ----------------------------------------------------------
    # A degraded answer is an HTTP 200. Error rate cannot see it, which is the
    # entire reason this counter exists separately from the http metrics.
    _i["answers"] = m.create_counter(
        "rag.answers",
        description="Answers returned, split by endpoint and whether degraded")
    # Zero retrieved sources is a silent failure: fast, 200, and useless.
    _i["sources"] = m.create_histogram(
        "rag.retrieval.sources", unit="{chunk}",
        description="Sources returned with an answer")

    # --- agent ------------------------------------------------------------
    _i["tool_calls"] = m.create_counter(
        "rag.agent.tool_calls",
        description="Tool calls the model chose, by tool (prefetch excluded)")

    # --- dependency health -------------------------------------------------
    # Observable, not a push: ComponentStatus already holds the truth, so a
    # callback reads it at collection time rather than every transition site
    # having to remember to report. recheck_failed() flapping becomes visible.
    def _components(_options):
        from opentelemetry.metrics import Observation

        try:
            from api.state import STATE
        except Exception:  # noqa: BLE001
            return []
        out = []
        for c in (STATE.qdrant, STATE.ollama, STATE.agent):
            out.append(Observation(1 if c.ready else 0,
                                   {"component": c.name, "required": c.required}))
        return out

    _i["component_ready"] = m.create_observable_gauge(
        "rag.component.ready", callbacks=[_components],
        description="1 when a dependency is usable, 0 when it is not")

    _ready = True
    logger.info("metric instruments created")
    return True


# --- recording helpers; all no-ops until init() ----------------------------


def queue_wait(endpoint: str, seconds: float, acquired: bool) -> None:
    if _ready:
        _i["queue_wait"].record(
            seconds, {"endpoint": endpoint, "outcome": "acquired" if acquired else "timeout"})
        if not acquired:
            _i["rejected"].add(1, {"endpoint": endpoint})


def slot_held(endpoint: str, delta: int) -> None:
    """+1 on acquire, -1 on release. Current depth = sum of this series."""
    if _ready:
        _i["active"].add(delta, {"endpoint": endpoint})


def answer(endpoint: str, degraded: bool, failure_stage: str | None = None,
           n_sources: int | None = None) -> None:
    if not _ready:
        return
    attrs = {"endpoint": endpoint, "degraded": bool(degraded)}
    if degraded and failure_stage:
        attrs["failure_stage"] = str(failure_stage)
    _i["answers"].add(1, attrs)
    if n_sources is not None:
        _i["sources"].record(n_sources, {"endpoint": endpoint})


def tool_calls(tools: list[str], prefetched: bool) -> None:
    """Prefetch is reported as its own pseudo-tool rather than mixed into the
    list, for the same reason routes.py keeps them apart: it fires on every
    question and would otherwise drown what the model actually decided."""
    if not _ready:
        return
    for tool in tools:
        _i["tool_calls"].add(1, {"tool": tool})
    if prefetched:
        _i["tool_calls"].add(1, {"tool": "_prefetch"})
