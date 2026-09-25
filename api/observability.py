"""OpenTelemetry tracing for the API — one place, or none.

Why this module exists at all
-----------------------------
Exactly one place may configure the global tracer provider. Two places means
two span processors on one provider and every span exported twice. That raises
no error; it just makes `avg(span:duration)` in TraceQL quietly wrong. So all
provider setup lives here and nowhere else.

What this does NOT do: MLflow
-----------------------------
An earlier note in claude/fastapi-as-built.md said MLflow's `@mlflow.trace`
spans "nest under whatever provider exists", so they would land inside the HTTP
server span automatically. Measured on mlflow 3.16.1, that is false:

    inside an active OTel server span, mlflow.start_span() produces a span with
    a DIFFERENT trace_id and parent=None, and it never reaches the global
    provider's exporter.

MLflow keeps its own TracerProvider and ignores the ambient OTel context. So
MLflow spans and OTel spans are two pipelines producing two unrelated traces.
Pointing both at Tempo yields orphan single-span traces beside the real ones,
not one tree.

Hence the split, decided by environment and nothing else:

    serving   OTEL_EXPORTER_OTLP_TRACES_ENDPOINT set
              MLFLOW_TRACING_ENABLED=false
              -> Tempo gets the tree; mlflow.db is never written to, which is
                 also what silences the mlflow-artifacts URI warning

    eval      MLFLOW_TRACKING_URI set, no OTLP endpoint
              -> MLflow gets the trace that DeepEval reads retrieval_context
                 and tools_called off. Do not set an OTLP endpoint here: MLflow
                 treats one as EXCLUSIVE and would send the trace to the
                 collector instead of to MLflow, and every scorer would report
                 the same uninformative "N/N failed" as a spent judge quota.
                 If you ever want both, MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT=true.

Where the spans come from
-------------------------
FastAPI instrumentation gives one server span per request. httpx and requests
instrumentation give a client span for every Qdrant search and every Ollama
generation — which is where the time actually goes, so the breakdown is honest
without a single manual span in src/.

/health is excluded, for the same reason it is already excluded from the access
log: a container healthcheck every few seconds would otherwise be most of what
Tempo stores.

With no endpoint configured every function here is a no-op and the service runs
exactly as before. Tracing is never a reason the API fails to start.
"""

import logging
import os

from api import settings

logger = logging.getLogger("api.observability")

_provider_configured = False
_app_instrumented = False
_metrics_configured = False
_logs_configured = False


def otlp_endpoint() -> str:
    """The configured OTLP traces endpoint, or "" when tracing is off.

    The per-signal variable wins over the general one, which is the convention
    the OTel spec sets and the exporters themselves follow.
    """
    return (
        os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        or ""
    ).strip()


def _protocol() -> str:
    return (
        os.environ.get("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL")
        or os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL")
        or "grpc"
    ).strip().lower()


def configure_tracing() -> bool:
    """Install the global tracer provider. Idempotent.

    Returns True when tracing is live, False when it is off for any reason —
    no endpoint, packages absent, or somebody else already owns the provider.
    """
    global _provider_configured
    if _provider_configured:
        return True

    endpoint = otlp_endpoint()
    if not endpoint:
        logger.info("tracing off: no OTLP endpoint configured")
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning("tracing requested but opentelemetry-sdk is not installed")
        return False

    # The protocol decides the exporter AND the port: gRPC is 4317, HTTP is
    # 4318. Giving one the other's port fails as a connection error at export
    # time, long after startup, which is a miserable thing to debug.
    protocol = _protocol()
    try:
        if protocol.startswith("http"):
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
        else:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
    except ImportError:
        logger.warning("tracing requested but opentelemetry-exporter-otlp is not installed")
        return False

    # Refuse to stack a second processor on somebody else's provider. A real
    # SDK provider already in place means this function ran twice, or an
    # auto-instrumentation agent (opentelemetry-instrument) set one up.
    existing = trace.get_tracer_provider()
    if isinstance(existing, TracerProvider):
        logger.warning(
            "a tracer provider is already installed — leaving it alone to avoid "
            "exporting every span twice"
        )
        _provider_configured = True
        return True

    resource = _resource(Resource)
    provider = TracerProvider(resource=resource)
    # Batch, not Simple: Simple exports on the calling thread and would add the
    # collector's round trip to every request's latency.
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)

    _provider_configured = True
    _disable_mlflow_tracing()
    logger.info(
        "tracing on",
        extra={"endpoint": endpoint, "protocol": protocol,
               "service": resource.attributes.get("service.name")},
    )
    return True


def _signal_endpoint(signal: str, http_path: str):
    """Endpoint override for a non-trace signal, or None to let the exporter
    read the environment itself.

    The trap this exists for: each OTLP exporter reads its OWN variables.
    OTLPMetricExporter() looks at OTEL_EXPORTER_OTLP_METRICS_ENDPOINT, then
    OTEL_EXPORTER_OTLP_ENDPOINT, then falls back to **localhost:4317** — it
    never looks at OTEL_EXPORTER_OTLP_TRACES_ENDPOINT. So configuring only the
    traces variable, which is the natural thing to do after wiring traces
    first, sends metrics and logs to localhost *inside the api container* —
    i.e. to itself. Traces appear in Grafana, metrics and logs never do, and
    nothing in the logs says why.

    Preferring OTEL_EXPORTER_OTLP_ENDPOINT (the general one) avoids this
    entirely, which is what .env.example and docker-compose.yml now use. This
    function covers the case where someone sets only the traces variable.
    """
    if os.environ.get("OTEL_EXPORTER_OTLP_%s_ENDPOINT" % signal) or os.environ.get(
        "OTEL_EXPORTER_OTLP_ENDPOINT"
    ):
        return None  # properly configured; the exporter handles paths itself
    traces = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    if not traces:
        return None
    logger.info("deriving the %s endpoint from OTEL_EXPORTER_OTLP_TRACES_ENDPOINT; "
                "set OTEL_EXPORTER_OTLP_ENDPOINT to configure all signals at once",
                signal.lower())
    if _protocol().startswith("http"):
        base = traces.rstrip("/")
        if base.endswith("/v1/traces"):
            base = base[: -len("/v1/traces")]
        return base + http_path
    return traces


def _resource(Resource):
    """One identity for traces, metrics and logs.

    service.name becomes resource.service.name in TraceQL, the `service_name`
    label in Loki and the `job`/`service_name` label in Prometheus — it is what
    ties the three panes to the same service. OTEL_SERVICE_NAME wins if set, so
    the same image can report under a different name during an eval run.
    """
    return Resource.create(
        {
            "service.name": os.environ.get("OTEL_SERVICE_NAME", settings.SERVICE_NAME),
            "service.version": settings.SERVICE_VERSION,
        }
    )


def _disable_mlflow_tracing() -> None:
    """Stop MLflow tracing in this process, once OTel tracing is live.

    Done in code rather than with MLFLOW_TRACING_ENABLED in .env on purpose:
    mlflow/ragflow.py, agentflow.py and retrievalflow.py all load the repo-root
    .env themselves, so a `false` there would disable tracing during EVAL runs
    too — and DeepEval builds retrieval_context and tools_called from the MLflow
    trace, so every scorer would come back "N/N failed" with no hint why. This
    function only ever runs inside the API process.

    mlflow absent (it is not in requirements-serve.txt, so not in the image) is
    the normal case and nothing needs doing: the soft import in src/retrieval.py
    already makes @mlflow.trace a no-op.
    """
    try:
        import mlflow
    except ImportError:
        return
    try:
        tracing = getattr(mlflow, "tracing", None)
        disable = getattr(tracing, "disable", None)
        if callable(disable):
            disable()
        logger.info("mlflow tracing disabled in the serving process — traces go to Tempo")
    except Exception:
        logger.exception("could not disable mlflow tracing")


def instrument_app(app) -> None:
    """Server spans for the app, client spans for everything it calls out to.

    Each instrumentor is installed independently: httpx missing is not a reason
    to lose FastAPI spans. Called after configure_tracing() so the provider the
    instrumentors pick up is the one configured above.
    """
    global _app_instrumented
    if _app_instrumented or not _provider_configured:
        return

    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        # excluded_urls: same exclusion as the access log, a comma-separated
        # list of regexes matched against the path.
        #
        # exclude_spans: the ASGI instrumentation otherwise emits an extra
        # INTERNAL "... http send" span per response event (two per request,
        # both named identically), which in Tempo reads as duplicated spans and
        # buries the one span that matters. Measured: without this a single
        # /query produced 4 spans, 2 of them send events.
        FastAPIInstrumentor.instrument_app(
            app,
            excluded_urls="health",
            exclude_spans=["send", "receive"],
        )
    except ImportError:
        logger.warning("opentelemetry-instrumentation-fastapi not installed — no server spans")
    except Exception:
        logger.exception("failed to instrument FastAPI")

    # Qdrant and Ollama are both HTTP. qdrant-client and the ollama client used
    # by langchain-ollama go through httpx; get_weather (Open-Meteo) uses
    # requests. Between them these two cover every outbound call the service
    # makes, so the slow part of a trace is visible without touching src/.
    for name, path, cls in (
        ("httpx", "opentelemetry.instrumentation.httpx", "HTTPXClientInstrumentor"),
        ("requests", "opentelemetry.instrumentation.requests", "RequestsInstrumentor"),
    ):
        try:
            module = __import__(path, fromlist=[cls])
            getattr(module, cls)().instrument()
        except ImportError:
            logger.info("no %s instrumentation installed — its calls will not appear as spans", name)
        except Exception:
            logger.exception("failed to instrument %s", name)

    _app_instrumented = True


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

# Seconds. The SDK's default histogram boundaries stop at 10s, so on this
# service — one 4b model on a 6GB card, two concurrent slots — almost every
# generation would land in the +Inf bucket and histogram_quantile() would
# return a confident, wrong number. These reach 5 minutes.
_SLOW_BUCKETS_S = [0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300]
_SLOW_BUCKETS_MS = [b * 1000 for b in _SLOW_BUCKETS_S]
# Chunk counts. The 0 bucket is the point: zero sources is a silent failure —
# fast, HTTP 200, and useless.
_COUNT_BUCKETS = [0, 1, 2, 3, 5, 8, 10, 15, 20, 30]


def configure_metrics() -> bool:
    """Install the global meter provider and create the instruments.

    Push, not pull. Prometheus lives *inside* the otel-lgtm container, so
    exposing /metrics here would mean handing its collector a scrape config —
    a mounted file, and a config directory to maintain. Exporting over OTLP
    instead lets the bundled collector do the Prometheus handoff, which is the
    push/pull asymmetry solved by not participating in it.
    """
    global _metrics_configured
    if _metrics_configured:
        return True
    if not otlp_endpoint():
        return False

    try:
        from opentelemetry import metrics
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
        from opentelemetry.sdk.resources import Resource
    except ImportError:
        logger.warning("metrics requested but the opentelemetry metrics SDK is missing")
        return False

    try:
        if _protocol().startswith("http"):
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        else:
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    except ImportError:
        logger.warning("metrics requested but no OTLP metric exporter is installed")
        return False

    existing = metrics.get_meter_provider()
    if isinstance(existing, MeterProvider):
        logger.warning("a meter provider is already installed — leaving it alone")
        _metrics_configured = True
        return True

    def bucketed(name, boundaries):
        return View(instrument_name=name,
                    aggregation=ExplicitBucketHistogramAggregation(boundaries))

    # Both spellings are listed on purpose: which one the instrumentation emits
    # depends on the semantic-convention opt-in, and a View naming an
    # instrument that never appears is simply ignored. Cheaper than guessing.
    views = [
        bucketed("http.server.request.duration", _SLOW_BUCKETS_S),   # stable semconv, seconds
        bucketed("http.server.duration", _SLOW_BUCKETS_MS),          # legacy, milliseconds
        bucketed("http.client.request.duration", _SLOW_BUCKETS_S),
        bucketed("http.client.duration", _SLOW_BUCKETS_MS),
        bucketed("rag.generation.queue_wait", _SLOW_BUCKETS_S),
        bucketed("rag.generation.time_to_first_token", _SLOW_BUCKETS_S),
        bucketed("rag.retrieval.sources", _COUNT_BUCKETS),
    ]

    ep = _signal_endpoint("METRICS", "/v1/metrics")
    reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=ep) if ep else OTLPMetricExporter(),
        # The default is 60s, which makes a local feedback loop painful: change
        # something, wait a minute to see it. Grafana's blog uses the same knob
        # (OTEL_METRIC_EXPORT_INTERVAL) for exactly this reason.
        export_interval_millis=int(os.environ.get("OTEL_METRIC_EXPORT_INTERVAL", "15000")),
    )
    metrics.set_meter_provider(
        MeterProvider(resource=_resource(Resource), metric_readers=[reader], views=views)
    )

    from api import metrics as rag_metrics

    rag_metrics.init()
    _metrics_configured = True
    logger.info("metrics on", extra={"export_interval_ms":
                                     reader._export_interval_millis if hasattr(
                                         reader, "_export_interval_millis") else None})
    return True


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------


def configure_log_export() -> bool:
    """Ship the same log records over OTLP, in addition to stdout.

    In addition, never instead: stdout stays so `docker logs` and a local run
    still show something, and losing the collector must not mean losing the
    logs entirely.

    Why OTLP and not the Docker `loki` logging driver: an OTLP log record
    carries trace context natively, so Loki stores trace_id as structured
    metadata and otel-lgtm's preconfigured Grafana query — `{...} | trace_id =
    "$${__trace.traceId}"` — matches without a parser. The Docker driver would
    instead hand Loki a text line to regex, and would need a plugin installed
    on the host.

    Handler placement is not "the root logger". configure_logging() sets
    `handlers = [...]` and `propagate = False` on uvicorn, uvicorn.error and
    rag, so a handler on root alone would silently miss every pipeline line.
    """
    global _logs_configured
    if _logs_configured:
        return True
    if not otlp_endpoint():
        return False

    try:
        from opentelemetry._logs import set_logger_provider
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.sdk.resources import Resource
    except ImportError:
        logger.warning("log export requested but the opentelemetry logs SDK is missing")
        return False

    try:
        if _protocol().startswith("http"):
            from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
        else:
            from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
    except ImportError:
        logger.warning("log export requested but no OTLP log exporter is installed")
        return False

    provider = LoggerProvider(resource=_resource(Resource))
    ep = _signal_endpoint("LOGS", "/v1/logs")
    provider.add_log_record_processor(
        BatchLogRecordProcessor(OTLPLogExporter(endpoint=ep) if ep else OTLPLogExporter())
    )
    set_logger_provider(provider)

    handler = LoggingHandler(level=logging.NOTSET, logger_provider=provider)

    # Without this, an export failure logs a warning, which becomes a log
    # record, which is exported, which fails... The collector being down must
    # not turn into a spin.
    handler.addFilter(lambda record: not record.name.startswith("opentelemetry"))

    for name in ("", "uvicorn", "uvicorn.error", "rag"):
        target = logging.getLogger(name)
        if not any(isinstance(h, LoggingHandler) for h in target.handlers):
            target.addHandler(handler)

    _logs_configured = True
    logger.info("log export on — records go to stdout and OTLP")
    return True
