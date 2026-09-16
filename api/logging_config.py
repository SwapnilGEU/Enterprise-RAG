"""JSON logging with trace and request correlation — the seam for Loki/Tempo.

Why this exists before OpenTelemetry does
-----------------------------------------
Loki can parse unstructured logs, but only with regex pipeline stages that are
slow, brittle and written under pressure later. More importantly, the thing
that makes Grafana genuinely useful — clicking a log line and landing on that
request's trace in Tempo — needs `trace_id` to be a **field**, not something a
pattern has to dig out of a sentence.

Both are cheap now and expensive to retrofit across every log call in the
codebase, so the format is settled here first and the exporters come later.

`trace_id` works today, with nothing installed
----------------------------------------------
MLflow's tracing is built on OpenTelemetry (`mlflow.tracing.provider` imports
the OTel SDK directly). So `src/retrieval.py`'s `@mlflow.trace` spans already
put a real span in the OTel context, and this formatter reads it. Before any
exporter exists the ids simply go unrecorded; once `OTEL_EXPORTER_OTLP_ENDPOINT`
is set they become the ids Tempo knows. No log call changes.

If opentelemetry is not installed at all, `_trace_ids` returns nothing and the
fields are omitted. The logger never becomes a reason the service fails.
"""

import json
import logging
import sys
from contextvars import ContextVar

from api import settings

# Set by the request-id middleware, read here. A ContextVar rather than a
# thread-local: FastAPI's threadpool copies the context into the worker thread,
# so a sync endpoint's logs carry the right id without being passed one.
request_id_var: ContextVar[str] = ContextVar("request_id", default="")

# Attributes LogRecord always has; anything else was passed as `extra=` and is
# worth putting in the JSON.
_STANDARD = frozenset(
    """args asctime created exc_info exc_text filename funcName levelname levelno
    lineno module msecs message msg name pathname process processName relativeCreated
    stack_info thread threadName taskName""".split()
)


def _trace_ids() -> dict:
    """Current OTel trace and span ids, hex-formatted as Tempo expects."""
    try:
        from opentelemetry import trace
    except ImportError:
        return {}

    span = trace.get_current_span()
    context = span.get_span_context()
    if not context or not context.is_valid:
        return {}
    return {
        "trace_id": format(context.trace_id, "032x"),
        "span_id": format(context.span_id, "016x"),
    }


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": settings.SERVICE_NAME,
        }

        if rid := request_id_var.get():
            payload["request_id"] = rid
        payload.update(_trace_ids())

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # Whatever the call site passed as extra={...}.
        for key, value in record.__dict__.items():
            if key not in _STANDARD and not key.startswith("_"):
                payload[key] = value

        return json.dumps(payload, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Human-readable, for local work. Same fields, less ceremony."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        bits = []
        if rid := request_id_var.get():
            bits.append(f"req={rid[:8]}")
        if ids := _trace_ids():
            bits.append(f"trace={ids['trace_id'][:16]}")
        return f"{base}  [{' '.join(bits)}]" if bits else base


def configure_logging() -> None:
    """Install the formatter on the root logger and on uvicorn's.

    uvicorn installs its own handlers, so without clearing them every line is
    emitted twice — once ours, once theirs — which in Loki looks like duplicate
    events rather than a formatting bug.
    """
    formatter: logging.Formatter
    if settings.LOG_FORMAT == "json":
        formatter = JsonFormatter()
    else:
        formatter = TextFormatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(settings.LOG_LEVEL)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        target = logging.getLogger(name)
        target.handlers = [handler]
        target.propagate = False

    # This one is noisy at INFO and says nothing the access log does not.
    logging.getLogger("httpx").setLevel(logging.WARNING)
