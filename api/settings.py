"""API-only settings — everything the service needs that the pipeline does not.

Pipeline configuration lives in `src/config.py` and is shared with the scripts.
This file holds what is true of the *server* and nothing else, so importing it
tells you exactly which knobs deployment has.

Env-only, with defaults that are safe rather than convenient (12-factor). These
become Docker `-e` flags and later Kubernetes env entries unchanged.
"""

import os


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("", "0", "false", "no", "off")


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# Identity — becomes the OTel resource `service.name` when tracing is wired up.
SERVICE_NAME = os.environ.get("SERVICE_NAME", "enterprise-rag-api")
SERVICE_VERSION = os.environ.get("SERVICE_VERSION", "0.1.0")

# Logging. `json` is the default because Loki is the destination: structured
# logs need no parsing pipeline, and trace_id can be a field rather than
# something a regex has to find. `text` is for reading locally.
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.environ.get("LOG_FORMAT", "json").lower()

# /agent, and with it Postgres, langgraph and langchain_community. Off means
# those are never imported — a retrieval-only image can drop them entirely.
ENABLE_AGENT = _flag("API_ENABLE_AGENT", True)

# Warm the clients in a background thread at startup. The server accepts
# connections immediately either way; this only decides whether the first
# request pays for the Ollama model load and the agent build.
WARMUP = _flag("API_WARMUP", True)

# The real bottleneck is one 4b model on a 6GB card, not FastAPI. Without a cap,
# ten concurrent requests do not go ten times faster — they thrash. Two keeps
# latency predictable; raise it only if the generator is hosted elsewhere.
MAX_CONCURRENT_GENERATIONS = _int("API_MAX_CONCURRENT_GENERATIONS", 2)

# Seconds a request will wait for a generation slot before giving up with 503.
# Better a fast honest refusal than a queue nobody can see.
GENERATION_QUEUE_TIMEOUT = _int("API_GENERATION_QUEUE_TIMEOUT", 120)

# Comma-separated, or "*" for any. Default is none: a service that talks to
# nothing is the safe starting point.
CORS_ORIGINS = [o.strip() for o in os.environ.get("API_CORS_ORIGINS", "").split(",") if o.strip()]
