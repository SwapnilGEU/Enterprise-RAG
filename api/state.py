"""Process-wide service state: warmup, readiness, and safe access to singletons.

Three jobs, all of which exist because the pipeline was written for scripts and
is now being run by a server.

1. **Thread safety.** `vector_store._client`, `generation._llm` and
   `agent._agent` are plain check-then-set module globals. One script caller
   makes that correct; a threadpool does not. Two concurrent first requests can
   both build an agent — two Postgres toolkits, two graph compiles, one of them
   silently orphaned. Every lazy build goes through a lock here.

2. **Readiness that is honest per component.** Qdrant being reachable and
   Postgres being reachable are different questions, so `/ready` answers them
   separately and `/query` keeps serving while `/agent` is down. A single
   boolean would take the whole service offline for a dependency half of it
   does not use.

3. **Bounding generation.** The bottleneck is one 4b model on a 6GB card. A
   semaphore makes that explicit: requests beyond the cap wait, and past the
   timeout get a 503 they can act on rather than a queue nobody can see.
"""

import threading
import time

from api import settings
from api.logging_config import request_id_var  # noqa: F401  (re-exported for convenience)

import logging

logger = logging.getLogger("api.state")


class ComponentStatus:
    """Whether one dependency is usable, and why not if it isn't."""

    def __init__(self, name: str, required: bool):
        self.name = name
        self.required = required
        self.ready = False
        self.error: str | None = None
        self.checked_at: float | None = None

    def ok(self) -> None:
        self.ready, self.error, self.checked_at = True, None, time.time()

    def fail(self, exc: BaseException) -> None:
        self.ready = False
        self.error = f"{type(exc).__name__}: {exc}"[:300]
        self.checked_at = time.time()
        logger.warning("component %s unavailable: %s", self.name, self.error)

    def as_dict(self) -> dict:
        return {
            "ready": self.ready,
            "required": self.required,
            "error": self.error,
            "checked_at": self.checked_at,
        }


class ServiceState:
    def __init__(self):
        self.started_at = time.time()
        self.warming = False
        self.warmed_at: float | None = None

        # Qdrant is required: without it there is no retrieval and so no
        # service. The agent is optional by construction — see the module
        # docstring.
        self.qdrant = ComponentStatus("qdrant", required=True)
        self.ollama = ComponentStatus("ollama", required=True)
        self.agent = ComponentStatus("agent", required=False)

        self._llm_lock = threading.Lock()
        self._agent_lock = threading.Lock()
        self._generation = threading.BoundedSemaphore(settings.MAX_CONCURRENT_GENERATIONS)

        self.missing_settings: list[str] = []

    # --- readiness ---------------------------------------------------------

    def components(self) -> dict:
        data = {
            "qdrant": self.qdrant.as_dict(),
            "ollama": self.ollama.as_dict(),
        }
        if settings.ENABLE_AGENT:
            data["agent"] = self.agent.as_dict()
        return data

    def is_ready(self) -> bool:
        """Ready when every *required* component is up and config is complete.

        A degraded agent deliberately does not make the service unready: /query
        is still fully functional, and reporting otherwise would take a healthy
        endpoint out of a load balancer for a dependency it never touches.
        """
        if self.missing_settings:
            return False
        return self.qdrant.ready and self.ollama.ready

    # --- guarded access to the pipeline singletons -------------------------

    def get_llm(self):
        from src.generation import get_llm

        with self._llm_lock:
            return get_llm()

    def get_agent(self):
        """Compile the LangGraph agent, once, under a lock.

        Raises whatever the build raises — the caller turns that into a 503 and
        records it on the agent component, rather than retrying here. A failing
        Postgres will keep failing for the same reason on the next request, and
        a retry loop in here would just hide it.
        """
        from src.agent import build_agent

        with self._agent_lock:
            return build_agent()

    # --- bounding the generator -------------------------------------------

    def acquire_generation_slot(self, timeout: float | None = None) -> bool:
        return self._generation.acquire(
            timeout=settings.GENERATION_QUEUE_TIMEOUT if timeout is None else timeout
        )

    def release_generation_slot(self) -> None:
        try:
            self._generation.release()
        except ValueError:
            # BoundedSemaphore raises on over-release. That means a bug in the
            # acquire/release pairing, but it must not take down a request that
            # has already produced an answer.
            logger.error("generation semaphore over-released — check acquire/release pairing")

    # --- warmup ------------------------------------------------------------

    def warm(self) -> None:
        """Build the expensive things once, off the request path.

        Runs in a background thread so uvicorn accepts connections immediately.
        Every step is independent and failure is recorded rather than raised:
        the point of warming is to find out what is broken and say so on
        /ready, not to prevent the process from starting.
        """
        self.warming = True
        started = time.time()
        logger.info("warmup starting")

        from src.config import CONFIG

        self.missing_settings = CONFIG.missing_settings(need_postgres=settings.ENABLE_AGENT)
        if self.missing_settings:
            logger.error("missing required settings: %s", ", ".join(self.missing_settings))

        # Qdrant — construct the client and actually ask it something. Merely
        # constructing proves nothing: QdrantClient does no network I/O until
        # the first call, so a wrong URL or key looks healthy until a user hits
        # it. Reading the collection is the cheapest real check.
        try:
            from src.vector_store import get_client

            info = get_client().get_collection(CONFIG.collection_name)
            self.qdrant.ok()
            logger.info(
                "qdrant ready", extra={"collection": CONFIG.collection_name,
                                       "points": getattr(info, "points_count", None)}
            )
        except Exception as exc:  # noqa: BLE001
            self.qdrant.fail(exc)

        # Ollama — construct the client and make one tiny generation, for the
        # same reason: ChatOllama constructs without contacting anything, and
        # the model load happens on first use. Paying that here is the entire
        # point of warming.
        try:
            llm = self.get_llm()
            llm.invoke("ok")
            self.ollama.ok()
            logger.info("ollama ready", extra={"model": CONFIG.ollama_model})
        except Exception as exc:  # noqa: BLE001
            self.ollama.fail(exc)

        if settings.ENABLE_AGENT:
            try:
                from src.history import ensure_schema

                ensure_schema()
                self.get_agent()
                self.agent.ok()
                logger.info("agent ready")
            except Exception as exc:  # noqa: BLE001
                self.agent.fail(exc)

        self.warming = False
        self.warmed_at = time.time()
        logger.info(
            "warmup finished", extra={"seconds": round(self.warmed_at - started, 2),
                                      "ready": self.is_ready()}
        )

    def warm_in_background(self) -> None:
        threading.Thread(target=self.warm, name="warmup", daemon=True).start()


STATE = ServiceState()
