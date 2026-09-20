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

import os
import threading
import time

from api import metrics, settings
from api.logging_config import request_id_var  # noqa: F401  (re-exported for convenience)

import logging

logger = logging.getLogger("api.state")

# Minimum gap between /ready-triggered retries of a failed component. Long
# enough that a healthcheck polling every 5s does not rebuild the agent on
# every poll; short enough that a dependency coming up is noticed in seconds.
RECHECK_INTERVAL = float(os.environ.get("API_RECHECK_INTERVAL", "10"))


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
        self._recheck_lock = threading.Lock()
        self._last_recheck = 0.0

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

    def acquire_generation_slot(self, endpoint: str = "unknown",
                                timeout: float | None = None) -> bool:
        """Wait for a slot, and record how long that took.

        The wait is measured here rather than in the handler because this is
        the only place that knows the difference between "waited and got one"
        and "waited and gave up" — and the second is a user-visible 503 that an
        HTTP error-rate panel cannot tell apart from a backend failure.
        """
        started = time.perf_counter()
        acquired = self._generation.acquire(
            timeout=settings.GENERATION_QUEUE_TIMEOUT if timeout is None else timeout
        )
        metrics.queue_wait(endpoint, time.perf_counter() - started, acquired)
        if acquired:
            metrics.slot_held(endpoint, 1)
        return acquired

    def release_generation_slot(self, endpoint: str = "unknown") -> None:
        try:
            self._generation.release()
            metrics.slot_held(endpoint, -1)
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

        self._refresh_missing_settings()
        self._check_qdrant()
        self._check_ollama()
        if settings.ENABLE_AGENT:
            self._check_agent()

        self.warming = False
        self.warmed_at = time.time()
        logger.info(
            "warmup finished", extra={"seconds": round(self.warmed_at - started, 2),
                                      "ready": self.is_ready()}
        )

    def warm_in_background(self) -> None:
        threading.Thread(target=self.warm, name="warmup", daemon=True).start()

    # --- individual checks -------------------------------------------------
    # Separate so `recheck_failed` can re-run exactly the ones that are down,
    # rather than redoing a warmup that would re-pay for everything working.

    def _refresh_missing_settings(self) -> None:
        from src.config import CONFIG

        self.missing_settings = CONFIG.missing_settings(need_postgres=settings.ENABLE_AGENT)
        if self.missing_settings:
            logger.error("missing required settings: %s", ", ".join(self.missing_settings))

    def _check_qdrant(self) -> None:
        """Construct the client and actually ask it something.

        Constructing proves nothing: QdrantClient does no network I/O until the
        first call, so a wrong URL or key looks perfectly healthy until a user
        hits it. Reading the collection is the cheapest real check.
        """
        from src.config import CONFIG

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

    def _check_ollama(self) -> None:
        """One tiny generation, for the same reason: ChatOllama constructs
        without contacting anything, and the model load happens on first use.
        Paying that here is the entire point of warming."""
        from src.config import CONFIG

        try:
            self.get_llm().invoke("ok")
            self.ollama.ok()
            logger.info("ollama ready", extra={"model": CONFIG.ollama_model})
        except Exception as exc:  # noqa: BLE001
            self.ollama.fail(exc)

    def _check_agent(self) -> None:
        try:
            from src.history import ensure_schema

            ensure_schema()
            self.get_agent()
            self.agent.ok()
            logger.info("agent ready")
        except Exception as exc:  # noqa: BLE001
            self.agent.fail(exc)

    # --- self-healing ------------------------------------------------------

    def recheck_failed(self) -> bool:
        """Retry the components that are currently down. Returns True if it ran.

        Warmup happens once, at startup, which was fine when every dependency
        was already up before uvicorn started. Under Compose it is not: the API
        container starts alongside Postgres and will usually win the race, so
        the agent fails once and — without this — stays dead until someone
        restarts the container, even though the database came up two seconds
        later.

        Only failed components are retried, so a healthy instance pays nothing:
        no repeated Qdrant round trip, and no repeated Ollama generation, which
        would otherwise make every /ready poll load the model.

        Rate limited (`RECHECK_INTERVAL`) because /ready gets polled — by a
        Docker healthcheck every few seconds, later by Kubernetes. Without the
        cooldown a down Postgres would mean rebuilding the agent on every poll.
        The lock stops concurrent polls piling up behind the same slow retry.
        """
        if self.warming:
            return False

        failed = [c for c in self._all_components() if not c.ready]
        if not failed and not self.missing_settings:
            return False

        now = time.time()
        if now - self._last_recheck < RECHECK_INTERVAL:
            return False

        if not self._recheck_lock.acquire(blocking=False):
            return False  # another poll is already retrying

        try:
            self._last_recheck = time.time()
            logger.info("rechecking failed components",
                        extra={"components": [c.name for c in failed]})

            # Settings can change between restarts of a *dependency*, not of
            # this process — but re-reading costs nothing and keeps the
            # reported list honest if the env was fixed and the container
            # restarted only partially.
            self._refresh_missing_settings()

            for component in failed:
                if component is self.qdrant:
                    self._check_qdrant()
                elif component is self.ollama:
                    self._check_ollama()
                elif component is self.agent and settings.ENABLE_AGENT:
                    self._check_agent()
            return True
        finally:
            self._recheck_lock.release()

    def _all_components(self) -> list[ComponentStatus]:
        components = [self.qdrant, self.ollama]
        if settings.ENABLE_AGENT:
            components.append(self.agent)
        return components


STATE = ServiceState()
