"""Request and response models.

Deliberately separate from `src/models.py`, which holds the pipeline's internal
dataclasses (`Chunk`, `StructuralUnit`). Those describe how the pipeline thinks;
these describe the contract with a caller, and the two should be free to change
independently — an API that re-exports its internals cannot be refactored
without breaking clients.
"""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, examples=["What is hybrid retrieval?"])
    top_k: int | None = Field(None, ge=1, le=50, description="Defaults to config.final_top_k")
    include_context: bool = Field(
        False, description="Return the retrieved chunk text as well as the citations"
    )


class Source(BaseModel):
    """One citation. Mirrors what `payload.format_source` reads, so the API and
    the console output can never disagree about what a source is."""

    document: str = ""
    section_heading: str = ""
    page_label: str = ""
    chunk_id: str = ""
    score: float | None = None
    citation: str = Field("", description="Pre-formatted, as format_source() renders it")
    text: str | None = Field(None, description="Only when include_context=true")


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source] = []
    degraded: bool = Field(
        False, description="True when the pipeline answered without healthy retrieval"
    )
    failure_stage: str | None = Field(
        None, description="Which stage degraded — set only when degraded is true"
    )
    request_id: str
    latency_ms: float


class AgentRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field("default", max_length=100)


class AgentResponse(BaseModel):
    answer: str
    tools_used: list[str] = Field([], description="In call order, deduplicated")
    request_id: str
    latency_ms: float


class HealthResponse(BaseModel):
    """Liveness only — the process is up. No dependency is consulted, so this
    stays fast and cannot fail because something downstream is having a bad
    day. That distinction is what makes it usable as a container healthcheck."""

    status: str = "ok"
    service: str
    version: str
    uptime_seconds: float


class ReadyResponse(BaseModel):
    """Readiness — can this instance actually serve?

    `components` is per-dependency on purpose: an unready agent with healthy
    retrieval is a real and useful state, and a single boolean would hide it.
    """

    ready: bool
    warming: bool
    components: dict
    missing_settings: list[str] = Field(
        [], description="Required env vars that are unset — see src/config.py::missing_settings"
    )


class ErrorResponse(BaseModel):
    detail: str
    request_id: str
