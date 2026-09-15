"""Hybrid retrieval and reranking — notebook Section 13.

Tracing note
------------
`retrieve()` is wrapped in an MLflow RETRIEVER span. That span is not
decoration: `mlflow.genai.scorers.deepeval` builds the DeepEval
`LLMTestCase.retrieval_context` *exclusively* from top-level RETRIEVER spans on
the trace (see `mlflow/genai/utils/trace_utils.py::extract_retrieval_context_from_trace`).
There is no way to pass retrieval context in through `inputs` or `expectations`.
Without this span, Faithfulness / ContextualPrecision / ContextualRecall do not
error — they silently score against an empty context.

The span's return value must be a **list of dicts** with the chunk text under
`page_content`, `content` or `text` (plus optional `metadata`, of which only
`metadata.doc_uri` is read). Anything else is dropped with a debug-level log.
That is why `retrieve()` now returns dicts rather than raw Qdrant ScoredPoints.

mlflow is an optional dependency here: if it isn't installed, `@_trace_retriever`
degrades to a no-op decorator and the pipeline runs exactly as before.
"""

import os
from functools import partial

from qdrant_client import models

from src.config import CONFIG, Config
from src.retry import call_with_retry, validate_retrieval
from src.vector_store import get_client

# --------------------------------------------------------------------------
# Optional MLflow tracing
# --------------------------------------------------------------------------

def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("", "0", "false", "no")


# MLflow builds retrieval_context with ONE entry per top-level retriever span --
# `[str(chunks) for chunks in span_id_to_context.values()]` -- not one per chunk.
# With a single span, all k chunks collapse into one context node, and
# ContextualPrecision (whose whole job is judging whether relevant nodes are
# ranked above irrelevant ones) degenerates to a yes/no on the blob.
#
# Set RETRIEVER_SPAN_PER_CHUNK=1 to emit one retriever span per chunk, in rank
# order, so precision/recall see k separate nodes. Off by default because a
# single "retrieve" span is the conventional shape and the one the later OTel
# work expects. Turn it on when comparing rerankers or top_k values.
SPAN_PER_CHUNK = _env_flag("RETRIEVER_SPAN_PER_CHUNK", default=False)

try:
    import mlflow
    from mlflow.entities import SpanType

    MLFLOW_TRACING = True
    # When chunks get their own retriever spans, the parent must NOT be a
    # retriever span -- a retriever span nested inside another one is ignored.
    _parent_span_type = SpanType.CHAIN if SPAN_PER_CHUNK else SpanType.RETRIEVER
    _trace_retriever = mlflow.trace(span_type=_parent_span_type, name="retrieve")
except Exception:  # mlflow not installed, or too old for SpanType
    def _trace_retriever(fn):
        return fn

    MLFLOW_TRACING = False


def _emit_chunk_spans(query: str, docs: list[dict]) -> None:
    """One top-level RETRIEVER span per chunk, in rank order."""
    for rank, doc in enumerate(docs, start=1):
        with mlflow.start_span(
            name=f"retrieved_chunk_{rank}", span_type=SpanType.RETRIEVER
        ) as span:
            span.set_inputs({"query": query, "rank": rank})
            span.set_outputs([doc])


class RetrievedDoc:
    """Duck-types a LangChain Document (`.page_content` / `.metadata`), which is
    what build_context() was written to expect.

    Accepts either a Qdrant ScoredPoint or one of the plain dicts that
    `retrieve()` now returns, so nothing downstream cares which it gets.
    """

    def __init__(self, source):
        if isinstance(source, dict):
            self.metadata = source.get("metadata") or {}
            self.page_content = (
                source.get("page_content")
                or source.get("content")
                or source.get("text")
                or ""
            )
            self.score = source.get("score")
        else:  # Qdrant ScoredPoint
            self.metadata = source.payload or {}
            self.page_content = self.metadata.get("text", "")
            self.score = getattr(source, "score", None)


def point_to_document(point) -> dict:
    """Qdrant ScoredPoint -> the dict shape the MLflow retriever span expects.

    `page_content` is the key MLflow looks for first. Everything the payload
    carries (section_heading, page_label, document, chunk_id, ...) stays under
    `metadata`, so format_source() keeps working unchanged, and `doc_uri` is
    set because it is the one metadata key MLflow reads back.
    """
    payload = dict(point.payload or {})
    text = payload.get("text", "")
    metadata = {k: v for k, v in payload.items() if k != "text"}
    metadata["doc_uri"] = payload.get("document") or ""
    score = getattr(point, "score", None)
    if score is not None:
        metadata["score"] = score
    return {"page_content": text, "metadata": metadata}


def retrieve_points(query: str, top_k: int | None = None, config: Config = CONFIG):
    """Hybrid retrieval: dense + sparse candidates, reranked by ColBERT.

    Two independent retrievers (dense semantic, sparse BM25) each contribute up
    to `hybrid_prefetch_limit` candidates. The late-interaction model then
    reranks that combined pool and only the top `top_k` survive — this is why
    the ColBERT model is "used for reranking, not ANN retrieval": it never scans
    the whole collection, only the prefetched candidates.

    `with_payload=True` is what carries section_heading / page_label back. Drop
    it and you get IDs and scores with nothing to cite.

    Returns raw Qdrant ScoredPoints. Most callers want `retrieve()` instead.
    """
    client = get_client(config)
    top_k = config.final_top_k if top_k is None else top_k

    results = client.query_points(
        config.collection_name,
        prefetch=[
            models.Prefetch(
                query=models.Document(text=query, model=config.dense_model),
                using="dense",
                limit=config.hybrid_prefetch_limit,
            ),
            models.Prefetch(
                query=models.Document(text=query, model=config.sparse_model),
                using="sparse",
                limit=config.hybrid_prefetch_limit,
            ),
        ],
        query=models.Document(text=query, model=config.late_interaction_model),
        using="multi",
        limit=top_k,
        with_payload=True,
    )
    return results.points


@_trace_retriever
def retrieve(query: str, top_k: int | None = None, config: Config = CONFIG) -> list[dict]:
    """Hybrid retrieval, returned in the shape MLflow's RETRIEVER span requires.

    This is the function the whole pipeline calls. The decorator is what puts
    the retrieved chunks on the trace where the DeepEval scorers can find them.

    The retry lives *inside* the span on purpose: retrying from the outside
    would emit one retriever span per attempt, and a failed attempt's empty
    output would land on the trace as an empty retrieval context.

    Raises once retries are exhausted — generate_answer() turns that into a
    degraded answer.
    """
    points = call_with_retry(
        partial(retrieve_points, config=config),
        query,
        top_k=top_k,
        validate=validate_retrieval,
        config=config,
        call_name="qdrant_retrieve",
    )
    docs = [point_to_document(p) for p in points]

    if MLFLOW_TRACING and SPAN_PER_CHUNK:
        _emit_chunk_spans(query, docs)

    return docs
