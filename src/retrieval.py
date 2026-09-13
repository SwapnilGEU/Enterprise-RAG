"""Hybrid retrieval and reranking — notebook Section 13."""

from qdrant_client import models

from src.config import CONFIG, Config
from src.vector_store import get_client


class RetrievedDoc:
    """Wraps a Qdrant ScoredPoint so build_context() can treat it like a
    LangChain Document (which is what it was written to expect)."""

    def __init__(self, point):
        self.metadata = point.payload or {}
        self.page_content = self.metadata.get("text", "")
        self.score = getattr(point, "score", None)


def retrieve(query: str, top_k: int | None = None, config: Config = CONFIG):
    """Hybrid retrieval: dense + sparse candidates, reranked by ColBERT.

    Two independent retrievers (dense semantic, sparse BM25) each contribute up
    to `hybrid_prefetch_limit` candidates. The late-interaction model then
    reranks that combined pool and only the top `top_k` survive — this is why
    the ColBERT model is "used for reranking, not ANN retrieval": it never scans
    the whole collection, only the prefetched candidates.

    `with_payload=True` is what carries section_heading / page_label back. Drop
    it and you get IDs and scores with nothing to cite.
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
