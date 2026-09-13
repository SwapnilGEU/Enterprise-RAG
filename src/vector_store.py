"""Qdrant: client, collection, deduplication, upload — notebook Sections 10-12.

Everything that talks to the vector database. The client is created lazily
(get_client) rather than at import time, so importing this module has no side
effects and no network call.
"""

import uuid
from collections import defaultdict

from qdrant_client import QdrantClient, models

from src.config import CONFIG, Config, logger
from src.payload import build_payload, content_hash


# --------------------------------------------------------------------------
# Client and collection (Section 10)
# --------------------------------------------------------------------------

_client: QdrantClient | None = None


def get_client(config: Config = CONFIG) -> QdrantClient:
    """Connect to Qdrant Cloud. One client per process, created on first use.

    A plain module-level singleton rather than @lru_cache: Config is a mutable
    dataclass and therefore unhashable, so lru_cache would raise
    `TypeError: unhashable type: 'Config'` the moment it tried to key on it.

    `cloud_inference=True` means Qdrant computes the embeddings server-side, so
    nothing here needs fastembed or a GPU. Three models, three roles:
      dense  — semantic similarity, catches paraphrases
      sparse — BM25 keyword matching, catches rare terms and model names
      multi  — ColBERT late interaction, reranks the prefetched pool only
    """
    global _client
    if _client is not None:
        return _client

    if not config.qdrant_api_key:
        raise RuntimeError(
            "QDRANT_API_KEY is not set. Copy .env.example to .env and fill it in, "
            "or export the variable in your shell."
        )

    _client = QdrantClient(
        url=config.qdrant_url,
        api_key=config.qdrant_api_key,
        cloud_inference=True,
    )
    logger.info(f"Connected to Qdrant. Target collection: {config.collection_name}")
    return _client


def ensure_collection(force_recreate: bool = False, config: Config = CONFIG) -> None:
    """Create the collection if missing. `force_recreate=True` wipes it first —
    only needed when you deliberately want to rebuild from scratch, since point
    IDs are deterministic and re-uploading overwrites in place."""
    client = get_client(config)
    name = config.collection_name

    if force_recreate and client.collection_exists(collection_name=name):
        client.delete_collection(collection_name=name)
        logger.warning(f"Collection '{name}' deleted for recreation.")

    if client.collection_exists(collection_name=name):
        logger.info(f"Collection '{name}' already exists — skipping creation.")
        return

    client.create_collection(
        name,
        vectors_config={
            "dense": models.VectorParams(
                size=config.dense_vector_size,
                distance=models.Distance.COSINE,
            ),
            "multi": models.VectorParams(
                size=config.late_interaction_vector_size,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                ),
                hnsw_config=models.HnswConfigDiff(m=0),   # disable HNSW — rerank only
            ),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )
    logger.info(f"Collection '{name}' created.")


def ensure_payload_indexes(config: Config = CONFIG) -> None:
    """Create keyword indexes on the fields you might filter by.

    Qdrant local mode (`:memory:`) filters in Python and needs no index, but the
    server rejects a filter on an unindexed field with
    `400 Index required but not found for "document" of type [keyword]`.
    Not needed by anything here today (dedup deliberately scrolls unfiltered),
    but required the moment you add filtered retrieval such as
    "search only within this document". Idempotent.
    """
    client = get_client(config)
    for field_name in ("document", "source_type"):
        try:
            client.create_payload_index(
                collection_name=config.collection_name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            logger.info(f"Payload index created on '{field_name}'.")
        except Exception as exc:
            logger.debug(f"Payload index on '{field_name}': {exc}")   # usually "already exists"


# --------------------------------------------------------------------------
# Deduplication (Section 11)
# --------------------------------------------------------------------------

def point_id(document: str, chunk_id: str, config: Config = CONFIG) -> str:
    """Deterministic UUID for a Qdrant point — the same (document, chunk_id)
    always produces the same ID, so re-upserting an unchanged chunk overwrites
    in place instead of creating a duplicate point."""
    return str(uuid.uuid5(uuid.UUID(config.dedup_namespace), f"{document}::{chunk_id}"))


def fetch_all_existing_state(config: Config = CONFIG) -> dict[str, dict[str, dict]]:
    """document -> {chunk_id: {"hash": ..., "has_section_metadata": bool}}

    One unfiltered pass over the collection, deliberately: filtering by document
    would need a payload index, and one scroll is fewer round trips than one
    filtered scroll per document anyway.
    """
    client = get_client(config)
    state: dict[str, dict[str, dict]] = defaultdict(dict)
    next_offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=config.collection_name,
            with_payload=["document", "chunk_id", "content_hash", "section_heading"],
            limit=256,
            offset=next_offset,
        )

        for p in points:
            payload = p.payload or {}
            document = payload.get("document")
            chunk_id = payload.get("chunk_id")
            stored_hash = payload.get("content_hash")

            if document and chunk_id is not None and stored_hash is not None:
                state[document][chunk_id] = {
                    "hash": stored_hash,
                    # key present at all (even as "") means build_payload wrote it
                    "has_section_metadata": "section_heading" in payload,
                }

        if next_offset is None:
            break

    return state


def classify_chunks(chunks: list[dict], config: Config = CONFIG) -> dict[str, list[dict]]:
    """Split chunks into new / changed / stale_metadata / unchanged.

    `stale_metadata` catches points written before build_payload existed: the
    text and hash still match, so a pure hash check would call them unchanged
    forever and they would keep citing "[Unknown section, p.?]".
    """
    existing_state = fetch_all_existing_state(config)
    buckets: dict[str, list[dict]] = {"new": [], "changed": [], "stale_metadata": [], "unchanged": []}

    for chunk in chunks:
        stored = existing_state.get(chunk["document"], {}).get(chunk["chunk_id"])
        current_hash = content_hash(chunk["text"])

        if stored is None:
            buckets["new"].append(chunk)
        elif stored["hash"] != current_hash:
            buckets["changed"].append(chunk)
        elif not stored["has_section_metadata"]:
            buckets["stale_metadata"].append(chunk)
        else:
            buckets["unchanged"].append(chunk)

    return buckets


# --------------------------------------------------------------------------
# Upload (Section 12.3)
# --------------------------------------------------------------------------

def upload_chunks(chunks: list[dict], batch_size: int = 25, config: Config = CONFIG) -> int:
    """Embed (server-side) and upsert. Returns how many points were written."""
    if not chunks:
        logger.info("Nothing to upload.")
        return 0

    client = get_client(config)

    points = [
        models.PointStruct(
            id=point_id(chunk["document"], chunk["chunk_id"], config),
            vector={
                "dense": models.Document(text=chunk["text"], model=config.dense_model),
                "sparse": models.Document(text=chunk["text"], model=config.sparse_model),
                "multi": models.Document(text=chunk["text"], model=config.late_interaction_model),
            },
            payload=build_payload(chunk),
        )
        for chunk in chunks
    ]

    client.upload_points(collection_name=config.collection_name, points=points, batch_size=batch_size)
    logger.info(f"Uploaded {len(points)} point(s) to '{config.collection_name}'.")
    return len(points)
