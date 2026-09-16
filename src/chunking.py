"""Chunking — notebook Sections 8 and 9.

Two paths:
  * XLSX  — a row is already one atomic semantic unit, so it becomes one chunk.
  * DOCX/PDF — sections are split where the meaning shifts between sentences
    (embedding similarity drops), falling back to a fixed-size sliding window
    when the section is too short or the embedding model isn't available.

Every chunk carries its section's metadata forward. That is the contract the
citations depend on.
"""

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.config import CONFIG, Config, logger
from src.models import Chunk, StructuralUnit


# --------------------------------------------------------------------------
# Embedding model (local, boundary detection only — not what Qdrant indexes)
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_embedder(model_name: str | None = None):
    """Load the SentenceTransformer used to find semantic breakpoints.

    Lazy and cached: importing this module must not pull a model into memory,
    or `import src.chunking` inside a test or a CLI --help would cost seconds
    and hundreds of MB. Returns None if sentence-transformers isn't installed
    or the model won't load — callers fall back to fixed-size chunking.
    """
    name = model_name or CONFIG.embedding_model_name
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning("sentence-transformers not installed — using fixed-size chunking only.")
        return None

    try:
        model = SentenceTransformer(name)
        logger.info(f"Loaded embedding model: {name}")
        return model
    except Exception:
        logger.exception("Could not load embedding model — using fixed-size chunking only.")
        return None


# --------------------------------------------------------------------------
# Sentence splitting and breakpoints
# --------------------------------------------------------------------------

def split_into_sentences(text: str) -> list[str]:
    raw = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text.strip())
    return [s.strip() for s in raw if s.strip()]


def semantic_breakpoints(embeddings: np.ndarray, percentile: float) -> list[int]:
    """Indices after which a section should be split — where the cosine distance
    between consecutive sentence embeddings exceeds the given percentile of all
    distances in this section (a bigger jump = bigger topic shift)."""
    a, b = embeddings[:-1], embeddings[1:]
    similarities = (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)
    distances = 1 - similarities

    if len(distances) == 0:
        return []

    threshold = np.percentile(distances, percentile)
    return [i for i, d in enumerate(distances) if d > threshold]


# --------------------------------------------------------------------------
# Fixed-size chunking (the fallback path)
# --------------------------------------------------------------------------

_SENTENCE_END_RE = re.compile(r'[.!?]["\')\]]?\s')


def find_chunk_boundary(text: str, start: int, end: int) -> int:
    """Prefer the last sentence end inside (start, end]; fall back to the last
    word boundary if no sentence end is found (e.g. a long run-on section) —
    avoids cutting a chunk off mid-sentence when a clean break is available."""
    window = text[start:end]
    matches = list(_SENTENCE_END_RE.finditer(window))
    if matches:
        return start + matches[-1].end()
    space = text.rfind(" ", start, end)
    return space if space > start else end


def _chunk_dict(unit: StructuralUnit, text: str, chunk_index: int) -> dict:
    """One place that decides what metadata a chunk inherits from its section."""
    return {
        "document": unit.document,
        "source_type": unit.source_type,
        "text": text,
        "metadata": {
            **unit.metadata,
            "section_title": unit.title,
            "section_level": unit.level,
            "chunk_index": chunk_index,
        },
    }


def chunk_structural_unit(unit: StructuralUnit, chunk_size: int = 1500,
                          overlap: int = 200) -> list[dict]:
    """Fixed-size sliding-window chunker — the fallback path."""
    text = unit.text.strip()

    if not text:
        return []

    if len(text) <= chunk_size:
        return [_chunk_dict(unit, text, 0)]

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            end = find_chunk_boundary(text, start, end)

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(_chunk_dict(unit, chunk_text, chunk_index))
            chunk_index += 1

        if end >= len(text):
            break

        next_start = max(end - overlap, start + 1)
        boundary = text.find(" ", next_start, end)
        start = boundary + 1 if boundary != -1 else next_start

    return chunks


# --------------------------------------------------------------------------
# Semantic chunking
# --------------------------------------------------------------------------

def semantic_chunk_unit(unit: StructuralUnit, config: Config = CONFIG) -> list[dict]:
    text = unit.text.strip()
    if not text:
        return []

    sentences = split_into_sentences(text)
    embedder = get_embedder(config.embedding_model_name)

    # Too short to meaningfully group, or no embedding model — fixed-size fallback.
    if embedder is None or len(sentences) < config.semantic_chunk_min_sentences:
        return chunk_structural_unit(
            unit,
            chunk_size=config.semantic_chunk_max_chars,
            overlap=config.fixed_chunk_overlap,
        )

    # Request NumPy output explicitly: semantic_breakpoints uses NumPy
    # operations, and SentenceTransformer may otherwise be typed as returning
    # a torch.Tensor.
    embeddings = np.asarray(embedder.encode(sentences, convert_to_numpy=True))
    breakpoints = semantic_breakpoints(embeddings, config.semantic_chunk_percentile)

    chunk_texts, start = [], 0
    for bp in breakpoints:
        chunk_texts.append(" ".join(sentences[start:bp + 1]))
        start = bp + 1
    chunk_texts.append(" ".join(sentences[start:]))
    chunk_texts = [c.strip() for c in chunk_texts if c.strip()]

    chunk_dicts = []
    for chunk_index, chunk_text in enumerate(chunk_texts):
        if len(chunk_text) > config.semantic_chunk_max_chars:
            # oversized semantic chunk — split further with the fixed-size chunker
            oversized_unit = StructuralUnit(
                unit.document, unit.source_type, unit.title, unit.level, chunk_text, unit.metadata
            )
            chunk_dicts.extend(chunk_structural_unit(
                oversized_unit,
                chunk_size=config.semantic_chunk_max_chars,
                overlap=config.fixed_chunk_overlap,
            ))
        else:
            chunk_dicts.append(_chunk_dict(unit, chunk_text, chunk_index))

    return chunk_dicts


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def chunk_dicts_to_objects(chunk_dicts: list[dict], unit_index: int = 0) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"unit_{unit_index}::chunk_{chunk_index}",
            document=item["document"],
            source_type=item["source_type"],
            text=item["text"],
            metadata=item["metadata"],
        )
        for chunk_index, item in enumerate(chunk_dicts)
    ]


def chunk_structural_units(units: list[StructuralUnit], config: Config = CONFIG) -> list[Chunk]:
    chunks: list[Chunk] = []
    for unit_index, unit in enumerate(units):
        chunks.extend(chunk_dicts_to_objects(semantic_chunk_unit(unit, config), unit_index=unit_index))
    return chunks


def xlsx_records_to_chunks(records: list[dict]) -> list[Chunk]:
    """XLSX rows go straight to chunks — no section splitting, no semantic
    chunking. A row is already one atomic semantic unit."""
    return [
        Chunk(
            chunk_id=f"row_{index}",
            document=record["document"],
            source_type="xlsx",
            text=record["text"],
            metadata=dict(record["metadata"]),
        )
        for index, record in enumerate(records)
    ]


def save_chunks(chunks: list[Chunk], config: Config = CONFIG,
                filename: str = "chunks.json") -> Path:
    config.ensure_dirs()
    out_path = config.processed_dir / filename
    out_path.write_text(
        json.dumps([c.to_dict() for c in chunks], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Saved {len(chunks)} chunks to {out_path}")
    return out_path


def load_chunks(config: Config = CONFIG, filename: str = "chunks.json") -> list[dict]:
    """Chunks as plain dicts — the shape vector_store and payload expect."""
    return json.loads((config.processed_dir / filename).read_text(encoding="utf-8"))
