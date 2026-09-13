"""The provenance contract — notebook Section 12.2 and 14.

These three functions decide what a chunk carries into Qdrant and how it comes
back out as a citation. They live in their own module because when they were
two unrelated notebook cells they silently disagreed: the upload cell wrote a
five-key payload with no section metadata, and the context builder read keys
(`heading_path`, `page`) this pipeline never produced. The result was every
single citation rendering as "[Unknown section, p.?]" while the metadata sat
perfectly intact one stage upstream.

Rule for this file: anything not written by build_payload does not exist at
query time, and anything not read by format_source never reaches the user.
Change them together, and test them together (tests/test_payload.py).
"""

import hashlib


def content_hash(text: str) -> str:
    """Stable identity for a chunk's text — drives dedup in vector_store.py."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def build_payload(chunk: dict) -> dict:
    """The single definition of what a Qdrant point carries.

    `chunk` is a plain dict: chunk_id, document, source_type, text, metadata.
    Everything the chunk knows about where it came from goes in — Qdrant only
    ever returns what the payload holds.
    """
    meta = chunk.get("metadata") or {}

    section_path = meta.get("section_path") or []
    if section_path:
        heading = " > ".join(str(h) for h in section_path if h)
    else:
        # XLSX (and any un-headed section) — fall back to the section title,
        # then the sheet name, which is the closest thing a spreadsheet has.
        heading = meta.get("section_title") or meta.get("sheet") or meta.get("sheet_title") or ""

    page_start, page_end = meta.get("page_start"), meta.get("page_end")
    if page_start is None:
        page_label = ""                                   # DOCX and XLSX have no pages
    elif page_end is not None and page_end != page_start:
        page_label = f"pp.{page_start}–{page_end}"   # a section spanning pages
    else:
        page_label = f"p.{page_start}"

    payload = {
        # identity
        "document": chunk["document"],
        "chunk_id": chunk["chunk_id"],
        "source_type": chunk["source_type"],
        # content
        "text": chunk["text"],
        "content_hash": content_hash(chunk["text"]),
        # provenance
        "section_path": section_path,
        "section_heading": heading,      # pre-flattened for display
        "page_label": page_label,        # pre-formatted for display
    }

    # Carry the rest of the chunk's metadata through untouched (section_title,
    # section_level, page_start/page_end, chunk_index, plus xlsx column metadata).
    for key, value in meta.items():
        payload.setdefault(key, value)

    return payload


def format_source(meta: dict, include_chunk_id: bool = False) -> str:
    """One canonical citation string, e.g.

        [Foundation-LLMs.pdf — Pre-training > Generalization, pp.41-42]

    `meta` is a Qdrant payload as written by build_payload. Degrades gracefully
    when a document genuinely has no headings (XLSX) or no pages (DOCX).
    """
    document = meta.get("document") or "Unknown document"
    heading = meta.get("section_heading") or meta.get("section_title") or "Unknown section"
    page = meta.get("page_label") or ""

    tail = ", ".join(p for p in [" — ".join([document, heading]), page] if p)

    if include_chunk_id and meta.get("chunk_id"):
        tail = f"{tail}  ({meta['chunk_id']})"

    return f"[{tail}]"
