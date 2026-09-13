"""Index-time entry point: raw documents -> Qdrant.

This is the notebook's Part I + Part II-up-to-upload, run top to bottom:

    extract -> markdown -> sections -> chunk -> save -> dedup -> embed + upload

Run it whenever the files in data/raw/ change.

    python scripts/run_index.py
    python scripts/run_index.py --force-recreate   # wipe the collection first
    python scripts/run_index.py --dry-run          # chunk and report, upload nothing
"""

import _bootstrap  # noqa: F401  (must come first — sets sys.path)

import argparse

from src.chunking import (
    chunk_structural_units, load_chunks, save_chunks, xlsx_records_to_chunks,
)
from src.config import CONFIG, logger
from src.extraction import run_ingestion, save_documents
from src.markdown import build_markdown
from src.payload import build_payload
from src.sections import build_structural_units
from src.vector_store import classify_chunks, ensure_collection, ensure_payload_indexes, upload_chunks


def build_chunks() -> list[dict]:
    """Everything from raw files to chunks-on-disk. Returns chunk dicts."""
    documents = run_ingestion()
    if not documents:
        logger.warning(f"No records extracted — is {CONFIG.raw_dir} empty?")
        return []
    save_documents(documents)

    markdown_by_type = build_markdown(documents)
    logger.info(
        f"Markdown: {len(markdown_by_type['docx'])} docx, {len(markdown_by_type['pdf'])} pdf"
    )

    units = build_structural_units(markdown_by_type)
    logger.info(f"Structural units: {len(units)}")

    chunks = chunk_structural_units(units)
    chunks += xlsx_records_to_chunks([d for d in documents if d["source_type"] == "xlsx"])
    logger.info(f"Chunks: {len(chunks)}")

    save_chunks(chunks)
    return [c.to_dict() for c in chunks]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the index from data/raw/")
    parser.add_argument("--force-recreate", action="store_true",
                        help="delete and rebuild the Qdrant collection first")
    parser.add_argument("--dry-run", action="store_true",
                        help="chunk and classify, but upload nothing")
    parser.add_argument("--reuse-chunks", action="store_true",
                        help="skip extraction and load data/processed/chunks.json instead")
    args = parser.parse_args()

    chunk_dicts = load_chunks() if args.reuse_chunks else build_chunks()
    if not chunk_dicts:
        return

    # Provenance check before anything is uploaded — a chunk with no heading
    # here is a citation you can never recover at query time.
    without_heading = sum(1 for c in chunk_dicts if not build_payload(c)["section_heading"])
    logger.info(f"Chunks with no section heading: {without_heading}/{len(chunk_dicts)}")

    if args.dry_run:
        # Stop before anything touches the network: --dry-run is for checking
        # extraction and chunking without needing Qdrant credentials at all.
        print(f"\n--dry-run: {len(chunk_dicts)} chunk(s) built, "
              f"{without_heading} without a section heading. Nothing uploaded.")
        return

    ensure_collection(force_recreate=args.force_recreate)
    ensure_payload_indexes()

    buckets = classify_chunks(chunk_dicts)
    print()
    print(f"New (never seen):                 {len(buckets['new'])}")
    print(f"Changed (text differs):           {len(buckets['changed'])}")
    print(f"Stale metadata (no headings yet): {len(buckets['stale_metadata'])}")
    print(f"Unchanged (skip):                 {len(buckets['unchanged'])}")

    to_embed = buckets["new"] + buckets["changed"] + buckets["stale_metadata"]
    uploaded = upload_chunks(to_embed)
    print(f"\nDone. {uploaded} point(s) written to '{CONFIG.collection_name}'.")


if __name__ == "__main__":
    main()
