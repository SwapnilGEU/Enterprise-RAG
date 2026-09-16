"""Label which chunks actually answer each golden question — run once, by hand.

    python evaluation/label_chunks.py              # interactive, resumable
    python evaluation/label_chunks.py --dump       # write candidates to review offline
    python evaluation/label_chunks.py --relabel    # start the whole set again

Writes `evaluation/datasets/rag_qa.labelled.json`, which is what
`mlflow/retrievalflow.py` needs to compute hit-rate@k, MRR and recall@k. Those
metrics are deterministic and judge-free, so once this file exists you can
measure "did adding documents help?" for free and without a judge quota.

This is the one genuinely manual step in the pipeline. Ten questions, a couple
of minutes each.

Why candidates come from three retrievers, not one
--------------------------------------------------
The obvious shortcut is to label whatever the current pipeline returns. That
quietly guarantees a good score: you would be marking the current ranking
correct by construction, and any future change could only look worse. The
labels would encode today's config rather than the truth.

So candidates are the **union** of three independent searches:

  * dense only   — semantic, catches paraphrases the keyword search misses
  * sparse only  — BM25, catches rare terms and exact model names
  * hybrid + ColBERT rerank — what the pipeline actually does

A chunk that only dense finds, or only sparse finds, still reaches you for
judgement. The labels then describe the corpus, and the current pipeline is
free to score badly against them — which is the entire point of having them.

Identity is `document::chunk_id`, matching `vector_store.point_id`, because
`chunk_id` alone is only unique within a document.
"""

import _common  # noqa: F401  (must come first — sets sys.path)

import argparse
import json
import sys
from pathlib import Path

from qdrant_client import models

from _common import DATASETS
from src.config import CONFIG
from src.vector_store import get_client

LABELLED = DATASETS / "rag_qa.labelled.json"
GOLDEN = DATASETS / "rag_qa.json"

# Wide enough that a relevant chunk is unlikely to fall outside it, small enough
# to stay reviewable by a human. This is the candidate pool, not a metric cutoff.
CANDIDATES_PER_STRATEGY = 10


def key_of(payload: dict) -> str:
    """The chunk identity used everywhere downstream."""
    return f"{payload.get('document', '?')}::{payload.get('chunk_id', '?')}"


# --- candidate generation ---------------------------------------------------


def dense_candidates(client, query: str, limit: int):
    return client.query_points(
        CONFIG.collection_name,
        query=models.Document(text=query, model=CONFIG.dense_model),
        using="dense",
        limit=limit,
        with_payload=True,
    ).points


def sparse_candidates(client, query: str, limit: int):
    return client.query_points(
        CONFIG.collection_name,
        query=models.Document(text=query, model=CONFIG.sparse_model),
        using="sparse",
        limit=limit,
        with_payload=True,
    ).points


def hybrid_candidates(client, query: str, limit: int):
    """The pipeline's own retrieval — same shape as `retrieval.retrieve_points`."""
    return client.query_points(
        CONFIG.collection_name,
        prefetch=[
            models.Prefetch(
                query=models.Document(text=query, model=CONFIG.dense_model),
                using="dense",
                limit=CONFIG.hybrid_prefetch_limit,
            ),
            models.Prefetch(
                query=models.Document(text=query, model=CONFIG.sparse_model),
                using="sparse",
                limit=CONFIG.hybrid_prefetch_limit,
            ),
        ],
        query=models.Document(text=query, model=CONFIG.late_interaction_model),
        using="multi",
        limit=limit,
        with_payload=True,
    ).points


def gather(client, query: str, limit: int) -> list[dict]:
    """Union of the three strategies, each chunk recording where it came from.

    Ordered by how many strategies found it, then by best rank — so chunks all
    three agree on come first and the long tail of single-strategy finds comes
    last. That ordering is a convenience for review only; it has no effect on
    the labels themselves.
    """
    found: dict[str, dict] = {}

    for strategy, fetch in (
        ("hybrid", hybrid_candidates),
        ("dense", dense_candidates),
        ("sparse", sparse_candidates),
    ):
        try:
            points = fetch(client, query, limit)
        except Exception as exc:  # noqa: BLE001 — one strategy failing is survivable
            print(f"    ({strategy} search failed: {exc})")
            continue

        for rank, point in enumerate(points, start=1):
            payload = dict(point.payload or {})
            key = key_of(payload)
            entry = found.setdefault(
                key,
                {"key": key, "payload": payload, "ranks": {}, "score": getattr(point, "score", None)},
            )
            entry["ranks"][strategy] = rank

    return sorted(
        found.values(),
        key=lambda e: (-len(e["ranks"]), min(e["ranks"].values())),
    )


# --- presentation -----------------------------------------------------------


def describe(entry: dict, width: int = 320) -> str:
    payload = entry["payload"]
    heading = payload.get("section_heading") or payload.get("section_title") or "(no heading)"
    page = payload.get("page_label") or ""
    ranks = ", ".join(f"{s}#{r}" for s, r in sorted(entry["ranks"].items()))
    text = " ".join(str(payload.get("text", "")).split())
    if len(text) > width:
        text = text[:width].rstrip() + "…"
    return (
        f"  {payload.get('document', '?')} — {heading} {page}\n"
        f"  found by: {ranks}\n"
        f"  {text}"
    )


# --- the loop ---------------------------------------------------------------


def load_existing() -> dict[str, list[str]]:
    if not LABELLED.exists():
        return {}
    rows = json.loads(LABELLED.read_text("utf-8"))
    return {r["query"]: r.get("relevant_chunk_ids", []) for r in rows if r.get("relevant_chunk_ids")}


def save(golden: list[dict], labels: dict[str, list[str]]) -> None:
    """Write after every question, so a Ctrl-C costs you one answer, not ten."""
    out = []
    for row in golden:
        out.append(
            {
                "query": row["query"],
                "reference_answer": row["reference_answer"],
                "relevant_chunk_ids": labels.get(row["query"], []),
            }
        )
    LABELLED.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


def label_interactive(golden: list[dict], limit: int, relabel: bool) -> int:
    client = get_client()
    labels = {} if relabel else load_existing()

    print(f"\n{len(golden)} questions. y = relevant, n = not, s = skip rest of this question, q = save and quit.\n")

    for number, row in enumerate(golden, start=1):
        query = row["query"]
        if query in labels and not relabel:
            print(f"[{number}/{len(golden)}] already labelled ({len(labels[query])} chunks) — skipping")
            continue

        print("=" * 100)
        print(f"[{number}/{len(golden)}] {query}")
        print(f"  reference: {row['reference_answer']}")
        print("=" * 100)

        candidates = gather(client, query, limit)
        if not candidates:
            print("  no candidates returned — is the collection populated?")
            labels[query] = []
            save(golden, labels)
            continue

        chosen: list[str] = []
        for index, entry in enumerate(candidates, start=1):
            print(f"\n[{index}/{len(candidates)}]")
            print(describe(entry))
            answer = input("  relevant? [y/n/s/q] ").strip().lower()
            if answer == "q":
                labels[query] = chosen
                save(golden, labels)
                print(f"\nSaved {LABELLED}. Re-run to continue where you stopped.")
                return 0
            if answer == "s":
                break
            if answer == "y":
                chosen.append(entry["key"])

        labels[query] = chosen
        save(golden, labels)
        print(f"\n  -> {len(chosen)} chunk(s) marked relevant. Saved.")

    print(f"\nDone. {LABELLED}")
    _report(labels)
    return 0


def dump(golden: list[dict], limit: int) -> int:
    """Write every candidate to a file, for labelling away from a terminal.

    Tick chunks by moving their id into `relevant_chunk_ids`. The candidate
    block is there to read from; only `relevant_chunk_ids` is loaded back.
    """
    client = get_client()
    out = []
    for number, row in enumerate(golden, start=1):
        print(f"[{number}/{len(golden)}] gathering candidates for {row['query'][:60]!r}...")
        candidates = gather(client, row["query"], limit)
        out.append(
            {
                "query": row["query"],
                "reference_answer": row["reference_answer"],
                "relevant_chunk_ids": [],
                "_candidates": [
                    {
                        "chunk_id": entry["key"],
                        "found_by": entry["ranks"],
                        "heading": entry["payload"].get("section_heading", ""),
                        "page": entry["payload"].get("page_label", ""),
                        "text": " ".join(str(entry["payload"].get("text", "")).split())[:500],
                    }
                    for entry in candidates
                ],
            }
        )

    target = DATASETS / "rag_qa.candidates.json"
    target.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"\nWrote {target}\n"
        "Move the ids you judge relevant into each row's `relevant_chunk_ids`,\n"
        f"then save the result as {LABELLED.name} in the same folder."
    )
    return 0


def _report(labels: dict[str, list[str]]) -> None:
    empty = [q for q, ids in labels.items() if not ids]
    total = sum(len(ids) for ids in labels.values())
    print(f"  {len(labels)} questions, {total} relevant chunks total")
    if empty:
        print(
            f"  {len(empty)} question(s) have NO relevant chunk:\n"
            + "\n".join(f"    - {q}" for q in empty)
            + "\n  retrievalflow.py excludes these from the averages rather than\n"
            "  scoring them zero — a question the corpus cannot answer says\n"
            "  nothing about retrieval quality."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=GOLDEN, help="golden set JSON")
    parser.add_argument(
        "--limit", type=int, default=CANDIDATES_PER_STRATEGY,
        help=f"candidates per strategy (default {CANDIDATES_PER_STRATEGY})",
    )
    parser.add_argument("--dump", action="store_true", help="write candidates to a file instead")
    parser.add_argument("--relabel", action="store_true", help="ignore existing labels")
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"Golden set not found: {args.data}")

    golden = json.loads(args.data.read_text("utf-8"))
    if not isinstance(golden, list) or not golden:
        raise SystemExit(f"{args.data} must be a non-empty JSON list.")

    if args.dump:
        return dump(golden, args.limit)
    return label_interactive(golden, args.limit, args.relabel)


if __name__ == "__main__":
    raise SystemExit(main())
