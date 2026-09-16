"""Retrieval metrics on MLflow — deterministic, no LLM judge anywhere.

    mlflow server                             # in one terminal, from the repo root
    python evaluation/label_chunks.py         # once, to create the labels
    python mlflow/retrievalflow.py            # then as often as you like

Third sibling of `ragflow.py` (answer quality) and `agentflow.py` (agent
behaviour), and the cheapest of the three by a wide margin: it asks only
whether the right chunks came back, which is a set comparison. **No judge, no
Ollama, no generation.** It cannot be rate limited, costs nothing per run, and
finishes in about as long as the searches take.

That is what makes it the right tool for the question you actually have — *I
added documents, is retrieval better or worse?* An answer-quality score moves
for many reasons at once (the generator, the prompt, the judge's mood). These
numbers move only when retrieval changes.

It also separates two failures an answer-level metric cannot tell apart:

  * **recall@20** is the prefetch pool — what dense + sparse managed to find at
    all. Low here means the chunk is missing, badly embedded, or the query has
    no lexical overlap. Reranking cannot fix it.
  * **recall@5** is what survives the ColBERT rerank. If recall@20 is high and
    recall@5 is low, retrieval found the right chunk and the reranker buried
    it — a completely different fix.

Metrics, at k in 1, 3, 5, 10, 20
-------------------------------
  hit_rate@k   did ANY relevant chunk appear in the top k (per question, 0 or 1)
  mrr@k        1 / rank of the first relevant chunk, else 0
  recall@k     how many of the relevant chunks were found, over how many exist
  precision@k  how many of the top k were relevant

Averaged over questions that have at least one label. Unlabelled questions are
**excluded, not scored zero** — a question the corpus genuinely cannot answer
says nothing about retrieval quality, and scoring it zero would drag every
number down and hide real movement.
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import mlflow  # noqa: E402

from src.config import CONFIG  # noqa: E402

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "enterprise-retrieval-eval")
DEFAULT_DATA = PROJECT_ROOT / "evaluation" / "datasets" / "rag_qa.labelled.json"

K_VALUES = (1, 3, 5, 10, 20)


# --- the metrics ------------------------------------------------------------
#
# Kept as free functions over plain lists so they are testable without Qdrant,
# MLflow or a network. This is the part where an off-by-one would not crash,
# it would just quietly report a wrong number forever.


def hit_rate_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """1.0 if any relevant chunk is in the top k, else 0.0."""
    return 1.0 if relevant.intersection(ranked[:k]) else 0.0


def mrr_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Reciprocal rank of the FIRST relevant chunk. Rank is 1-based, so a hit at
    position 0 of the list scores 1.0, not 0.5 — the classic off-by-one here."""
    for index, key in enumerate(ranked[:k], start=1):
        if key in relevant:
            return 1.0 / index
    return 0.0


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the relevant chunks that appear in the top k."""
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the top k that are relevant.

    Divided by k, not by len(ranked[:k]): if the search returns fewer than k
    results, the missing slots are genuinely wasted and precision should say so.
    """
    if k <= 0:
        return 0.0
    return len(relevant.intersection(ranked[:k])) / k


def score_question(ranked: list[str], relevant: set[str]) -> dict[str, float]:
    scores = {}
    for k in K_VALUES:
        scores[f"hit_rate@{k}"] = hit_rate_at_k(ranked, relevant, k)
        scores[f"mrr@{k}"] = mrr_at_k(ranked, relevant, k)
        scores[f"recall@{k}"] = recall_at_k(ranked, relevant, k)
        scores[f"precision@{k}"] = precision_at_k(ranked, relevant, k)
    return scores


def first_hit_rank(ranked: list[str], relevant: set[str]) -> int | None:
    for index, key in enumerate(ranked, start=1):
        if key in relevant:
            return index
    return None


# --- data -------------------------------------------------------------------


def key_of(payload: dict) -> str:
    """Must match `label_chunks.key_of` and `vector_store.point_id`."""
    return f"{payload.get('document', '?')}::{payload.get('chunk_id', '?')}"


def load_labelled(path: Path) -> tuple[list[dict], list[str]]:
    """Returns (rows with labels, queries that have none)."""
    if not path.exists():
        raise SystemExit(
            f"Labelled dataset not found: {path}\n"
            "Create it first — it is the one manual step:\n"
            "    python evaluation/label_chunks.py\n"
            "It shows you candidate chunks per question and asks which are relevant."
        )

    rows = json.loads(path.read_text("utf-8"))
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"{path} must be a non-empty JSON list.")

    labelled, unlabelled = [], []
    for row in rows:
        if row.get("relevant_chunk_ids"):
            labelled.append(row)
        else:
            unlabelled.append(row.get("query", "(no query)"))

    if not labelled:
        raise SystemExit(
            f"{path} has no labelled questions at all.\n"
            "Every row's relevant_chunk_ids is empty — run label_chunks.py."
        )
    return labelled, unlabelled


# --- run it -----------------------------------------------------------------


def evaluate(rows: list[dict], top_k: int, quiet: bool) -> tuple[list[dict], dict[str, float]]:
    from src.retrieval import retrieve_points

    per_question = []
    for number, row in enumerate(rows, start=1):
        query = row["query"]
        relevant = set(row["relevant_chunk_ids"])

        points = retrieve_points(query, top_k=top_k)
        ranked = [key_of(dict(p.payload or {})) for p in points]

        scores = score_question(ranked, relevant)
        rank = first_hit_rank(ranked, relevant)
        found = len(relevant.intersection(ranked))

        per_question.append(
            {
                "query": query,
                "n_relevant": len(relevant),
                "n_found": found,
                "first_hit_rank": rank,
                **scores,
            }
        )

        if not quiet:
            marker = f"rank {rank}" if rank else "MISS"
            print(
                f"[{number}/{len(rows)}] {marker:>8}  "
                f"found {found}/{len(relevant)}  {query[:64]}"
            )

    aggregate = {}
    for name in per_question[0]:
        if name in ("query", "n_relevant", "n_found", "first_hit_rank"):
            continue
        aggregate[name] = statistics.mean(q[name] for q in per_question)
    return per_question, aggregate


def print_table(aggregate: dict[str, float]) -> None:
    print("\n" + "=" * 62)
    print(f"{'k':>4}  {'hit_rate':>9}  {'mrr':>7}  {'recall':>8}  {'precision':>10}")
    print("-" * 62)
    for k in K_VALUES:
        print(
            f"{k:>4}  {aggregate[f'hit_rate@{k}']:>9.3f}  {aggregate[f'mrr@{k}']:>7.3f}  "
            f"{aggregate[f'recall@{k}']:>8.3f}  {aggregate[f'precision@{k}']:>10.3f}"
        )
    print("=" * 62)

    # The diagnostic the whole module exists for.
    pool, final = aggregate["recall@20"], aggregate["recall@5"]
    print(f"\nrecall@20 (prefetch pool): {pool:.3f}")
    print(f"recall@5  (after rerank):  {final:.3f}")
    if pool - final > 0.15:
        print(
            "  -> The chunks ARE being found and the reranker is burying them.\n"
            "     Look at the ColBERT rerank and final_top_k, not at chunking or embeddings."
        )
    elif pool < 0.6:
        print(
            "  -> The prefetch pool itself is missing chunks, so reranking cannot help.\n"
            "     Look at chunking, the dense model, or hybrid_prefetch_limit."
        )
    else:
        print("  -> Pool and final ranking broadly agree; no obvious rerank pathology.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="labelled dataset")
    parser.add_argument(
        "--top-k", type=int, default=max(K_VALUES),
        help=f"how many to retrieve (default {max(K_VALUES)} — the largest k scored)",
    )
    parser.add_argument("--label", default=None, help="run name, e.g. 'after adding 3 papers'")
    parser.add_argument("--no-mlflow", action="store_true", help="print only, log nothing")
    parser.add_argument("--quiet", action="store_true", help="suppress per-question lines")
    args = parser.parse_args()

    if args.top_k < max(K_VALUES):
        print(
            f"warning: --top-k {args.top_k} is below the largest scored k ({max(K_VALUES)}); "
            "metrics above it will be truncated, not missing."
        )

    rows, unlabelled = load_labelled(args.data)
    print(f"dataset: {args.data.name} — {len(rows)} labelled question(s)")
    if unlabelled:
        print(f"  excluding {len(unlabelled)} unlabelled: " + "; ".join(q[:40] for q in unlabelled))

    per_question, aggregate = evaluate(rows, args.top_k, args.quiet)
    print_table(aggregate)

    if args.no_mlflow:
        return 0

    mlflow.set_tracking_uri(TRACKING_URI)
    try:
        mlflow.set_experiment(EXPERIMENT)
    except Exception as exc:
        raise SystemExit(
            f"Could not reach the MLflow tracking server at {TRACKING_URI} ({exc}).\n"
            "Start it first:  mlflow server   (or pass --no-mlflow)"
        ) from exc

    with mlflow.start_run(run_name=args.label):
        # Params are what makes two runs comparable later. Anything that could
        # change a retrieval number belongs here, or a future you will be
        # staring at two different scores with no idea what differed.
        mlflow.log_params(
            {
                "collection": CONFIG.collection_name,
                "dense_model": CONFIG.dense_model,
                "sparse_model": CONFIG.sparse_model,
                "late_interaction_model": CONFIG.late_interaction_model,
                "hybrid_prefetch_limit": CONFIG.hybrid_prefetch_limit,
                "final_top_k": CONFIG.final_top_k,
                "retrieved_top_k": args.top_k,
                "dataset": args.data.name,
                "n_questions": len(rows),
                "n_relevant_total": sum(len(r["relevant_chunk_ids"]) for r in rows),
            }
        )

        # Corpus size is the number you are usually varying, so record it.
        try:
            from src.vector_store import get_client

            info = get_client().get_collection(CONFIG.collection_name)
            mlflow.log_param("points_in_collection", info.points_count)
        except Exception as exc:  # noqa: BLE001 — nice to have, never fatal
            print(f"(could not read collection size: {exc})")

        mlflow.log_metrics(aggregate)

        report = PROJECT_ROOT / "evaluation" / "results" / "retrieval_metrics.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps({"aggregate": aggregate, "per_question": per_question}, indent=2),
            encoding="utf-8",
        )
        mlflow.log_artifact(str(report))

        print(f"\nlogged to {TRACKING_URI}  experiment: {EXPERIMENT}")
        print(f"per-question detail: {report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
