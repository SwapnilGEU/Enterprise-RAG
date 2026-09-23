"""
MLflow block at the bottom is commented out and ready to fill in.

    python evaluation/eval_rag.py
    python evaluation/eval_rag.py --top-k 8        # one point in the sweep
    python evaluation/eval_rag.py --limit 3
When you wire up MLflow: the value is comparing runs, not
logging one. The parameters worth sweeping are --top-k, and in CONFIG the
semantic_chunk_percentile and semantic_chunk_max_chars (both need a re-index).
"""

import _common  # noqa: F401

import argparse
import json
import time

import pandas as pd

from _common import DATASETS, RESULTS, judge, show
from src.config import CONFIG
from src.generation import generate_answer, get_llm
from src.payload import format_source


def _ms(value) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{value / 1000:.2f}s"


def _rate(value) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{value:.1f} tok/s"


def _num(value) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{value:.0f}"


def performance_summary(df: pd.DataFrame) -> dict:
    """Aggregate latency and token metrics. Means skip missing values, so a
    case that failed before generation does not drag the averages to zero.

    run_total_tokens_per_sec is total tokens / total latency over the whole
    run — the same ratio as the per-query column, but weighted by duration
    rather than averaged per query, so one slow query cannot hide."""
    def mean(col):
        value = pd.to_numeric(df[col], errors="coerce").mean()
        return None if pd.isna(value) else round(float(value), 2)

    tokens = int(pd.to_numeric(df["total_tokens"], errors="coerce").fillna(0).sum())
    latency_s = float(pd.to_numeric(df["total_latency_ms"], errors="coerce").fillna(0).sum()) / 1000
    p95 = pd.to_numeric(df["total_latency_ms"], errors="coerce").quantile(0.95)
    return {
        "mean_retrieval_latency_ms": mean("retrieval_latency_ms"),
        "mean_llm_latency_ms": mean("llm_latency_ms"),
        "mean_total_latency_ms": mean("total_latency_ms"),
        "p95_total_latency_ms": None if pd.isna(p95) else round(float(p95), 1),
        "mean_prompt_tokens": mean("prompt_tokens"),
        "mean_completion_tokens": mean("completion_tokens"),
        "mean_total_tokens": mean("total_tokens"),
        "sum_total_tokens": tokens,
        "mean_tokens_per_sec": mean("tokens_per_sec"),
        "mean_total_tokens_per_sec": mean("total_tokens_per_sec"),
        "run_total_tokens_per_sec": round(tokens / latency_s, 2) if latency_s > 0 and tokens else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG answer quality")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None, help="override CONFIG.final_top_k")
    parser.add_argument("--preview", type=int, default=600)
    parser.add_argument("--dataset", default=str(DATASETS / "rag_qa.json"))
    args = parser.parse_args()

    dataset = json.loads(open(args.dataset, encoding="utf-8").read())
    if args.limit:
        dataset = dataset[:args.limit]

    top_k = args.top_k or CONFIG.final_top_k
    print("=" * 100)
    print(f"RAG EVALUATION — {len(dataset)} cases | top_k={top_k} | model={CONFIG.ollama_model}")
    print("=" * 100)

    results = []
    for number, case in enumerate(dataset, start=1):
        query = case["query"]
        print(f"\n[{number}/{len(dataset)}] {query}")
        print("-" * 100)

        started = time.time()
        result = generate_answer(query, top_k=top_k)
        elapsed = time.time() - started

        citations = [format_source(meta) for meta in result["sources"]]
        m = result.get("metrics") or {}

        print(f"  time:   {elapsed:.1f}s   degraded: {result['degraded']}")
        print(f"  split:  retrieval {_ms(m.get('retrieval_latency_ms'))}  "
              f"llm {_ms(m.get('llm_latency_ms'))}  | tokens {m.get('prompt_tokens', 0)} in + "
              f"{m.get('completion_tokens', 0)} out = {m.get('total_tokens', 0)}  "
              f"| {_rate(m.get('tokens_per_sec'))} gen, {_rate(m.get('total_tokens_per_sec'))} total")
        for citation in citations:
            print(f"  source: {citation}")
        show("expected:", case["reference_answer"])
        show("ANSWER:", result["answer"], limit=args.preview)

        verdict, is_correct = judge(get_llm(), query, case["reference_answer"], result["answer"])
        print(f"  judge:  {verdict[:80]!r} -> {'CORRECT' if is_correct else 'INCORRECT'}")

        results.append({
            "query": query,
            "reference_answer": case["reference_answer"],
            "actual_answer": result["answer"],
            "sources": " | ".join(citations),
            "retrieved_count": len(result["sources"]),
            # a source with no heading means provenance was lost upstream
            "sources_without_heading": sum(1 for m in result["sources"] if not m.get("section_heading")),
            "degraded": result["degraded"],
            "judge_verdict": verdict,
            "answer_correct": is_correct,
            "seconds": round(elapsed, 1),
            # Performance. Measured inside generate_answer, so the judge call
            # above is never counted against the system under test.
            "retrieval_latency_ms": m.get("retrieval_latency_ms"),
            "llm_latency_ms": m.get("llm_latency_ms"),
            "total_latency_ms": m.get("total_latency_ms"),
            "prompt_tokens": m.get("prompt_tokens"),
            "completion_tokens": m.get("completion_tokens"),
            "total_tokens": m.get("total_tokens"),
            "tokens_per_sec": m.get("tokens_per_sec"),
            "total_tokens_per_sec": m.get("total_tokens_per_sec"),
        })

    df = pd.DataFrame(results)
    total = len(df)
    answer_accuracy = df["answer_correct"].sum() / total

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Answer correctness:        {df['answer_correct'].sum()}/{total}  ({answer_accuracy * 100:.0f}%)")
    print(f"Degraded responses:        {df['degraded'].sum()}")
    print(f"Sources missing a heading: {df['sources_without_heading'].sum()}")
    print(f"Mean latency:              {df['seconds'].mean():.1f}s")
    print(f"Total time:                {df['seconds'].sum():.0f}s")

    perf = performance_summary(df)
    print("\nPERFORMANCE (means per query unless stated)")
    print(f"  Retrieval latency:       {_ms(perf['mean_retrieval_latency_ms'])}")
    print(f"  LLM latency:             {_ms(perf['mean_llm_latency_ms'])}")
    print(f"  Total latency:           {_ms(perf['mean_total_latency_ms'])}  (p95 {_ms(perf['p95_total_latency_ms'])})")
    print(f"  Prompt tokens:           {_num(perf['mean_prompt_tokens'])}")
    print(f"  Completion tokens:       {_num(perf['mean_completion_tokens'])}")
    print(f"  Total tokens:            {_num(perf['mean_total_tokens'])}  (run total {perf['sum_total_tokens']:,})")
    print(f"  Tokens/s (generation):   {_rate(perf['mean_tokens_per_sec'])}")
    print(f"  Total tokens/s:          {_rate(perf['mean_total_tokens_per_sec'])}  "
          f"(run-wide {_rate(perf['run_total_tokens_per_sec'])})\n")
    print(df[["query", "retrieved_count", "answer_correct", "seconds",
              "total_tokens", "tokens_per_sec"]].to_string(index=False))

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"rag_evaluation_topk{top_k}.csv"
    df.to_csv(out, index=False)
    print(f"\nReport saved to {out}")

    # ---- MLflow goes here -------------------------------------------------
    # The point is the comparison across runs, so log the parameters that
    # changed as well as the metric:
    #
    # import mlflow
    # mlflow.set_tracking_uri("http://localhost:5000")
    # mlflow.set_experiment("rag_quality")
    # with mlflow.start_run(run_name=f"topk={top_k}"):
    #     mlflow.log_params({
    #         "top_k": top_k,
    #         "hybrid_prefetch_limit": CONFIG.hybrid_prefetch_limit,
    #         "semantic_chunk_percentile": CONFIG.semantic_chunk_percentile,
    #         "semantic_chunk_max_chars": CONFIG.semantic_chunk_max_chars,
    #         "dense_model": CONFIG.dense_model,
    #         "llm": CONFIG.ollama_model,
    #     })
    #     mlflow.log_metrics({
    #         "answer_accuracy": answer_accuracy,
    #         "mean_latency_s": float(df["seconds"].mean()),
    #         "degraded_count": int(df["degraded"].sum()),
    #         # None values must be dropped — mlflow rejects them
    #         **{k: v for k, v in perf.items() if v is not None},
    #     })
    #     mlflow.log_artifact(str(out))


if __name__ == "__main__":
    main()
