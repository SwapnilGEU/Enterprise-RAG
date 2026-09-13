"""RAG answer-quality evaluation — the retrieval half, without the agent.

This is where MLflow goes. Right now it runs the golden questions through
generate_answer(), judges each with the local model, and writes a CSV; the
MLflow block at the bottom is commented out and ready to fill in.

    python evaluation/eval_rag.py
    python evaluation/eval_rag.py --top-k 8        # one point in the sweep
    python evaluation/eval_rag.py --limit 3

Worth remembering when you wire up MLflow: the value is comparing runs, not
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

        print(f"  time:   {elapsed:.1f}s   degraded: {result['degraded']}")
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
    print(f"Total time:                {df['seconds'].sum():.0f}s\n")
    print(df[["query", "retrieved_count", "answer_correct", "seconds"]].to_string(index=False))

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
    #     })
    #     mlflow.log_artifact(str(out))


if __name__ == "__main__":
    main()
