"""
DeepEval RAG scorers on MLflow, judged by a local Ollama model.

    Faithfulness         generation  is the answer grounded in retrieved context?
    AnswerRelevancy      generation  does the answer address the question?
    ContextualPrecision  retrieval   are relevant chunks ranked above irrelevant ones?
    ContextualRecall     retrieval   does the context contain everything the answer needs?

Usage:
    python mlflow/ragflow.py smoke
    python mlflow/ragflow.py generate
    python mlflow/ragflow.py judge --run-id <id>
    python mlflow/ragflow.py all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import mlflow
import mlflow.genai as mlflow_genai
from mlflow.entities import SpanType
from mlflow.genai.scorers.deepeval import (
    AnswerRelevancy,
    ContextualPrecision,
    ContextualRecall,
    ContextualRelevancy,
    Faithfulness,
)

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "enterprise-rag-eval")

# Ollama is a *native* MLflow provider — no API key, no litellm.
# Base URL is fixed at http://localhost:11434/v1, no env override.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "ollama:/qwen3:4b-instruct")

# temperature=0 is not optional — the judge's JSON is prompt-injected, not
# schema-enforced, so any sampling raises the chance of an unparseable reply.
JUDGE_KWARGS: dict[str, Any] = {"temperature": 0.0}

THRESHOLD = 0.5

DATASET_PATH = PROJECT_ROOT / "evaluation" / "datasets" / "rag_qa.json"

# ContextualRelevancy is the weakest of the five and doubles judge time.
INCLUDE_CONTEXTUAL_RELEVANCY = False


# --------------------------------------------------------------------------
# Scorers
# --------------------------------------------------------------------------


def build_scorers() -> list[Any]:
    """The four core RAG scorers, all pointed at the local judge.

    Every scorer needs an explicit `model=`. Omit it and MLflow falls back to
    its default judge (OpenAI gpt-4o-mini) and will look for OPENAI_API_KEY.
    """
    common = {"model": JUDGE_MODEL, "threshold": THRESHOLD, "model_kwargs": JUDGE_KWARGS}
    scorers = [
        Faithfulness(**common),
        AnswerRelevancy(**common),
        ContextualPrecision(**common),
        ContextualRecall(**common),
    ]
    if INCLUDE_CONTEXTUAL_RELEVANCY:
        scorers.append(ContextualRelevancy(**common))
    return scorers


# --------------------------------------------------------------------------
# Tracing — this is what feeds retrieval_context to the scorers
# --------------------------------------------------------------------------
#
# Three of the four scorers read `retrieval_context`, and MLflow builds that
# ONLY from spans of type RETRIEVER. There is no way to hand it in as a plain
# dict. The retriever span's output must be a list of dicts carrying the chunk
# text under one of: "page_content", "content", "text". Anything else and
# retrieval_context comes back empty and those three scorers silently score
# against nothing.


def as_retriever_docs(chunks: Iterable[Any]) -> list[dict[str, Any]]:
    """Normalise retriever output into the shape MLflow recognises."""
    docs = []
    for c in chunks:
        if isinstance(c, dict):
            text = c.get("page_content") or c.get("content") or c.get("text") or ""
            meta = c.get("metadata") or {}
        else:  # LangChain Document, or anything with .page_content
            text = getattr(c, "page_content", None) or getattr(c, "text", "") or str(c)
            meta = getattr(c, "metadata", {}) or {}
        docs.append({"page_content": text, "metadata": meta})
    return docs


def traced_rag(
    retrieve_fn: Callable[[str], Sequence[Any]],
    generate_fn: Callable[[str, Sequence[Any]], str],
) -> Callable[[str], str]:
    """Wrap an existing retrieve/generate pair so MLflow records proper spans.

    When you are ready to make it permanent, move the decorators onto the real
    functions in src/ — same span boundaries, and the OTel work later reuses them.
    """

    @mlflow.trace(span_type=SpanType.RETRIEVER, name="retrieve")
    def _retrieve(question: str) -> list[dict[str, Any]]:
        return as_retriever_docs(retrieve_fn(question))

    @mlflow.trace(span_type=SpanType.LLM, name="generate")
    def _generate(question: str, docs: list[dict[str, Any]]) -> str:
        return generate_fn(question, docs)

    @mlflow.trace(span_type=SpanType.CHAIN, name="rag")
    def _rag(question: str) -> str:
        docs = _retrieve(question)
        return _generate(question, docs)

    return _rag


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------


def load_dataset(path: Path = DATASET_PATH) -> list[dict[str, str]]:
    """Reshape {query, reference_answer} into what the scorers expect.

    ContextualPrecision and ContextualRecall both need a ground-truth answer,
    and they look for it under the key `expected_output` specifically.
    """
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        {
            "question": r.get("query") or r["question"],
            "expected_output": r.get("reference_answer") or r["expected_output"],
        }
        for r in rows
    ]


# --------------------------------------------------------------------------
# Phase 1 — generate answers, log traces
# --------------------------------------------------------------------------


def generate_traces(
    answer_fn: Callable[[str], str],
    dataset: list[dict[str, str]],
    run_name: str = "generate",
) -> str:
    """Run the pipeline over every question. Only the generator is resident here."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name=run_name) as run:
        for i, row in enumerate(dataset, 1):
            print(f"  [{i}/{len(dataset)}] {row['question'][:70]}")
            answer_fn(row["question"])

            # Attach ground truth to the trace so the scorers can find it later.
            trace_id = mlflow.get_last_active_trace_id()
            if trace_id:
                mlflow.log_expectation(
                    trace_id=trace_id,
                    name="expected_output",
                    value=row["expected_output"],
                )
        print(f"\n  run_id: {run.info.run_id}")
        return run.info.run_id


# --------------------------------------------------------------------------
# Phase 2 — judge the stored traces
# --------------------------------------------------------------------------


def judge_traces(run_id: str):
    """Score already-generated traces. Only the judge is resident here.

    Documented mode 1 of mlflow.genai.evaluate: a DataFrame with a `trace`
    column from mlflow.search_traces. The scorers pull inputs, outputs,
    retrieval_context and expectations straight off each trace.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    trace_df = mlflow.search_traces(run_id=run_id)
    if len(trace_df) == 0:
        raise SystemExit(f"No traces found for run_id={run_id}")
    print(f"  judging {len(trace_df)} traces with {JUDGE_MODEL}")

    results = mlflow_genai.evaluate(data=trace_df, scorers=build_scorers())
    print("\n  metrics:")
    for k, v in results.metrics.items():
        print(f"    {k}: {v}")
    return results


# --------------------------------------------------------------------------
# Smoke test — run this before the full set
# --------------------------------------------------------------------------


def smoke(answer_fn: Callable[[str], str], dataset: list[dict[str, str]]) -> bool:
    """One question through the whole path, with each scorer called directly.

    A small judge sometimes returns JSON the scorer cannot parse. MLflow catches
    that and marks the row errored rather than crashing, so a full run can come
    back quietly half-empty. Better to find out on one row than on forty.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    row = dataset[0]
    print(f"  question: {row['question']}")
    with mlflow.start_run(run_name="smoke"):
        answer_fn(row["question"])
        trace_id = mlflow.get_last_active_trace_id()
        if trace_id is None:
            raise RuntimeError("No active trace was created for the smoke test")
        mlflow.log_expectation(
            trace_id=trace_id, name="expected_output", value=row["expected_output"]
        )

    trace = mlflow.get_trace(trace_id)
    if trace is None:
        raise RuntimeError("Unable to retrieve the smoke-test trace")

    if not [s for s in trace.data.spans if s.span_type == SpanType.RETRIEVER]:
        print("  !! no RETRIEVER span — the three context scorers will score nothing")
        return False

    ok = True
    for scorer in build_scorers():
        fb = scorer(trace=trace)
        if fb.error:
            ok = False
            print(f"  {scorer.name:20} ERROR  {fb.error}")
        else:
            print(f"  {scorer.name:20} {fb.value}  score={fb.metadata['score']}")
    return ok


# --------------------------------------------------------------------------
# Wire in your pipeline here  <-- THE ONLY PART YOU MUST EDIT
# --------------------------------------------------------------------------


def get_answer_fn() -> Callable[[str], str]:
    """Point this at your pipeline.

    Two shapes are needed:
        retrieve_fn(question) -> sequence of chunks   (any object type)
        generate_answer_fn(question, docs) -> str

    If src/ already has a single end-to-end answer(question) -> str, return it
    directly instead — but then add @mlflow.trace(span_type=SpanType.RETRIEVER)
    to your retrieve() in src/, or the three context scorers get nothing.
    """
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.retrieval import retrieve  # noqa: PLC0415
    from src.generation import generate_answer  # noqa: PLC0415

    def _generate_answer(question: str, _docs: Sequence[Any]) -> str:
        result = generate_answer(question)
        if isinstance(result, dict):
            answer = result.get("answer") or result.get("response") or result.get("output")
            return str(answer if answer is not None else result)
        return str(result)

    return traced_rag(retrieve, _generate_answer)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["smoke", "generate", "judge", "all"])
    p.add_argument("--run-id", help="run_id to judge (required for `judge`)")
    args = p.parse_args()

    # One model resident at a time, or Ollama keeps both loaded and spills to CPU.
    os.environ.setdefault("OLLAMA_MAX_LOADED_MODELS", "1")

    dataset = load_dataset()

    if args.command == "smoke":
        print("\n== smoke ==")
        sys.exit(0 if smoke(get_answer_fn(), dataset) else 1)

    elif args.command == "generate":
        print("\n== phase 1: generate ==")
        generate_traces(get_answer_fn(), dataset)

    elif args.command == "judge":
        if not args.run_id:
            p.error("--run-id is required for `judge`")
        print("\n== phase 2: judge ==")
        judge_traces(args.run_id)

    elif args.command == "all":
        print("\n== phase 1: generate ==")
        run_id = generate_traces(get_answer_fn(), dataset)
        print("\n== phase 2: judge ==")
        judge_traces(run_id)


if __name__ == "__main__":
    main()
