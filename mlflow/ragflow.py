"""RAG evaluation: DeepEval scorers on MLflow, judged by Gemini.

    mlflow server                       # in one terminal, from the repo root
    python mlflow/ragflow.py --smoke    # one row, per scorer — always do this first
    python mlflow/ragflow.py            # the full golden set

Requires GEMINI_API_KEY (repo-root .env or mlflow/.env), a running Ollama, and
a populated Qdrant collection. To judge locally instead, with no API key and no
quota at all:

    python mlflow/ragflow.py --judge ollama:/qwen3:4b-instruct

Two preflights, cheapest first
------------------------------
One judge call, then one traced generation. Both exist because the harness
swallows failures: `DeepEvalScorer.__call__` turns any exception into
`Feedback(error=e)`, so an exhausted quota, an unparseable reply and a missing
retriever span all print the same `'Faithfulness': 1/1 failed`. Finding out
which one it is *before* the run is worth the two calls.

Retrieval context comes ONLY from the trace
-------------------------------------------
Faithfulness, ContextualPrecision and ContextualRecall read
`LLMTestCase.retrieval_context`, which MLflow builds exclusively from top-level
RETRIEVER spans on the trace. There is no way to hand them context through
`inputs` or `expectations`. `src/retrieval.py::retrieve` carries that span —
see the note at the bottom of this file. Without it those three scorers do not
error, they quietly score an empty context and the numbers are meaningless.
This script refuses to run the full set if the smoke trace has no retriever
span, rather than let that happen silently.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# .env may sit at the repo root or next to this script — load both, first wins.
for env_path in (PROJECT_ROOT / ".env", HERE / ".env"):
    if env_path.exists():
        load_dotenv(env_path)

if not os.environ.get("GEMINI_API_KEY"):
    raise SystemExit(
        "GEMINI_API_KEY not found.\n"
        f"Put it in {PROJECT_ROOT / '.env'} (or {HERE / '.env'}) as:\n"
        '    GEMINI_API_KEY="..."'
    )

# Ollama keeps one model resident on a 6GB card; parallel predict_fn workers
# make it thrash. Must be set BEFORE mlflow is imported — the harness reads it
# at import time. Serial is also what makes the logs readable.
os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_WORKERS", "1")
os.environ.setdefault("OLLAMA_MAX_LOADED_MODELS", "1")

import mlflow  # noqa: E402
import mlflow.genai  # noqa: E402
from mlflow.entities import SpanType  # noqa: E402

try:
    from mlflow.genai.scorers.deepeval import (  # noqa: E402
        AnswerRelevancy,
        ContextualPrecision,
        ContextualRecall,
        ContextualRelevancy,
        Faithfulness,
    )
except ImportError as exc:  # deepeval missing, or mlflow too old
    raise SystemExit(
        f"Could not import MLflow's DeepEval scorers ({exc}).\n"
        "    pip install -r mlflow/requirements-eval.txt"
    ) from exc

# Plain `import judge_json`, not `from mlflow.judge_json import ...`: the
# installed mlflow package wins that name over this folder (a regular package
# beats a namespace package in the import scan). This script's own directory is
# sys.path[0], so the bare name resolves to the file next door.
sys.path.insert(0, str(HERE))
from judge_json import install as install_judge_patches  # noqa: E402
import judge_quota  # noqa: E402

from src.generation import generate_answer  # noqa: E402

# Every scorer needs an explicit model=. MLflow's default judge is OpenAI
# gpt-4o-mini, so a scorer built without it reaches for OPENAI_API_KEY and dies.
JUDGE = os.environ.get("GEMINI_JUDGE", "gemini:/gemini-2.5-flash")
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "enterprise-rag-eval")
DEFAULT_DATA = PROJECT_ROOT / "evaluation" / "datasets" / "rag_qa.json"


# --- the app ---------------------------------------------------------------


@mlflow.trace(span_type=SpanType.CHAIN, name="rag")
def rag(question: str) -> str:
    """The thing under evaluation. Returns a plain string: MLflow stringifies
    the output for the judge anyway, and returning the whole result dict puts
    the full context blob in `actual_output`, which AnswerRelevancy then marks
    down for irrelevance."""
    result = generate_answer(question)
    if isinstance(result, dict):
        for key in ("answer", "text", "output"):
            value = result.get(key)
            if isinstance(value, str):
                return value
    return str(result)


# --- the data --------------------------------------------------------------


def load_dataset(path: Path, limit: int | None = None) -> list[dict]:
    """rag_qa.json is a list of {query, reference_answer}; reshape to the
    inputs/expectations pairs mlflow.genai.evaluate wants.

    `expected_output` is not a free choice of key — ContextualPrecision and
    ContextualRecall look for exactly that name.
    """
    if not path.exists():
        raise SystemExit(
            f"Golden dataset not found: {path}\n"
            "Create it as a JSON list of {\"query\": ..., \"reference_answer\": ...}\n"
            f"There is a template at {PROJECT_ROOT / 'evaluation/datasets/rag_qa.example.json'}\n"
            "or point at another file with --data."
        )

    rows = json.loads(path.read_text("utf-8"))
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"{path} must be a non-empty JSON list.")

    dataset = []
    for i, row in enumerate(rows):
        try:
            dataset.append(
                {
                    "inputs": {"question": row["query"]},
                    "expectations": {"expected_output": row["reference_answer"]},
                }
            )
        except KeyError as exc:
            raise SystemExit(
                f"{path} row {i} is missing {exc}; each row needs "
                '"query" and "reference_answer".'
            ) from exc

    return dataset[:limit] if limit else dataset


def build_scorers(judge: str, contextual_relevancy: bool = False) -> list:
    scorers = [
        Faithfulness(model=judge),          # generation — needs the retriever span
        AnswerRelevancy(model=judge),       # generation — input + output only
        ContextualPrecision(model=judge),   # retrieval  — span + expected_output
        ContextualRecall(model=judge),      # retrieval  — span + expected_output
    ]
    if contextual_relevancy:
        scorers.append(ContextualRelevancy(model=judge))  # weakest, doubles judge time
    return scorers


# --- preflight -------------------------------------------------------------


def check_judge(judge: str) -> bool:
    """One judge call, before anything expensive happens.

    This runs first and for one reason: the cheapest failure is the one you
    find before generating anything. A dead judge quota used to surface only
    after a full generation pass, disguised as four scorers "failing" — see
    mlflow/judge_quota.py for why that disguise is so convincing.
    """
    print(f"[preflight] asking the judge one question ({judge}) ...")
    ok, message = judge_quota.probe(judge)
    print(f"[preflight] {'OK — ' if ok else 'FAIL — '}{message}")
    return ok


def check_retriever_span(question: str) -> bool:
    """Run one real question and confirm the trace carries a retriever span
    whose chunks parse. Catches the failure mode where three of four scorers
    score an empty context and report a plausible-looking number."""
    print(f"[preflight] tracing one question: {question!r}")
    with mlflow.start_span(name="preflight") as span:
        answer = rag(question)
        trace_id = span.trace_id

    mlflow.flush_trace_async_logging()
    trace = mlflow.get_trace(trace_id)
    if trace is None:
        print("[preflight] WARNING: could not read the trace back from the server.")
        return False

    from mlflow.genai.utils.trace_utils import extract_retrieval_context_from_trace

    context = extract_retrieval_context_from_trace(trace)
    chunks = [c for chunk_list in context.values() for c in chunk_list]

    print(f"[preflight] answer: {answer[:120]!r}...")
    print(f"[preflight] retriever spans: {len(context)}  parsed chunks: {len(chunks)}")

    if not chunks:
        print(
            "[preflight] FAIL — no retrieval context on the trace.\n"
            "  Faithfulness / ContextualPrecision / ContextualRecall would score\n"
            "  an empty context. Check that src/retrieval.py::retrieve is still\n"
            "  decorated with @mlflow.trace(span_type=SpanType.RETRIEVER) and\n"
            "  returns a list of dicts with the text under 'page_content'."
        )
        return False

    print("[preflight] OK — retrieval context is on the trace.")
    return True


# --- run it ----------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="golden set JSON")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N rows")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="preflights + one row only — catches quota and JSON-parse failures cheaply",
    )
    parser.add_argument("--judge", default=JUDGE, help=f"judge model URI (default {JUDGE})")
    parser.add_argument("--contextual-relevancy", action="store_true", help="add the 5th scorer")
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="skip the retriever-span check (does not skip the judge check)",
    )
    parser.add_argument(
        "--skip-judge-preflight",
        action="store_true",
        help="do not spend one judge call checking the judge is reachable first",
    )
    parser.add_argument(
        "--stock-judge",
        action="store_true",
        help="skip the JSON transport patches (see mlflow/judge_json.py) and use "
        "MLflow's prompt-injected JSON as-is",
    )
    args = parser.parse_args()

    if args.stock_judge:
        os.environ["JUDGE_NATIVE_JSON"] = "0"
        os.environ["JUDGE_JSON_REPAIR"] = "0"
    patches = install_judge_patches()
    print(f"judge JSON patches: {patches}")
    # Installed *after* the JSON patches so the retry wraps the whole
    # native-then-fallback attempt rather than half of it.
    judge_quota.reset()
    print(f"judge quota guard: {judge_quota.install()}")

    mlflow.set_tracking_uri(TRACKING_URI)
    try:
        mlflow.set_experiment(EXPERIMENT)
    except Exception as exc:
        raise SystemExit(
            f"Could not reach the MLflow tracking server at {TRACKING_URI} ({exc}).\n"
            "Start it first:  mlflow server"
        ) from exc

    print(f"tracking: {TRACKING_URI}   experiment: {EXPERIMENT}   judge: {args.judge}")

    limit = 1 if args.smoke else args.limit
    dataset = load_dataset(args.data, limit=limit)
    print(f"dataset: {args.data.name} — {len(dataset)} row(s)")

    # Cheapest check first: one judge call costs a second, a generation pass
    # costs minutes.
    if not args.skip_judge_preflight:
        if not check_judge(args.judge):
            print("\nAborting before generation: the judge cannot answer.")
            return 2

    if not args.skip_preflight:
        ok = check_retriever_span(dataset[0]["inputs"]["question"])
        if not ok and not args.smoke:
            print("Aborting: fix the retriever span, or re-run with --skip-preflight.")
            return 1

    results = mlflow.genai.evaluate(
        data=dataset,
        predict_fn=rag,
        scorers=build_scorers(args.judge, args.contextual_relevancy),
    )

    print("\nmetrics:")
    for name, value in sorted(results.metrics.items()):
        print(f"  {name}: {value}")

    # How much judge did this cost? Nobody guesses this number correctly the
    # first time, and on a metered key it is the number that matters.
    counts = judge_quota.stats()
    print(
        f"\njudge calls: {counts['calls']}"
        f"   retries: {counts['retries']}   failures: {counts['failures']}"
    )

    # A row can error for two very different reasons and they used to print the
    # same sentence. Name the real one.
    errored = any("error" in name.lower() for name in results.metrics)
    if counts["quota_exhausted"]:
        quota_info = judge_quota.tripped()
        if quota_info is None:
            quota_info = judge_quota.QuotaInfo(
                exhausted=counts["quota_exhausted"],
                limit=0,
                calls=counts["calls"],
            )
        print(
            "\n"
            + "=" * 70
            + f"\nRUN INVALID — judge quota ran out mid-run: {counts['quota_exhausted']}\n"
            + "Scores above are not trustworthy: every call after the limit was hit\n"
            "failed without reaching the judge.\n\n"
            + judge_quota.quota_advice(args.judge, quota_info)
            + "\n"
            + "=" * 70
        )
        return 3
    if errored:
        print(
            "\nNOTE: some rows errored. The quota was fine, so the likely cause is the\n"
            "judge returning unparseable JSON. Get the real message with:\n"
            "    python mlflow/diagnose.py --full"
        )

    print(f"\nOpen {TRACKING_URI} to see per-row scores and traces.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------
# The retriever span (already applied in src/retrieval.py):
#
#     @mlflow.trace(span_type=SpanType.RETRIEVER, name="retrieve")
#     def retrieve(query, top_k=None, config=CONFIG) -> list[dict]:
#         points = retrieve_points(query, top_k=top_k, config=config)
#         return [{"page_content": p.payload["text"], "metadata": {...}} for p in points]
#
# The span output MUST be a list of dicts with the text under "page_content"
# (or "content" / "text"); optional "metadata", of which only metadata.doc_uri
# is read. Anything else is dropped with a debug-level log. Only *top-level*
# retriever spans count — one nested inside another retriever span is ignored.
# ---------------------------------------------------------------------------
