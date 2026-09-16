"""Agent evaluation: DeepEval's four agent metrics on MLflow.

    mlflow server                         # in one terminal, from the repo root
    python mlflow/agentflow.py --smoke    # preflight + one case — always do this first
    python mlflow/agentflow.py            # the whole tool_routing.json set

Sibling of `ragflow.py`, same shape, same judge plumbing (`judge_json.py`,
`judge_quota.py`), different question. `ragflow.py` asks whether the *answer*
was any good; this asks whether the agent *behaved* well — did it pick the right
tool, call it with sane arguments, take a sensible number of steps, and actually
finish the job.

| scorer | judged? | needs on the test case |
|---|---|---|
| `TaskCompletion`      | LLM | input + actual_output |
| `ToolCorrectness`     | **no — deterministic** | input + tools_called + expected_tools |
| `ArgumentCorrectness` | LLM | input + tools_called |
| `StepEfficiency`      | LLM | input + actual_output |

`ToolCorrectness` is pure comparison — DeepEval's metric has no
`evaluation_model` at all — so routing accuracy costs nothing and cannot be
rate limited. `--no-judge` runs that one alone, which is the cheap check to
reach for when the judge quota is gone (see mlflow/judge_quota.py).

Tool calls come ONLY from TOOL spans
------------------------------------
This is the agent-side twin of ragflow's retriever-span trap, and it is worse,
because this repo has no tracing at all in `src/agent.py` and never calls
`mlflow.langchain.autolog()`. DeepEval's `tools_called` is built solely by
`_extract_tool_calls_from_trace`, which reads `trace.search_spans(TOOL)` and
returns **None** when there are none — and `ToolCorrectness` and
`ArgumentCorrectness` *raise* on a None. Uninstrumented, every row errors.

So `instrument_tools()` below wraps each tool from `src.agent.build_tools` in a
TOOL span before the graph is compiled. Verified end to end: wrapping
`StructuredTool.func` survives LangChain's own `invoke` path and DeepEval then
sees each call's name, arguments and output.

Requires Qdrant, Ollama and Postgres — `build_agent()` builds the SQL toolkit at
compile time, so the graph will not compile without Postgres.
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

for env_path in (PROJECT_ROOT / ".env", HERE / ".env"):
    if env_path.exists():
        load_dotenv(env_path)

# Same reasoning as ragflow.py: one Ollama model resident on a 6GB card, so
# parallel predict_fn workers thrash. Must precede `import mlflow`.
os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_WORKERS", "1")
os.environ.setdefault("OLLAMA_MAX_LOADED_MODELS", "1")

import mlflow  # noqa: E402
import mlflow.genai  # noqa: E402
from mlflow.entities import SpanType  # noqa: E402

try:
    from mlflow.genai.scorers.deepeval import (  # noqa: E402
        ArgumentCorrectness,
        StepEfficiency,
        TaskCompletion,
        ToolCorrectness,
    )
except ImportError as exc:  # deepeval missing, or mlflow too old
    raise SystemExit(
        f"Could not import MLflow's DeepEval agent scorers ({exc}).\n"
        "    pip install -r mlflow/requirements-eval.txt"
    ) from exc

sys.path.insert(0, str(HERE))
from judge_json import install as install_judge_patches  # noqa: E402
import judge_quota  # noqa: E402

JUDGE = os.environ.get("GEMINI_JUDGE", "gemini:/gemini-2.5-flash")
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "enterprise-agent-eval")
DEFAULT_DATA = PROJECT_ROOT / "evaluation" / "datasets" / "tool_routing.json"

# The dataset's name for "should answer without calling anything".
DIRECT = "direct_llm"


# --- making the agent's tools visible to DeepEval ---------------------------


def instrument_tools() -> None:
    """Give every agent tool a TOOL span, by patching `src.agent.build_tools`.

    Must run before `build_agent()`, which caches the compiled graph in a module
    singleton — instrument afterwards and the graph still holds the bare tools.

    Wrapping `StructuredTool.func` rather than re-decorating the tool keeps the
    name, description and args schema exactly as the LLM sees them, so the
    agent's routing behaviour is unchanged by being measured. The alternative,
    `mlflow.langchain.autolog()`, would also emit spans, but whether it marks
    them `span_type=TOOL` belongs to the integration rather than to this repo,
    and a silent change there would take every tool metric down with it.
    """
    import src.agent as agent_module

    original_build_tools = agent_module.build_tools
    if getattr(original_build_tools, "_mlflow_instrumented", False):
        return

    def build_instrumented_tools(*args, **kwargs):
        tools = original_build_tools(*args, **kwargs)
        for t in tools:
            inner = getattr(t, "func", None)
            if inner is None or getattr(t, "_mlflow_traced", False):
                continue
            name = t.name

            def make(inner, name):
                def traced(*a, **kw):
                    with mlflow.start_span(name=name, span_type=SpanType.TOOL) as span:
                        span.set_inputs(kw if kw else {"args": list(a)})
                        out = inner(*a, **kw)
                        span.set_outputs(out)
                        return out

                return traced

            t.func = make(inner, name)
            try:
                object.__setattr__(t, "_mlflow_traced", True)
            except Exception:  # noqa: BLE001 — pydantic model, best effort
                pass
        return tools

    build_instrumented_tools._mlflow_instrumented = True
    agent_module.build_tools = build_instrumented_tools
    # A graph compiled earlier in this process would still hold bare tools.
    agent_module._agent = None


def patch_empty_tools_called() -> bool:
    """Make "the agent called no tools" survive as `[]` rather than `None`.

    `_extract_tool_calls_from_trace` returns None when a trace has no TOOL
    spans, and `ToolCorrectness` / `ArgumentCorrectness` raise
    `MissingTestCaseParamsError` on a None. That is right for an uninstrumented
    trace and wrong for the `direct_llm` cases, where calling nothing *is* the
    expected behaviour — without this they error instead of scoring.

    This is only safe because `check_agent_trace()` proves, on a case that must
    route, that TOOL spans do appear. With that established, an empty list means
    "really called nothing" rather than "instrumentation is broken" — which is
    the distinction that makes an empty list reportable instead of a lie.
    """
    try:
        from mlflow.genai.scorers.deepeval import utils as de_utils
    except ImportError:
        return False

    original = de_utils._extract_tool_calls_from_trace
    if getattr(original, "_empty_patched", False):
        return True

    def extract(trace):
        return original(trace) or []

    extract._empty_patched = True
    de_utils._extract_tool_calls_from_trace = extract
    return True


# --- the app ---------------------------------------------------------------


@mlflow.trace(span_type=SpanType.AGENT, name="agent")
def agent(question: str) -> str:
    """The thing under evaluation.

    Returns the final answer as a plain string, for the same reason
    `ragflow.rag` does: MLflow stringifies whatever comes back into
    `actual_output`, and handing over the whole state dict would bury the answer
    under the message history and every tool's raw return value — which
    `TaskCompletion` and `StepEfficiency` both read.
    """
    import time

    from langchain_core.messages import HumanMessage

    from src.agent import build_agent

    state = build_agent().invoke(
        {
            "session_id": "agentflow_eval",
            "user_query": question,
            "start_time": time.time(),
            "messages": [HumanMessage(content=question)],
        }
    )

    final = state["messages"][-1].content
    return final if isinstance(final, str) else _flatten(final)


def _flatten(content) -> str:
    """LangChain message content is usually a str but can be a list of blocks."""
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
        return " ".join(p for p in parts if p).strip()
    return str(content).strip()


# --- scorers ----------------------------------------------------------------


def build_scorers(judge: str | None) -> list:
    """The four agent metrics.

    `ToolCorrectness` is deterministic — DeepEval's metric has no
    `evaluation_model` — so it is the only one that survives `--no-judge`, and
    it happens to be the one measuring the failure that matters most in a
    multi-tool agent.

    Every judged scorer needs an explicit `model=`: MLflow's default judge is
    OpenAI `gpt-4.1-mini`, and a scorer built without one reaches for
    `OPENAI_API_KEY`.
    """
    scorers = [ToolCorrectness()]
    if judge:
        scorers = [
            TaskCompletion(model=judge),
            ToolCorrectness(),
            ArgumentCorrectness(model=judge),
            StepEfficiency(model=judge),
        ]
    return scorers


# --- the data --------------------------------------------------------------


def load_dataset(path: Path, limit: int | None = None) -> list[dict]:
    """tool_routing.json is `{query, expected_tool, reference_answer}`.

    Neither expectation key is a free choice. `expected_tool_calls` is the only
    key MLflow turns into DeepEval's `expected_tools` (as a *list of dicts*,
    each with a "name"), and `expected_output` is the only one it maps to
    `expected_output`. Anything else lands in `context` and is ignored by these
    four metrics.

    `direct_llm` becomes an empty list rather than a tool named "direct_llm" —
    DeepEval scores "expected nothing, called nothing" as 1.0 and "expected
    nothing, called something" as 0.0, which is exactly the intent.
    """
    if not path.exists():
        raise SystemExit(
            f"Agent dataset not found: {path}\n"
            'Expected a JSON list of {"query": ..., "expected_tool": ..., '
            '"reference_answer": ...}\n'
            "or point at another file with --data."
        )

    rows = json.loads(path.read_text("utf-8"))
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"{path} must be a non-empty JSON list.")

    dataset = []
    for i, row in enumerate(rows):
        try:
            expected_tool = row["expected_tool"]
            expected_calls = [] if expected_tool == DIRECT else [{"name": expected_tool}]
            dataset.append(
                {
                    "inputs": {"question": row["query"]},
                    "expectations": {
                        "expected_tool_calls": expected_calls,
                        "expected_output": row["reference_answer"],
                    },
                }
            )
        except KeyError as exc:
            raise SystemExit(
                f"{path} row {i} is missing {exc}; each row needs "
                '"query", "expected_tool" and "reference_answer".'
            ) from exc

    return dataset[:limit] if limit else dataset


# --- preflight -------------------------------------------------------------


def check_judge(judge: str) -> bool:
    """One judge call, before anything expensive. Skipped under --no-judge."""
    print(f"[preflight] asking the judge one question ({judge}) ...")
    ok, message = judge_quota.probe(judge)
    print(f"[preflight] {'OK — ' if ok else 'FAIL — '}{message}")
    return ok


def check_agent_trace(question: str) -> bool:
    """Run one real case and confirm TOOL spans reach DeepEval.

    The agent-side equivalent of ragflow's retriever-span check. Without spans,
    `ToolCorrectness` and `ArgumentCorrectness` raise on every row — and thanks
    to `DeepEvalScorer.__call__` swallowing exceptions into `Feedback(error=e)`,
    that surfaces as the same uninformative "N/N failed" as a bad judge or an
    exhausted quota. Ten seconds here saves that entire hunt.

    Pick a `--data` whose first case must call a tool: a question that correctly
    routes to no tool proves nothing about instrumentation.
    """
    print(f"[preflight] running one case through the agent: {question!r}")
    with mlflow.start_span(name="preflight") as span:
        answer = agent(question)
        trace_id = span.trace_id

    mlflow.flush_trace_async_logging()
    trace = mlflow.get_trace(trace_id)
    if trace is None:
        print("[preflight] WARNING: could not read the trace back from the server.")
        return False

    from mlflow.genai.scorers.deepeval.utils import _extract_tool_calls_from_trace

    calls = _extract_tool_calls_from_trace(trace) or []

    print(f"[preflight] answer: {answer[:120]!r}...")
    print(f"[preflight] TOOL spans DeepEval can see: {len(calls)}")
    for call in calls:
        args = call.input_parameters
        print(f"             - {call.name}({args if args else ''})")

    if not calls:
        print(
            "[preflight] FAIL — no TOOL spans on the trace.\n"
            "  ToolCorrectness and ArgumentCorrectness would raise on every row,\n"
            "  and the harness would report it as an uninformative scorer failure.\n"
            "  Either instrument_tools() did not take effect (was build_agent()\n"
            "  already compiled earlier in this process?), or this question\n"
            "  genuinely routed to no tool — try one that must call one."
        )
        return False

    print("[preflight] OK — tool calls are visible to the scorers.")
    return True


# --- reporting -------------------------------------------------------------


def print_routing_breakdown(rows: list[dict]) -> None:
    """Per-tool case counts, so a mediocre score says *which* route is broken.

    An aggregate is nearly useless here: 7/10 with every miss on `sql_tool` is a
    prompt problem fixable in one place, while 7/10 scattered across four tools
    is a model-capability problem. Same number, opposite response.
    """
    counts = Counter(
        (row["expectations"]["expected_tool_calls"] or [{"name": DIRECT}])[0]["name"]
        for row in rows
    )
    print("\ncases per expected tool:")
    for name, count in sorted(counts.items()):
        print(f"  {name:<14} {count}")


# --- run it ----------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="tool-routing JSON")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N cases")
    parser.add_argument("--smoke", action="store_true", help="preflights + one case only")
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="run only ToolCorrectness (deterministic) — no judge calls, no quota",
    )
    parser.add_argument("--judge", default=JUDGE, help=f"judge model URI (default {JUDGE})")
    parser.add_argument(
        "--skip-preflight", action="store_true", help="skip the TOOL-span visibility check"
    )
    parser.add_argument("--skip-judge-preflight", action="store_true")
    parser.add_argument(
        "--stock-judge",
        action="store_true",
        help="skip the JSON transport patches (see mlflow/judge_json.py)",
    )
    args = parser.parse_args()

    judge_uri = None if args.no_judge else args.judge

    if args.stock_judge:
        os.environ["JUDGE_NATIVE_JSON"] = "0"
        os.environ["JUDGE_JSON_REPAIR"] = "0"
    if judge_uri:
        print(f"judge JSON patches: {install_judge_patches()}")
        judge_quota.reset()
        print(f"judge quota guard: {judge_quota.install()}")

    # Both must happen before the graph is compiled or a scorer runs.
    instrument_tools()
    print(f"tool spans: instrumented   empty-tools patch: {patch_empty_tools_called()}")

    mlflow.set_tracking_uri(TRACKING_URI)
    try:
        mlflow.set_experiment(EXPERIMENT)
    except Exception as exc:
        raise SystemExit(
            f"Could not reach the MLflow tracking server at {TRACKING_URI} ({exc}).\n"
            "Start it first:  mlflow server"
        ) from exc

    mode = f"judge: {judge_uri}" if judge_uri else "judge: none (ToolCorrectness only)"
    print(f"tracking: {TRACKING_URI}   experiment: {EXPERIMENT}   {mode}")

    limit = 1 if args.smoke else args.limit
    dataset = load_dataset(args.data, limit=limit)
    print(f"dataset: {args.data.name} — {len(dataset)} case(s)")
    print_routing_breakdown(dataset)

    if judge_uri and not args.skip_judge_preflight:
        if not check_judge(judge_uri):
            print("\nAborting before the agent runs: the judge cannot answer.")
            print("ToolCorrectness needs no judge — re-run with --no-judge to score routing.")
            return 2

    if not args.skip_preflight:
        ok = check_agent_trace(dataset[0]["inputs"]["question"])
        if not ok and not args.smoke:
            print("Aborting: fix tool visibility, or re-run with --skip-preflight.")
            return 1

    results = mlflow.genai.evaluate(
        data=dataset,
        predict_fn=agent,
        scorers=build_scorers(judge_uri),
    )

    print("\nmetrics:")
    for name, value in sorted(results.metrics.items()):
        print(f"  {name}: {value}")

    if judge_uri:
        counts = judge_quota.stats()
        print(
            f"\njudge calls: {counts['calls']}"
            f"   retries: {counts['retries']}   failures: {counts['failures']}"
        )
        if counts["quota_exhausted"]:
            print(
                "\n"
                + "=" * 70
                + f"\nJUDGED SCORES INVALID — judge quota ran out: {counts['quota_exhausted']}\n"
                "ToolCorrectness above is unaffected: it never touches a judge.\n\n"
                + judge_quota.quota_advice(args.judge, judge_quota.tripped())
                + "\n"
                + "=" * 70
            )
            return 3

        errored = any("error" in name.lower() for name in results.metrics)
        if errored:
            print(
                "\nNOTE: some rows errored. The quota was fine, so the likely causes are\n"
                "an unparseable judge reply, or a metric that needed tool calls on a case\n"
                "that made none. Get the real message with:\n"
                "    python mlflow/diagnose.py --full"
            )

    print(f"\nOpen {TRACKING_URI} to see per-case scores and the traces behind them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
