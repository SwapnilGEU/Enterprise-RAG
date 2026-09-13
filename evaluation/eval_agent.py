"""Agent tool-routing evaluation — notebook Section 22, as a script.

Measures routing accuracy: did the agent pick the right tool? That's the
failure mode that matters most in a multi-tool agent — a perfect RAG answer to
a question that should have gone to SQL is still wrong.

    python evaluation/eval_agent.py
    python evaluation/eval_agent.py --limit 3        # quick smoke run
    python evaluation/eval_agent.py --preview 1200   # show more of each answer

Needs Qdrant, Ollama and Postgres. Budget a few minutes: it is one agent run
plus one judge call per case, on a local model.
"""

import _common  # noqa: F401  (must come first — sets sys.path)

import argparse
import json
import time

import pandas as pd
from langchain_core.messages import HumanMessage

from _common import DATASETS, RESULTS, judge, message_text, show
from src.agent import build_agent
from src.generation import get_llm
from src.history import ensure_schema


def run_case(app, case: dict, preview: int) -> dict:
    query, expected_tool = case["query"], case["expected_tool"]
    started = time.time()

    try:
        state = app.invoke({
            "session_id": "tool_eval_session",
            "user_query": query,
            "start_time": started,
            "messages": [HumanMessage(content=query)],
        })
        agent_error = None
    except Exception as exc:
        agent_error, state = f"{type(exc).__name__}: {exc}", None

    elapsed = time.time() - started

    if state is None:
        print(f"  AGENT CRASHED after {elapsed:.1f}s")
        show("error:", agent_error)
        return {
            "query": query, "expected_tool": expected_tool, "actual_tools": "ERROR",
            "tool_correct": False, "reference_answer": case["reference_answer"],
            "actual_answer": agent_error, "tool_outputs": "", "judge_verdict": "",
            "answer_correct": False, "seconds": round(elapsed, 1),
        }

    # Which tools did the agent actually reach for? dict.fromkeys dedupes while
    # preserving call order, which set() would scramble.
    actual_tools_used = [
        tc["name"] for msg in state["messages"]
        if getattr(msg, "tool_calls", None) for tc in msg.tool_calls
    ]
    actual_tool_str = ", ".join(dict.fromkeys(actual_tools_used)) if actual_tools_used else "direct_llm"

    if expected_tool == "direct_llm":
        # "answered without tools" — not "called a tool literally named direct_llm"
        tool_matched = not actual_tools_used
    else:
        tool_matched = expected_tool in actual_tools_used

    final_answer = message_text(state["messages"][-1].content)

    # What the tools returned — the usual reason a correct route still gives a
    # wrong answer (a SQL error string, an empty retrieval).
    tool_outputs = [
        f"{getattr(m, 'name', '?')} -> {message_text(m.content)}"
        for m in state["messages"] if getattr(m, "type", None) == "tool"
    ]

    verdict, is_correct = judge(get_llm(), query, case["reference_answer"], final_answer)

    print(f"  tool:   expected {expected_tool!r} | got {actual_tool_str!r}   {'PASS' if tool_matched else 'FAIL'}")
    print(f"  time:   {elapsed:.1f}s")
    for output in tool_outputs:
        show("tool returned:", output, limit=preview)
    show("expected:", case["reference_answer"])
    show("ANSWER:", final_answer, limit=preview)
    print(f"  judge:  {verdict[:80]!r} -> {'CORRECT' if is_correct else 'INCORRECT'}")

    return {
        "query": query, "expected_tool": expected_tool, "actual_tools": actual_tool_str,
        "tool_correct": tool_matched, "reference_answer": case["reference_answer"],
        "actual_answer": final_answer, "tool_outputs": " || ".join(tool_outputs),
        "judge_verdict": verdict, "answer_correct": is_correct, "seconds": round(elapsed, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate agent tool routing")
    parser.add_argument("--limit", type=int, default=None, help="run only the first N cases")
    parser.add_argument("--preview", type=int, default=600, help="chars of each answer to show inline")
    parser.add_argument("--dataset", default=str(DATASETS / "tool_routing.json"))
    args = parser.parse_args()

    dataset = json.loads(open(args.dataset, encoding="utf-8").read())
    if args.limit:
        dataset = dataset[:args.limit]

    ensure_schema()
    app = build_agent()

    print("=" * 100)
    print(f"AGENT EVALUATION — {len(dataset)} cases")
    print("=" * 100)

    results = []
    for number, case in enumerate(dataset, start=1):
        print(f"\n[{number}/{len(dataset)}] {case['query']}")
        print("-" * 100)
        results.append(run_case(app, case, args.preview))

    df = pd.DataFrame(results)
    total = len(df)
    tool_accuracy = df["tool_correct"].sum() / total
    answer_accuracy = df["answer_correct"].sum() / total

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Tool-selection accuracy: {df['tool_correct'].sum()}/{total}  ({tool_accuracy * 100:.0f}%)")
    print(f"Answer correctness:      {df['answer_correct'].sum()}/{total}  ({answer_accuracy * 100:.0f}%)")
    print(f"Total time:              {df['seconds'].sum():.0f}s\n")
    print(df[["query", "expected_tool", "actual_tools", "tool_correct", "answer_correct", "seconds"]].to_string(index=False))

    failures = df[~df["tool_correct"] | ~df["answer_correct"]]
    if not failures.empty:
        print("\n" + "=" * 100)
        print(f"FAILURES ({len(failures)})")
        print("=" * 100)
        for _, row in failures.iterrows():
            reasons = []
            if not row["tool_correct"]:
                reasons.append(f"wrong tool (expected {row['expected_tool']}, got {row['actual_tools']})")
            if not row["answer_correct"]:
                reasons.append("judge said incorrect")
            print(f"\n{row['query']}")
            print(f"  why: {'; '.join(reasons)}")
            show("expected:", str(row["reference_answer"]))
            show("ANSWER:", str(row["actual_answer"]))   # full text, no truncation
    else:
        print("\nAll cases passed.")

    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "agent_evaluation_report.csv", index=False)
    summary = {
        "total_cases": total,
        "tool_selection_accuracy": f"{tool_accuracy * 100:.0f}%",
        "answer_correctness": f"{answer_accuracy * 100:.0f}%",
        "cases": results,
    }
    (RESULTS / "agent_evaluation_summary.json").write_text(
        json.dumps(summary, indent=4, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nReports saved to {RESULTS}/ (full answers are in both files).")

    # ---- MLflow goes here -------------------------------------------------
    # mlflow.set_experiment("agent_tool_routing")
    # with mlflow.start_run():
    #     mlflow.log_params({"model": CONFIG.ollama_model, "num_ctx": CONFIG.ollama_num_ctx})
    #     mlflow.log_metrics({"tool_accuracy": tool_accuracy, "answer_accuracy": answer_accuracy})
    #     mlflow.log_artifact(RESULTS / "agent_evaluation_report.csv")


if __name__ == "__main__":
    main()
