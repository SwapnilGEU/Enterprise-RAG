"""Agent entry point: ask the tool-using agent one question.

    python scripts/run_agent.py "What is 15.5 * 42?"
    python scripts/run_agent.py "How many queries have failed so far?"

Needs Qdrant, Ollama AND Postgres — the agent logs every turn to query_history.
"""

import _bootstrap  # noqa: F401

import argparse

from langchain_core.messages import AIMessage

from src.agent import ask_agent
from src.history import ensure_schema


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the agent a question")
    parser.add_argument("question")
    parser.add_argument("--session-id", default="cli")
    parser.add_argument("--show-tools", action="store_true", help="print which tools were called")
    args = parser.parse_args()

    ensure_schema()
    state = ask_agent(args.question, session_id=args.session_id)

    if args.show_tools:
        used = [tc["name"] for m in state["messages"]
                if isinstance(m, AIMessage) and m.tool_calls
                for tc in m.tool_calls]
        print(f"\nTOOLS USED: {', '.join(dict.fromkeys(used)) or 'none (answered directly)'}")

    print("\nANSWER:")
    print(state["messages"][-1].content)


if __name__ == "__main__":
    main()
