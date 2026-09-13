"""Query-time entry point: ask the RAG pipeline one question.

    python scripts/run_query.py "What is hybrid retrieval?"
    python scripts/run_query.py "..." --top-k 8 --show-context

Needs Qdrant (indexed) and Ollama running. Does not touch Postgres or the agent.
"""

import _bootstrap  # noqa: F401

import argparse

from src.generation import generate_answer, print_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the knowledge base a question")
    parser.add_argument("question", help="the question to answer")
    parser.add_argument("--top-k", type=int, default=None, help="chunks to retrieve")
    parser.add_argument("--show-context", action="store_true",
                        help="also print the exact context sent to the model")
    args = parser.parse_args()

    result = generate_answer(args.question, top_k=args.top_k)
    print()
    print_result(result, show_context=args.show_context)


if __name__ == "__main__":
    main()
