"""Shared helpers for both evaluation scripts."""

import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASETS = Path(__file__).resolve().parent / "datasets"
RESULTS = Path(__file__).resolve().parent / "results"


def message_text(content) -> str:
    """LangChain message content is usually a str but can be a list of content
    blocks. Flatten either shape to plain text."""
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
        return " ".join(p for p in parts if p).strip()
    return str(content).strip()


def show(label: str, text: str, limit: int | None = None) -> None:
    """Print a labelled block, wrapped and indented so long answers stay readable."""
    body = text if limit is None or len(text) <= limit else text[:limit].rstrip() + f"… [+{len(text) - limit} chars]"
    print(f"  {label}")
    for line in body.splitlines() or [""]:
        for wrapped in (textwrap.wrap(line, width=96) or [""]):
            print(f"      {wrapped}")


def judge(llm, question: str, reference: str, actual: str) -> tuple[str, bool]:
    """LLM-as-a-judge. Returns (raw verdict, is_correct).

    Caveat worth remembering: when the judge is the same model that produced the
    answer, you are measuring the judge's ceiling, not the system's quality.
    Point OLLAMA_MODEL at a larger model for judging, or treat this as a smoke
    test rather than a metric.
    """
    prompt = (
        f"Evaluate if the actual answer satisfies the expected answer's intent.\n"
        f"Question: {question}\n"
        f"Expected Intent: {reference}\n"
        f"Actual Answer: {actual}\n"
        f"Is the Actual Answer correct? Answer strictly 'Yes' or 'No'."
    )
    verdict = message_text(llm.invoke(prompt).content)
    low = verdict.strip().lower()
    return verdict, low.startswith("yes") or "yes" in low[:20]
