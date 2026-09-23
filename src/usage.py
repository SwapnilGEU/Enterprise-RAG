"""Token usage and throughput, read off the responses Ollama already returns.

Every ChatOllama reply carries its own accounting, so nothing here costs an
extra call:

    usage_metadata     input_tokens / output_tokens / total_tokens
    response_metadata  prompt_eval_count / eval_count, and eval_duration (ns)

Two throughput numbers, because they answer different questions:

    tokens_per_sec        completion tokens / time spent decoding them.
                          The model's generation speed — what a GPU or model
                          change moves. Uses Ollama's own eval_duration, so
                          queueing, retrieval and prompt prefill are excluded.
    total_tokens_per_sec  (prompt + completion) / end-to-end latency.
                          What the whole request delivered per wall-clock
                          second, retrieval included.

Pure functions, no imports from the rest of src/, so any layer can use them.
"""

from typing import Iterable


def _usage_of(message) -> tuple[int, int, int] | None:
    """(prompt_tokens, completion_tokens, eval_duration_ns) for one reply, or
    None when the object is not an LLM reply (a HumanMessage, a ToolMessage)."""
    usage = getattr(message, "usage_metadata", None) or {}
    meta = getattr(message, "response_metadata", None) or {}

    prompt = usage.get("input_tokens", meta.get("prompt_eval_count"))
    completion = usage.get("output_tokens", meta.get("eval_count"))
    if prompt is None and completion is None:
        return None
    return int(prompt or 0), int(completion or 0), int(meta.get("eval_duration") or 0)


def llm_usage(messages: Iterable) -> dict:
    """Sum token usage over every LLM reply in `messages`.

    Pass one response as `[response]`. For the agent, pass the whole message
    list: a request can take several model turns (routing, final answer) and
    each one is paid for.
    """
    prompt = completion = eval_ns = calls = 0
    for message in messages:
        found = _usage_of(message)
        if found is None:
            continue
        p, c, ns = found
        prompt += p
        completion += c
        eval_ns += ns
        calls += 1

    generation_s = eval_ns / 1e9
    return {
        "llm_calls": calls,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "tokens_per_sec": round(completion / generation_s, 2) if generation_s > 0 else None,
    }


def total_tokens_per_sec(total_tokens: int, total_latency_s: float) -> float | None:
    """Whole-request throughput. None rather than a division by zero, or a
    misleading 0.0 when no tokens were counted."""
    if not total_tokens or not total_latency_s or total_latency_s <= 0:
        return None
    return round(total_tokens / total_latency_s, 2)
