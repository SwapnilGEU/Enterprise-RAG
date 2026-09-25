"""Prompting, the local LLM, retries, and the end-to-end answer.

Notebook Sections 14-17 in one module: context building, the Ollama client,
the generic retry wrapper, and generate_answer() which ties them together.
This is the module a FastAPI layer would import.
"""

import time
from functools import partial

from src.config import CONFIG, Config, logger
from src.payload import format_source
from src.retrieval import RetrievedDoc, retrieve
from src.usage import llm_usage, total_tokens_per_sec

# Section 16 now lives in src/retry.py so retrieval.py can share it (the
# RETRIEVER span needs the retry to happen inside the span). Re-exported here
# so `from src.generation import call_with_retry` keeps working.
from src.retry import (  # noqa: F401
    ValidationFailed,
    call_with_retry,
    validate_llm_response,
    validate_retrieval,
)


# --------------------------------------------------------------------------
# Local model (Section 15)
# --------------------------------------------------------------------------

_llm = None


def get_llm(config: Config = CONFIG):
    """ChatOllama, created lazily, one per process.

    A module-level singleton rather than @lru_cache — Config is a mutable
    dataclass and unhashable, so lru_cache cannot key on it.

    `num_ctx` matters: Ollama defaults it to 4096 regardless of what the model
    supports (qwen3:4b handles 32k), and `num_predict` comes OUT of that budget
    rather than on top of it. Five retrieved chunks plus an agent loop overflow
    4096 and Ollama returns a 400 'exceeds the available context size'.
    """
    global _llm
    if _llm is not None:
        return _llm

    from langchain_ollama import ChatOllama

    _llm = ChatOllama(
        model=config.ollama_model,
        base_url=config.ollama_base_url,
        temperature=config.ollama_temperature,
        num_predict=config.ollama_num_predict,
        num_ctx=config.ollama_num_ctx,
        reasoning=False,
        # Stream plain answers, but never a call that has tools bound. When a
        # tool-calling turn streams, Ollama has to recognise the tool call in
        # a token stream, and small models (llama3.2:3b especially) then leak
        # it as text — "I'll call get_weather {...}" — instead of calling it.
        # Non-streamed, a tool turn behaves exactly as it did before 2026-09-25.
        disable_streaming="tool_calling",
    )
    logger.info(f"Ollama model: {config.ollama_model} (num_ctx={config.ollama_num_ctx})")
    return _llm


# --------------------------------------------------------------------------
# Context and prompt (Section 14)
# --------------------------------------------------------------------------

def build_context(docs) -> str:
    """Per-chunk context block. Content first, source tag last — mirrors the
    "cite the source at the end" instruction and reads like a real citation
    rather than a label stapled to the front of every passage."""
    return "\n\n".join(f"{doc.page_content}\n{format_source(doc.metadata)}" for doc in docs)


def build_prompt(context: str, question: str) -> str:
    """The prompt, rewritten 2026-09-17 to stop the model narrating its own
    reasoning about the context.

    The previous version said "give a clear and sufficiently detailed answer,
    use multiple sentences", and buried the refusal line at the bottom. A 4b
    model reads that as an instruction to write several sentences no matter
    what, so asking it "hello" produced a paragraph explaining which topics the
    context covered and why none of them applied, and only then the refusal.

    Two changes fix that: the refusal is now an exact string with "and nothing
    else" attached, and there is an explicit ban on describing the context.
    Length guidance is now "as long as it needs to be" rather than a floor —
    a floor is what produced the padding.
    """
    return f"""<|system|>
You are a question-answering assistant for machine learning topics.

Answer the question using ONLY the supplied context.

If the context does not contain the answer, reply with exactly:
I don't know based on the provided context.
and nothing else. No explanation, no apology, no description of what the
context does or does not cover.

Never describe the context itself. Do not write sentences like "the context
includes...", "the provided context does not define...", or "this query does
not require...". Answer the question, or give the refusal line above.

Be direct. Use as many sentences as the answer genuinely needs and no more.
Do not restate the question before answering it.

Do not add information that is not supported by the context.

Cite the relevant source tag at the end of the answer when you used it.

<|user|>
Context:
{context}

Question:
{question}

<|assistant|>
"""


# --------------------------------------------------------------------------
# End-to-end answer (Section 17)
# --------------------------------------------------------------------------

def _metrics(started: float, retrieval_s: float | None = None,
             llm_s: float | None = None, response=None) -> dict:
    """Per-request timing and token accounting, attached to every result —
    degraded ones included, since a slow failure is worth seeing too.

    Latencies are in ms. llm_latency_ms is wall-clock around the call, so it
    includes retries and prompt prefill; tokens_per_sec is Ollama's own decode
    speed and excludes both. The gap between them is informative."""
    total_s = time.perf_counter() - started
    usage = llm_usage([response] if response is not None else [])
    return {
        "retrieval_latency_ms": round(retrieval_s * 1000, 1) if retrieval_s is not None else None,
        "llm_latency_ms": round(llm_s * 1000, 1) if llm_s is not None else None,
        "total_latency_ms": round(total_s * 1000, 1),
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "tokens_per_sec": usage["tokens_per_sec"],
        "total_tokens_per_sec": total_tokens_per_sec(usage["total_tokens"], total_s),
    }


def retrieve_with_retry(query: str, top_k: int | None = None, config: Config = CONFIG):
    """Kept for backwards compatibility. retrieve() now retries internally —
    inside its MLflow RETRIEVER span — so this is a plain pass-through."""
    return retrieve(query, top_k=top_k, config=config)


def generate_answer(question: str, top_k: int | None = None, config: Config = CONFIG) -> dict:
    """End-to-end RAG: retrieve -> build context -> build prompt -> generate.

    Returns {answer, sources, context, degraded, metrics, failure_stage?}.
    `sources` are the raw Qdrant payloads, so they carry section_heading and
    page_label. `metrics` is latency and token usage — see _metrics().
    """
    started = time.perf_counter()
    try:
        points = retrieve(question, top_k=top_k, config=config)
    except Exception as exc:
        logger.error(f"generate_answer: retrieval failed permanently -- {exc!r}")
        return {
            "answer": "I wasn't able to search the knowledge base right now. Please try again shortly.",
            "sources": [],
            "context": "",
            "degraded": True,
            "failure_stage": "retrieval",
            "metrics": _metrics(started, retrieval_s=time.perf_counter() - started),
        }
    retrieval_s = time.perf_counter() - started

    docs = [RetrievedDoc(p) for p in points]
    context = build_context(docs)
    prompt = build_prompt(context, question)

    llm_started = time.perf_counter()
    try:
        response = call_with_retry(
            partial(get_llm(config).invoke),
            prompt,
            config=config,
            validate=validate_llm_response,
            call_name="llm_invoke",
        )
    except Exception as exc:
        logger.error(f"generate_answer: generation failed permanently -- {exc!r}")
        return {
            "answer": "I found relevant information but couldn't generate a response right now. Please try again.",
            "sources": [d.metadata for d in docs],
            "context": context,
            "degraded": True,
            "failure_stage": "generation",
            "metrics": _metrics(started, retrieval_s, time.perf_counter() - llm_started),
        }
    llm_s = time.perf_counter() - llm_started

    return {
        "answer": response.content,
        "sources": [d.metadata for d in docs],
        "context": context,
        "degraded": False,
        "metrics": _metrics(started, retrieval_s, llm_s, response),
    }


def print_result(result: dict, show_context: bool = False) -> None:
    print("ANSWER:")
    print(result["answer"])

    if result["degraded"]:
        print(f"\nDEGRADED: True  (failed at: {result.get('failure_stage')})")
    else:
        print("\nDEGRADED: False")

    print("\nSOURCES:")
    if not result["sources"]:
        print("  (none)")
    for meta in result["sources"]:
        print(f"  - {meta.get('document')}")
        print(f"      section: {meta.get('section_heading') or '(no heading — no sections detected)'}")
        print(f"      page:    {meta.get('page_label') or '(n/a)'}")
        print(f"      chunk:   {meta.get('chunk_id')}")

    if show_context:
        print("\nCONTEXT SENT TO THE MODEL:")
        print(result["context"])


# --------------------------------------------------------------------------
# Streaming answer (added 2026-09-25)
# --------------------------------------------------------------------------
# Why this exists: generate_answer() calls `llm.invoke()`, which returns only
# once the LAST token is decoded. Ollama was producing tokens the whole time,
# but nothing reached the UI until the whole answer was done, so a 15s answer
# looked like a 15s freeze.
#
# stream_answer() is the same pipeline, as a generator of events:
#
#   {"type": "status", "stage": ..., "message": ...}   what is happening now
#   {"type": "token",  "text": ...}                    a piece of the answer
#   {"type": "done",   "answer", "sources", "degraded", "failure_stage", "metrics"}
#
# generate_answer() is left exactly as it was — evaluation, the agent's
# rag_tool and /query all still use it.
#
# The one real trade-off: retries. Once the first token has been sent it
# cannot be taken back, so a failure MID-stream ends the answer as degraded
# instead of retrying. A failure BEFORE the first token still gets the normal
# call_with_retry() treatment via a non-streaming fallback.


def _status(stage: str, message: str) -> dict:
    return {"type": "status", "stage": stage, "message": message}


def stream_answer(question: str, top_k: int | None = None, config: Config = CONFIG):
    """Generator version of generate_answer(). See the block comment above."""
    started = time.perf_counter()

    yield _status("retrieving", "🔎 Searching the knowledge base…")
    try:
        points = retrieve(question, top_k=top_k, config=config)
    except Exception as exc:
        logger.error(f"stream_answer: retrieval failed permanently -- {exc!r}")
        answer = "I wasn't able to search the knowledge base right now. Please try again shortly."
        yield {"type": "token", "text": answer}
        yield {
            "type": "done", "answer": answer, "sources": [], "degraded": True,
            "failure_stage": "retrieval",
            "metrics": _metrics(started, retrieval_s=time.perf_counter() - started),
        }
        return
    retrieval_s = time.perf_counter() - started

    docs = [RetrievedDoc(p) for p in points]
    sources = [d.metadata for d in docs]
    documents = list(dict.fromkeys(str(m.get("document", "")) for m in sources if m.get("document")))
    found = f"📚 Found {len(docs)} passage(s)"
    if documents:
        found += " from " + ", ".join(documents[:3]) + (f" +{len(documents) - 3} more" if len(documents) > 3 else "")
    yield _status("retrieved", found)

    prompt = build_prompt(build_context(docs), question)
    yield _status("generating", f"✍️ Writing the answer with {config.ollama_model}…")

    llm_started = time.perf_counter()
    first_token_s = None
    final = None          # AIMessageChunk accumulated over the stream
    parts: list[str] = []

    try:
        for chunk in get_llm(config).stream(prompt):
            final = chunk if final is None else final + chunk
            text = chunk.content if isinstance(chunk.content, str) else ""
            if text:
                if first_token_s is None:
                    first_token_s = time.perf_counter() - llm_started
                parts.append(text)
                yield {"type": "token", "text": text}
    except Exception as exc:
        if parts:
            # Mid-stream: what was sent stays sent. End it honestly.
            logger.error(f"stream_answer: generation failed mid-stream -- {exc!r}")
            note = "\n\n_(The answer was cut off — generation failed partway through.)_"
            yield {"type": "token", "text": note}
            yield {
                "type": "done", "answer": "".join(parts) + note, "sources": sources,
                "degraded": True, "failure_stage": "generation",
                "metrics": _metrics(started, retrieval_s, time.perf_counter() - llm_started),
            }
            return

        # Nothing sent yet, so the ordinary retry path is still available.
        logger.warning(f"stream_answer: stream failed before first token, retrying without streaming -- {exc!r}")
        yield _status("retrying", "⚠️ The model stumbled — retrying…")
        try:
            final = call_with_retry(
                partial(get_llm(config).invoke), prompt, config=config,
                validate=validate_llm_response, call_name="llm_invoke",
            )
        except Exception as exc2:
            logger.error(f"stream_answer: generation failed permanently -- {exc2!r}")
            answer = "I found relevant information but couldn't generate a response right now. Please try again."
            yield {"type": "token", "text": answer}
            yield {
                "type": "done", "answer": answer, "sources": sources, "degraded": True,
                "failure_stage": "generation",
                "metrics": _metrics(started, retrieval_s, time.perf_counter() - llm_started),
            }
            return
        first_token_s = time.perf_counter() - llm_started
        parts = [str(final.content)]
        yield {"type": "token", "text": parts[0]}

    llm_s = time.perf_counter() - llm_started
    metrics = _metrics(started, retrieval_s, llm_s, final)
    metrics["time_to_first_token_ms"] = round(first_token_s * 1000, 1) if first_token_s is not None else None

    yield {
        "type": "done", "answer": "".join(parts), "sources": sources,
        "degraded": False, "failure_stage": None, "metrics": metrics,
    }
