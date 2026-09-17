"""Prompting, the local LLM, retries, and the end-to-end answer.

Notebook Sections 14-17 in one module: context building, the Ollama client,
the generic retry wrapper, and generate_answer() which ties them together.
This is the module a FastAPI layer would import.
"""

from functools import partial

from src.config import CONFIG, Config, logger
from src.payload import format_source
from src.retrieval import RetrievedDoc, retrieve

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

def retrieve_with_retry(query: str, top_k: int | None = None, config: Config = CONFIG):
    """Kept for backwards compatibility. retrieve() now retries internally —
    inside its MLflow RETRIEVER span — so this is a plain pass-through."""
    return retrieve(query, top_k=top_k, config=config)


def generate_answer(question: str, top_k: int | None = None, config: Config = CONFIG) -> dict:
    """End-to-end RAG: retrieve -> build context -> build prompt -> generate.

    Returns {answer, sources, context, degraded, failure_stage?}. `sources` are
    the raw Qdrant payloads, so they carry section_heading and page_label.
    """
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
        }

    docs = [RetrievedDoc(p) for p in points]
    context = build_context(docs)
    prompt = build_prompt(context, question)

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
        }

    return {
        "answer": response.content,
        "sources": [d.metadata for d in docs],
        "context": context,
        "degraded": False,
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
