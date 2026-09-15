"""Generic retry with exponential backoff — notebook Section 16.

Lifted out of generation.py so that retrieval.py can use it too without an
import cycle (config -> retry -> ... -> retrieval -> generation). generation.py
re-exports everything here, so existing imports keep working.

Why retrieval needed it: the RETRIEVER span now lives on `retrieve()`, and
retrying a traced function from the outside emits one span per attempt — a
failed attempt would put an empty context on the trace next to the good one.
Retrying *inside* the traced function means one span per retrieval, always the
successful one.
"""

import random
import time
from typing import Callable, TypeVar

from src.config import CONFIG, Config, logger

T = TypeVar("T")


class ValidationFailed(Exception):
    """Raised internally when a call succeeded (no exception) but the result
    failed its validation check -- e.g. an empty LLM response, or a retrieval
    that returned zero points. Treated the same as an exception by the retry
    loop, so bad-but-non-crashing results still get retried."""


def call_with_retry(
    fn: Callable[..., T],
    *args,
    validate: Callable[[T], bool] | None = None,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    config: Config = CONFIG,
    call_name: str = "call",
    **kwargs,
) -> T:
    """Call fn(*args, **kwargs), retrying on exception or failed validation.

    Deliberately generic: this wraps a single external call (one LLM invoke, one
    Qdrant query, one tool call) rather than a whole LangGraph node. That
    granularity matters once nodes bundle several external calls together --
    you want to retry the flaky call, not redo everything else in the node.

    NOTE `config` here is the *retry* configuration and is NOT forwarded to
    `fn`. If `fn` needs a Config, bind it with functools.partial before calling
    this (see retrieval.retrieve). Previously `config=` was passed through here
    in the hope it would reach retrieve()/invoke() -- it never did, and the
    callee silently used the module-level CONFIG.

    Raises the last exception (or ValidationFailed) once retries are exhausted.
    Callers decide what "graceful failure" means for them.
    """
    last_exc: Exception | None = None

    for attempt in range(1, config.retry_max_attempts + 1):
        try:
            result = fn(*args, **kwargs)
            if validate is not None and not validate(result):
                raise ValidationFailed(f"{call_name}: result failed validation")
            if attempt > 1:
                logger.info(f"{call_name}: succeeded on attempt {attempt}")
            return result

        except retryable_exceptions as exc:
            last_exc = exc
            if attempt == config.retry_max_attempts:
                logger.error(f"{call_name}: failed after {attempt} attempts -- {exc!r}")
                break

            delay = min(
                config.retry_base_delay_seconds * (config.retry_backoff_factor ** (attempt - 1)),
                config.retry_max_delay_seconds,
            )
            delay += random.uniform(0, config.retry_jitter_seconds)
            logger.warning(f"{call_name}: attempt {attempt} failed ({exc!r}), retrying in {delay:.2f}s")
            time.sleep(delay)

    if last_exc is not None:
        raise last_exc

    raise RuntimeError(
        f"{call_name}: no attempts were made; "
        "CONFIG.retry_max_attempts must be greater than zero"
    )


def validate_retrieval(points: list) -> bool:
    """A retrieval returning zero points isn't an exception, but it isn't usable
    either -- worth a retry (transient Qdrant hiccup) before giving up."""
    return len(points) > 0


def validate_llm_response(response) -> bool:
    """Empty or whitespace-only content from Ollama usually means the model was
    still loading or the call was truncated -- retry rather than return blank."""
    return bool(response.content and response.content.strip())
