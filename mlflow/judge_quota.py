"""Make the judge's *quota* survive contact with a real API.

The problem this solves
-----------------------
`DeepEvalScorer.__call__` turns every exception into `Feedback(error=e)`, so a
hard HTTP 429 from the judge looks exactly like a judge that disagreed, or one
that returned bad JSON: the harness prints `'Faithfulness': 1/1 failed` and
nothing else. On 2026-09-15 that cost a full run to diagnose, and the answer
turned out to be four characters long::

    "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
    "quotaValue": "20"

Twenty judge requests **per day** on the `gemini-2.5-flash` free tier. One
evaluation row costs far more than that — see `calls_per_row()` below — so the
run was over before it started and every scorer reported a plausible-looking
failure instead of the real reason.

Two distinct 429s, two opposite responses
-----------------------------------------
This is the distinction the retry logic turns on, and getting it backwards is
worse than having no retry at all:

* **Per-minute** (`...PerMinute...`) — transient. Sleep for the `retryDelay`
  Google hands back and try again. A few seconds of patience buys the run.
* **Per-day** (`...PerDay...`) — terminal. Retrying is not merely useless, it
  is actively harmful: the harness will grind through every remaining scorer
  and every remaining row, waiting on a limit that resets at midnight Pacific.
  So the first per-day 429 **trips a circuit breaker** and every later judge
  call fails instantly, offline, with the real reason attached.

The breaker cannot abort `mlflow.genai.evaluate` from inside a scorer — there
is no hook for that — but it turns a ten-minute pantomime into a few seconds,
and `ragflow.py` reads `stats()` afterwards and prints the real diagnosis
rather than the stock "probably unparseable JSON" guess.

Better still is not starting: `ragflow.py::preflight_judge` spends one judge
call up front and aborts before generation if the quota is already gone.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time

_logger = logging.getLogger("rag")

# Defaults are deliberately small. A judge that needs more than a couple of
# retries is telling you something the retries will not fix.
MAX_RETRIES = int(os.environ.get("JUDGE_MAX_RETRIES", "3"))
MAX_SLEEP = float(os.environ.get("JUDGE_MAX_RETRY_SLEEP", "65"))


class JudgeQuotaExhausted(RuntimeError):
    """A per-day quota is gone. Nothing in this process will bring it back."""


# --- reading the provider's mind -------------------------------------------

_QUOTA_ID = re.compile(r'"quotaId"\s*:\s*"([^"]+)"')
_QUOTA_VALUE = re.compile(r'"quotaValue"\s*:\s*"?(\d+)"?')
_RETRY_DELAY = re.compile(r'"retryDelay"\s*:\s*"?(\d+(?:\.\d+)?)s"?')
_RETRY_IN = re.compile(r"[Pp]lease retry in (\d+(?:\.\d+)?)s")
_MODEL = re.compile(r'"model"\s*:\s*"([^"]+)"')


class QuotaInfo:
    """What a 429 body actually said. Every field is best-effort: providers
    differ, and an unparsed field must never turn into a wrong decision."""

    def __init__(self, text: str):
        self.raw = text
        self.quota_id = _first(_QUOTA_ID, text)
        self.quota_value = _first(_QUOTA_VALUE, text)
        self.model = _first(_MODEL, text)
        delay = _first(_RETRY_DELAY, text) or _first(_RETRY_IN, text)
        self.retry_delay = float(delay) if delay else None

    @property
    def is_daily(self) -> bool:
        """True only when the provider *said* per-day.

        Note what this deliberately does not do: guess. An unrecognised 429 is
        treated as transient and retried, because retrying a daily limit three
        times wastes twelve seconds, while refusing to retry a per-minute limit
        throws away the whole run.
        """
        return bool(self.quota_id) and "perday" in self.quota_id.lower()

    @property
    def is_per_minute(self) -> bool:
        return bool(self.quota_id) and "perminute" in self.quota_id.lower()

    def describe(self) -> str:
        bits = []
        if self.quota_id:
            bits.append(self.quota_id)
        if self.quota_value:
            bits.append(f"limit {self.quota_value}")
        if self.model:
            bits.append(self.model)
        return " — ".join(bits) if bits else "quota details not in the response body"


def _first(pattern: re.Pattern, text: str) -> str | None:
    match = pattern.search(text)
    return match.group(1) if match else None


def is_rate_limit(exc: BaseException) -> bool:
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text or "Too Many Requests" in text


# --- the breaker ------------------------------------------------------------


class _State:
    def __init__(self):
        self.lock = threading.Lock()
        self.calls = 0          # judge calls attempted (retries not counted twice)
        self.retries = 0
        self.failures = 0
        self.tripped: QuotaInfo | None = None

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "calls": self.calls,
                "retries": self.retries,
                "failures": self.failures,
                "quota_exhausted": self.tripped.describe() if self.tripped else None,
                "retry_delay": self.tripped.retry_delay if self.tripped else None,
            }


_state = _State()


def stats() -> dict:
    """Judge-call accounting for the run just finished."""
    return _state.snapshot()


def reset() -> None:
    global _state
    _state = _State()


def tripped() -> QuotaInfo | None:
    return _state.tripped


def calls_per_row(n_scorers: int = 4) -> str:
    """Why a 20/day quota cannot run even one row.

    Each DeepEval metric is several judge calls, not one: Faithfulness alone
    extracts truths from the context, extracts claims from the answer, then
    judges each claim. The exact count varies with the answer, which is the
    point — you cannot budget for it, you can only measure it, which is what
    `stats()['calls']` is for.
    """
    return (
        f"{n_scorers} scorers x 1 row is NOT {n_scorers} judge calls — each DeepEval "
        "metric makes several (extract, then verdict, sometimes per claim or per "
        "chunk). Expect roughly 10-25 calls per row. Check stats()['calls'] after a "
        "run for the real number for your data."
    )


# --- the patch --------------------------------------------------------------


def install() -> dict:
    """Wrap `MlflowDeepEvalLLM.generate` with retry + the circuit breaker.

    Install this *after* judge_json.install(), so the retry sits outside the
    native-JSON/fallback dance and one retry re-runs the whole attempt rather
    than half of it.
    """
    applied = {"quota_guard": False, "max_retries": MAX_RETRIES}

    try:
        from mlflow.genai.scorers.deepeval import models as dm
    except ImportError:
        _logger.warning("judge_quota: DeepEval scorers not importable; no guard installed")
        return applied

    original_generate = dm.MlflowDeepEvalLLM.generate

    def generate(self, prompt: str, schema=None):
        # Breaker already open: fail now, offline, with the real reason. This is
        # what stops a dead run from taking ten minutes to admit it.
        if _state.tripped is not None:
            raise JudgeQuotaExhausted(
                f"judge quota exhausted ({_state.tripped.describe()}); "
                "skipping call without contacting the provider"
            )

        with _state.lock:
            _state.calls += 1

        last_exc: BaseException | None = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                return original_generate(self, prompt, schema=schema)
            except BaseException as exc:  # noqa: BLE001 — re-raised below
                last_exc = exc
                if not is_rate_limit(exc):
                    raise

                info = QuotaInfo(str(exc))

                if info.is_daily:
                    with _state.lock:
                        _state.tripped = info
                        _state.failures += 1
                    _logger.error(
                        "judge_quota: DAILY quota exhausted (%s). Retrying cannot help; "
                        "failing every further judge call immediately.",
                        info.describe(),
                    )
                    raise JudgeQuotaExhausted(
                        f"judge daily quota exhausted: {info.describe()}"
                    ) from exc

                if attempt == MAX_RETRIES:
                    break

                # Honour the provider's own number when it gives one; it knows
                # when the window rolls over and we do not.
                sleep_for = info.retry_delay if info.retry_delay else 2.0 * (2**attempt)
                sleep_for = min(sleep_for + 0.5, MAX_SLEEP)
                with _state.lock:
                    _state.retries += 1
                _logger.warning(
                    "judge_quota: rate limited (%s); retrying in %.1fs (attempt %d/%d)",
                    info.describe(),
                    sleep_for,
                    attempt + 1,
                    MAX_RETRIES,
                )
                time.sleep(sleep_for)

        with _state.lock:
            _state.failures += 1
        raise last_exc  # type: ignore[misc]

    dm.MlflowDeepEvalLLM.generate = generate
    applied["quota_guard"] = True
    return applied


# --- a cheap question to ask the judge --------------------------------------


def probe(model_uri: str, timeout_note: str = "") -> tuple[bool, str]:
    """Spend exactly one judge call to find out whether the judge is usable.

    Returns (ok, message). Cheaper than discovering the same thing after a full
    generation pass, which is the whole reason this exists.
    """
    from mlflow.genai.scorers.llm_backend import ScorerLLMClient

    try:
        client = ScorerLLMClient(model_uri)
    except Exception as exc:  # noqa: BLE001
        return False, f"could not build a judge client for {model_uri!r}: {exc}"

    try:
        reply = client.complete_prompt("Reply with the single word: ok")
    except BaseException as exc:  # noqa: BLE001
        if is_rate_limit(exc):
            info = QuotaInfo(str(exc))
            if info.is_daily:
                return False, quota_advice(model_uri, info)
            delay = f" (provider suggests {info.retry_delay:.0f}s)" if info.retry_delay else ""
            return False, (
                f"judge is rate limited{delay}: {info.describe()}\n"
                "  This one is per-minute, so the run's retry/backoff would probably "
                "ride it out — re-run, or lower --limit."
            )
        return False, f"judge call failed: {type(exc).__name__}: {str(exc)[:400]}{timeout_note}"

    return True, f"judge reachable ({model_uri}) — replied {str(reply).strip()[:40]!r}"


def quota_advice(model_uri: str, info: QuotaInfo) -> str:
    return (
        f"DAILY judge quota already exhausted: {info.describe()}\n"
        f"  {calls_per_row()}\n"
        "  Nothing in this script can fix a per-day cap. Pick one:\n"
        "    1. Judge locally, no quota at all (you already run Ollama):\n"
        "         python mlflow/ragflow.py --judge ollama:/qwen3:4b-instruct\n"
        "       MLflow routes ollama:/ natively to http://localhost:11434/v1.\n"
        "       Set OLLAMA_API_BASE if yours is elsewhere. Weaker judge than\n"
        "       Gemini, but it never runs out, which beats a perfect judge that\n"
        "       will not answer.\n"
        "    2. A Gemini model with a larger free daily allowance, e.g.\n"
        "         --judge gemini:/gemini-2.5-flash-lite\n"
        "         --judge gemini:/gemini-2.0-flash\n"
        "       Free-tier numbers move; check https://ai.dev/rate-limit.\n"
        "    3. Enable billing on the key, or use a different one.\n"
        "  The limit resets at midnight US/Pacific."
    )
