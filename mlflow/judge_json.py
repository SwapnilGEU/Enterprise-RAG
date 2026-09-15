"""Make the judge's JSON survive contact with a real model.

MLflow asks DeepEval's judge for structured output by **prompt injection**: it
appends the pydantic schema to the prompt, asks for "ONLY the JSON object", and
then runs a bare `json.loads` on the reply. No fence stripping, no retry. Any
model that answers with

    ```json
    {"verdicts": [...]}
    ```

or with one sentence of preamble fails to parse. `DeepEvalScorer.__call__`
catches the exception and returns `Feedback(error=e)` rather than raising, so
the run completes and every row is silently marked errored — which is exactly
the "1/1 failed" for all four scorers.

Two patches, both opt-out:

1. **Native structured output** (`JUDGE_NATIVE_JSON`, default on).
   `MlflowDeepEvalLLM.generate` never passes `response_format` down, so
   Gemini's own JSON mode is left switched off. Passing the schema through
   makes MLflow's gateway set `responseJsonSchema` + `responseMimeType:
   application/json`, and Gemini then *cannot* emit a fence or a preamble.
   This is a fix at the source rather than a cleanup after the fact.
   Falls back to the prompt-injection path if the provider rejects the schema
   (not every provider accepts every JSON-Schema construct).

2. **Tolerant parsing** (`JUDGE_JSON_REPAIR`, default on).
   Strips ``` fences and any prose around the outermost JSON object before
   `json.loads`. This is the safety net for patch 1's fallback, and for
   providers with no JSON mode at all.

Neither patch changes what is asked of the judge or how it scores — only how
its reply is transported. Set either env var to 0 to compare against stock
behaviour.
"""

import json
import logging
import os
import re

_logger = logging.getLogger("rag")

_FENCE = re.compile(r"^\s*```(?:json|JSON)?\s*(.*?)\s*```\s*$", re.DOTALL)


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("", "0", "false", "no")


def extract_json(text: str) -> str:
    """Best-effort: pull the JSON value out of whatever the judge actually said.

    Handles ```json fences, prose before/after, and a leading 'Here is the
    JSON:'. Returns the input unchanged if nothing better is found, so the
    original json.loads error message is what the caller sees.
    """
    if not isinstance(text, str):
        return text

    candidate = text.strip()

    fenced = _FENCE.match(candidate)
    if fenced:
        candidate = fenced.group(1).strip()

    # Outermost {...} or [...] anywhere in the reply, brace-matched so a nested
    # object doesn't cut it short and trailing prose doesn't come along. Strings
    # are tracked so a brace inside a quoted value doesn't count. Whichever
    # opener appears first wins, so an array reply isn't mistaken for the first
    # object inside it.
    starts = [(candidate.find(o), o, c) for o, c in (("{", "}"), ("[", "]"))]
    starts = sorted((s, o, c) for s, o, c in starts if s != -1)

    for start, opener, closer in starts:
        depth = 0
        in_string = False
        escaped = False
        for i, ch in enumerate(candidate[start:], start):
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return candidate[start : i + 1]

    return candidate


def install() -> dict:
    """Apply the patches. Returns which ones went on, for logging."""
    applied = {"native_json": False, "json_repair": False}

    try:
        from mlflow.genai.scorers.deepeval import models as dm
    except ImportError:
        _logger.warning("judge_json: MLflow DeepEval scorers not importable; no patches applied")
        return applied

    # --- 2. tolerant parsing ------------------------------------------------
    if _env_flag("JUDGE_JSON_REPAIR"):
        original_parse = dm._parse_json_output_with_schema

        def tolerant_parse(output, schema):
            try:
                return original_parse(output, schema)
            except ValueError:
                repaired = extract_json(output)
                if repaired == output:
                    raise
                _logger.debug("judge_json: repaired a non-bare-JSON judge reply")
                return schema(**json.loads(repaired))

        dm._parse_json_output_with_schema = tolerant_parse
        applied["json_repair"] = True

    # --- 1. native structured output ---------------------------------------
    if _env_flag("JUDGE_NATIVE_JSON"):
        original_generate = dm.MlflowDeepEvalLLM.generate

        def generate(self, prompt: str, schema=None):
            if schema is None:
                return original_generate(self, prompt, schema=None)

            try:
                response = self._backend.complete_prompt(
                    prompt, response_format=schema, **self._model_kwargs
                )
                return dm._parse_json_output_with_schema(response.strip(), schema)
            except Exception as exc:
                # Provider rejected the schema, or returned something unusable.
                # Fall back to MLflow's prompt-injection path, which the repair
                # patch above now backs up.
                _logger.debug(
                    f"judge_json: native structured output failed ({exc!r}); "
                    "falling back to prompt-injected JSON"
                )
                return original_generate(self, prompt, schema=schema)

        dm.MlflowDeepEvalLLM.generate = generate
        applied["native_json"] = True

    return applied
