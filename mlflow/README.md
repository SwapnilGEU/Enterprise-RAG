# RAG evaluation — MLflow + DeepEval, judged by Gemini

## Run it

```bash
pip install -r mlflow/requirements-eval.txt

# 1. tracking server, in its own terminal, from the repo root
mlflow server

# 2. smoke test — 1 row, all four scorers. Always do this first.
python mlflow/ragflow.py --smoke

# 3. the full golden set
python mlflow/ragflow.py
```

Open <http://127.0.0.1:5000> for per-row scores and the traces behind them.

Useful flags: `--limit N`, `--data path/to/other.json`, `--judge gemini:/gemini-2.5-pro`,
`--contextual-relevancy` (adds the fifth, weakest scorer and roughly doubles judge time),
`--skip-preflight`, `--skip-judge-preflight`.

**If the Gemini free tier is in your way, judge locally instead:**

```bash
python mlflow/ragflow.py --judge ollama:/qwen3:4b-instruct
```

MLflow routes `ollama:/` through its own native provider to
`http://localhost:11434/v1` — no API key, no quota, no daily cap. The 4b model
is a weaker judge than Gemini, so treat its absolute numbers with suspicion;
for comparing two retrieval configurations against each other it is fine, and
it always answers. Set `OLLAMA_API_BASE` if yours is not on the default port.
This does put the judge and the generator on the same 6GB card — see the
two-phase note under "Judge notes".

## What you need before it will run

| | |
|---|---|
| `GEMINI_API_KEY` | repo-root `.env` or `mlflow/.env`. Both are loaded, root wins. |
| `mlflow server` | on `127.0.0.1:5000`, or set `MLFLOW_TRACKING_URI`. |
| Ollama | running, with `qwen3:4b-instruct` pulled. |
| Qdrant | the `RAG-hybrid-search` collection populated. |
| `evaluation/datasets/rag_qa.json` | a JSON list of `{query, reference_answer}`. Not in the repo — see `rag_qa.example.json` next to it for the shape. |

## The thing that actually breaks this: judge quota

Measured on 2026-09-15, from the 429 body itself:

```
"quotaId":    "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
"quotaValue": "20"
"model":      "gemini-2.5-flash"
```

**Twenty judge requests per day.** Not per minute — per day, resetting at
midnight US/Pacific. One evaluation row costs far more than that, because each
DeepEval metric is several judge calls rather than one: Faithfulness extracts
truths from the context, extracts claims from the answer, then judges the
claims. Budget roughly 10–25 calls per row for four scorers; the exact number
depends on the answer, which is why `ragflow.py` now prints

```
judge calls: 47   retries: 2   failures: 0
```

at the end of every run. Nobody guesses that number correctly the first time.

So on the free tier, `gemini-2.5-flash` cannot evaluate even a single row. The
run still *completes* — that is the trap. Every scorer returns
`Feedback(error=...)` and the harness prints the same `1/1 failed` it prints
for a JSON parse failure, so the natural next move is to go and debug the JSON,
which is fine, and not the problem.

`mlflow/judge_quota.py` handles this in three ways:

1. **Preflight.** One judge call before anything else — before generation, not
   after. An exhausted quota now aborts in about a second with exit code 2 and
   tells you what to run instead. Skip it with `--skip-judge-preflight`.
2. **Per-minute vs per-day are opposites.** A `...PerMinute...` 429 is
   transient: sleep for the `retryDelay` the provider hands back, retry
   (`JUDGE_MAX_RETRIES`, default 3). A `...PerDay...` 429 is terminal, so
   retrying is not merely useless but actively harmful — it makes a dead run
   take ten minutes to admit it. The first per-day 429 trips a circuit breaker
   and every later judge call fails instantly, offline. An *unrecognised* 429
   is treated as transient, on the grounds that retrying a daily cap three
   times wastes twelve seconds while not retrying a per-minute cap throws away
   the run.
3. **Honest reporting.** If the breaker tripped mid-run, the closing summary
   says `RUN INVALID` and returns exit code 3, instead of printing scores that
   are mostly holes.

What it cannot do is invent quota. The fixes are the ones the preflight prints:
a local Ollama judge, a Gemini model with a bigger free allowance
(`gemini-2.5-flash-lite`, `gemini-2.0-flash` — check
<https://ai.dev/rate-limit>, the numbers move), or billing.

## The other thing that silently breaks this

`Faithfulness`, `ContextualPrecision` and `ContextualRecall` get their
`retrieval_context` **only** from top-level `RETRIEVER` spans on the trace.
MLflow builds it in `extract_retrieval_context_from_trace`; there is no way to
pass context in through `inputs` or `expectations`. Without the span those
three scorers do not error — they score an empty context and return a
plausible-looking number. Only `AnswerRelevancy` is unaffected.

So `src/retrieval.py::retrieve` is decorated:

```python
@mlflow.trace(span_type=SpanType.RETRIEVER, name="retrieve")
def retrieve(query, top_k=None, config=CONFIG) -> list[dict]:
    ...
    return [{"page_content": ..., "metadata": {...}}, ...]
```

The span's output must be a **list of dicts** with the chunk text under
`page_content`, `content` or `text`, plus an optional `metadata` dict of which
only `doc_uri` is read. Anything else is dropped at debug level. That is why
`retrieve()` now returns dicts rather than raw Qdrant `ScoredPoint`s;
`RetrievedDoc` accepts either, so nothing downstream changed.

`ragflow.py --smoke` runs one real question, reads the trace back and prints
how many chunks parsed. If it prints zero, the scorers are about to lie to you
and the script aborts.

### Chunk-level granularity (optional)

MLflow's `retrieval_context` has one entry per retriever **span**, not per
chunk — all five chunks arrive as one stringified blob. `ContextualPrecision`
judges whether relevant nodes rank above irrelevant ones, and with a single
node that question is degenerate.

```bash
RETRIEVER_SPAN_PER_CHUNK=1 python mlflow/ragflow.py
```

emits one retriever span per chunk in rank order (`retrieved_chunk_1..k`), with
`retrieve` demoted to a `CHAIN` span so the per-chunk ones stay top-level.
Worth it when you are comparing rerankers or sweeping `top_k`; off by default
because a single `retrieve` span is the conventional shape.

## When all the scorers fail at once

```
WARNING ... 'AnswerRelevancy': 1/1 failed, 'ContextualPrecision': 1/1 failed, ...
```

`DeepEvalScorer.__call__` swallows every exception into `Feedback(error=e)`, so
the harness only tells you *that* they failed. Get the real message:

```bash
python mlflow/diagnose.py            # latest run
python mlflow/diagnose.py --full     # whole error text
```

The usual cause is **JSON transport, not the judge's opinion**. MLflow asks for
structured output by prompt injection — it appends the schema to the prompt and
runs a bare `json.loads` on the reply, with no fence stripping. A judge that
answers with

````
```json
{"verdicts": [...]}
```
````

fails to parse, every row, every scorer. `mlflow/judge_json.py` fixes it two ways,
both on by default and both applied before the scorers are built:

1. **Native structured output** (`JUDGE_NATIVE_JSON=0` to disable) — MLflow never
   passes `response_format` down, so Gemini's own JSON mode is left off. Passing
   the schema through makes the gateway set `responseJsonSchema` +
   `responseMimeType: application/json`, and Gemini then cannot emit a fence at
   all. Fix at the source. Falls back to prompt injection if the provider
   rejects the schema.
2. **Tolerant parsing** (`JUDGE_JSON_REPAIR=0` to disable) — strips fences and
   surrounding prose before `json.loads`. The safety net for the fallback.

`--stock-judge` turns both off, to compare against stock MLflow behaviour.

If `diagnose.py` shows something else instead:

| error | meaning |
|---|---|
| `API key not valid` / `PERMISSION_DENIED` | `GEMINI_API_KEY` is wrong or is an OAuth token rather than an AI Studio key. |
| `404 models/... is not found` | wrong model name for your key's API version — try `--judge gemini:/gemini-2.0-flash`. |
| `429` / `RESOURCE_EXHAUSTED` | read `quotaId` in the body. `...PerMinute...` — the retry in `judge_quota.py` should ride it out; lower `--limit`. `...PerDay...` — done for the day, see the quota section above. You should not normally get here, because the judge preflight catches it first. |

## Judge notes

- **Every scorer needs an explicit `model=`.** MLflow's default judge is OpenAI
  `gpt-4o-mini`; a scorer built without it reaches for `OPENAI_API_KEY`.
  Centralised in `build_scorers()`.
- `gemini:/gemini-2.5-flash` routes through MLflow's **native** gateway
  provider (verified: `ScorerLLMClient.route == "native"`), not litellm, and
  reads `GEMINI_API_KEY` from the environment.
- MLflow does structured output by **prompt injection** — it appends the JSON
  schema to the prompt and `json.loads` the reply. It does not use native
  structured output. A judge that emits prose around its JSON fails to parse,
  and `DeepEvalScorer.__call__` catches that and marks the row errored instead
  of raising — so a run can come back quietly half-empty. Hence `--smoke`.
- Gemini is a hosted judge, so there is no VRAM contention with Ollama and the
  two-phase generate-then-judge split the plan describes is unnecessary here.
  Switch back to a two-phase run only if you move the judge onto Ollama —
  which `--judge ollama:/qwen3:4b-instruct` now does, so if you adopt that as
  the default judge, the two-phase split stops being hypothetical. Generator
  and judge then compete for the same 6GB, and `OLLAMA_MAX_LOADED_MODELS=1`
  (already set in `ragflow.py`) means they evict each other on every
  alternation. Generating all answers first, then judging them, would fix it.
- `MLFLOW_GENAI_EVAL_MAX_WORKERS=1` is set in the script before mlflow is
  imported (it is read at import time) so `predict_fn` calls are serial —
  parallel workers make a 6GB card thrash between model loads.

## Secrets

`mlflow/.env` and the Qdrant API key defaulted in `src/config.py` are both live
credentials sitting in the tree. Rotate them, move them to `.env` (gitignored),
and leave `.env.example` as the committed template.
