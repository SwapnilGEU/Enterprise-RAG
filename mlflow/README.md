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
`--skip-preflight`.

## What you need before it will run

| | |
|---|---|
| `GEMINI_API_KEY` | repo-root `.env` or `mlflow/.env`. Both are loaded, root wins. |
| `mlflow server` | on `127.0.0.1:5000`, or set `MLFLOW_TRACKING_URI`. |
| Ollama | running, with `qwen3:4b-instruct` pulled. |
| Qdrant | the `RAG-hybrid-search` collection populated. |
| `evaluation/datasets/rag_qa.json` | a JSON list of `{query, reference_answer}`. Not in the repo — see `rag_qa.example.json` next to it for the shape. |

## The one thing that silently breaks this

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
| `429` / quota | free-tier rate limit; run with `--limit` and wait. |

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
  Switch back to a two-phase run only if you move the judge onto Ollama.
- `MLFLOW_GENAI_EVAL_MAX_WORKERS=1` is set in the script before mlflow is
  imported (it is read at import time) so `predict_fn` calls are serial —
  parallel workers make a 6GB card thrash between model loads.

## Secrets

`mlflow/.env` and the Qdrant API key defaulted in `src/config.py` are both live
credentials sitting in the tree. Rotate them, move them to `.env` (gitignored),
and leave `.env.example` as the committed template.
