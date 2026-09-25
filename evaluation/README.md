# evaluation/

Quick, local evaluation scripts, plus the datasets every evaluation in this
repo reads, including the MLflow ones in [`mlflow/`](../mlflow/README.md).

| Where | What it's for |
|---|---|
| `evaluation/` (here) | Fast checks that print to the console and write a CSV. No MLflow and no Gemini. The judge is your local Ollama model. |
| `mlflow/` | The full tracked evaluations: DeepEval metrics, a Gemini judge, and runs logged to MLflow so you can compare them. |

Use this folder for "did my change break anything?" and `mlflow/` for numbers
you want to keep and compare.

## What's here

```
evaluation/
├── _common.py          shared helpers; loads .env (must be imported first)
├── eval_rag.py         RAG answer quality + latency/token metrics
├── eval_agent.py       agent tool routing: did it pick the right tool?
├── label_chunks.py     one-off manual labelling for the retrieval metrics
├── datasets/
│   ├── rag_qa.json              golden set: 10 questions + reference answers
│   ├── rag_qa.labelled.json     the same, plus which chunks answer each one
│   ├── rag_qa.example.json      tiny example of the format
│   ├── tool_routing.json        8 questions + the tool each should use
│   └── archive/                 older versions (pre-2026-09-20), kept for comparison
└── results/            CSV / JSON output (git-ignored apart from .gitkeep)
```

## Before you run anything

Every script needs **Qdrant** and **Ollama**. `eval_agent.py` also needs
**Postgres**, because the agent's SQL tool connects when the graph is built.

Settings come from `.env` in the repo root. `_common.py` loads it before
anything imports `src/`, which is why every script starts with
`import _common`. If you skip it, the Qdrant key comes through blank.

Run from the repo root:

```bash
python evaluation/eval_rag.py
```

## The scripts

### `eval_rag.py`: answer quality

```bash
python evaluation/eval_rag.py                 # full golden set
python evaluation/eval_rag.py --limit 3       # smoke run
python evaluation/eval_rag.py --top-k 8       # try a different top-k
python evaluation/eval_rag.py --preview 1200  # show more of each answer
```

For each question in `rag_qa.json` it runs `generate_answer()`, then asks the
judge whether the answer matches the reference. It reports accuracy plus
latency and tokens (retrieval ms, LLM ms, tokens/sec).

Output: `results/rag_evaluation_topk<k>.csv`. The top-k is in the file name,
so a sweep leaves one file per setting side by side.

### `eval_agent.py`: tool routing

```bash
python evaluation/eval_agent.py
python evaluation/eval_agent.py --limit 3
```

Runs each question in `tool_routing.json` through the LangGraph agent and
checks two things:

1. **Routing**: did it call the expected tool? One of `rag_tool`,
   `sql_tool`, `get_weather`, or `direct_llm` (no tool at all).
2. **Answer**: does the judge accept the final answer?

Output: `results/agent_evaluation_report.csv` and
`results/agent_evaluation_summary.json`.

With `AGENT_RAG_FIRST=1` (the default), the knowledge base is searched
automatically before the model's first turn, so there is no toolless path and
the `direct_llm` rows will fail. That is expected. To measure the model's own
routing, run with `AGENT_RAG_FIRST=0`.

### `label_chunks.py`: labels for retrieval metrics

```bash
python evaluation/label_chunks.py            # interactive, resumable
python evaluation/label_chunks.py --dump     # write candidates to review offline
python evaluation/label_chunks.py --relabel  # start over
```

The one manual step. For each golden question it shows candidate chunks from
three retrievers (dense, sparse, and hybrid + ColBERT), and you mark the ones
that answer it. The result goes to `datasets/rag_qa.labelled.json`.

Candidates come from all three retrievers, not just the current pipeline, so
the labels describe the corpus rather than today's ranking. Otherwise any
change could only ever look like a regression.

Once the file exists, `mlflow/retrievalflow.py` computes hit-rate@k, MRR,
recall@k and precision@k from it. Those metrics are deterministic and need no
judge. Re-label when you add documents that should answer existing questions.

## Datasets

| File | Shape | Read by |
|---|---|---|
| `rag_qa.json` | `{query, reference_answer}` | `eval_rag.py`, `mlflow/ragflow.py`, `label_chunks.py` |
| `rag_qa.labelled.json` | `rag_qa.json` + relevant chunk ids | `mlflow/retrievalflow.py` |
| `tool_routing.json` | `{query, expected_tool, reference_answer}` | `eval_agent.py`, `mlflow/agentflow.py` |

To add a case, append an object in the same shape. For RAG questions, re-run
`label_chunks.py` afterwards so the new case gets labels. It resumes and only
asks about unlabelled questions.

## The judge here is the same model that answers

`_common.judge()` uses `get_llm()`, which is whatever `OLLAMA_MODEL` is set
to (`qwen3:4b-instruct` by default). A model grading its own answers measures the
judge's limits as much as the system's quality, so read these scores as a smoke
test. For scores you'll quote, use `mlflow/ragflow.py`, which uses Gemini as
the judge.

If you change `OLLAMA_MODEL`, the judge changes with it, so don't compare
scores across models.

## Streaming doesn't affect these scripts

These scripts call `generate_answer()` and `graph.invoke()` directly, not the
new `/query/stream` or `/agent/stream` endpoints. Their latency numbers are
total time. Time-to-first-token is recorded by the API (the
`rag.generation.time_to_first_token` metric) and shown in the UI.
