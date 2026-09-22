<h1 align="center">Enterprise RAG</h1>

<p align="center">
  A document question-answering system you can actually run, watch and measure.<br>
  <sub>PDF · DOCX · XLSX → hybrid retrieval → grounded, cited answers → tool-using agent</sub>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white">
  <img alt="Qdrant" src="https://img.shields.io/badge/Qdrant-DC244C?logo=qdrant&logoColor=white">
  <img alt="Ollama" src="https://img.shields.io/badge/Ollama-local%20LLM-000000?logo=ollama&logoColor=white">
  <img alt="LangGraph" src="https://img.shields.io/badge/LangGraph-agent-1C3C3C">
  <img alt="Grafana" src="https://img.shields.io/badge/Grafana-LGTM-F46800?logo=grafana&logoColor=white">
  <img alt="MLflow" src="https://img.shields.io/badge/MLflow-evaluation-0194E2?logo=mlflow&logoColor=white">
</p>

---

Ask a question about your own documents and get an answer **with citations you
can check**. Everything runs on your machine except the vector store — no
answer-quality data leaves the box, because the model is local Ollama.

What makes this more than a demo:

| | |
|---|---|
| **Citations that survive the round trip** | Every chunk carries its heading chain and page numbers all the way into the answer |
| **Hybrid retrieval** | Dense + sparse (BM25) + ColBERT reranking, computed server-side by Qdrant |
| **An agent that routes** | Knowledge base, SQL over query history, or live weather — and it explains which it used |
| **Real observability** | Traces, metrics and logs into Grafana, correlated by `trace_id`, in one container |
| **Evaluation you can trust** | Retrieval metrics with **no LLM judge**, so they cost nothing and can't be rate limited |
| **Honest readiness** | `/ready` reports each dependency separately; Postgres going down doesn't take retrieval with it |

<p align="center">
  <img src="docs/images/grafana-dashboard.png" alt="Grafana dashboard: request volume, latency percentiles, retrieval vs generation time" width="900">
</p>

---

## Results

Measured on an RTX 4050 (6GB) with Qwen3-4B-Instruct via Ollama, against a
10-question golden set over **5,667 chunks** from 4 documents. Judge is
Gemini 2.5 Flash. Reproduce with `python mlflow/ragflow.py`.

### Answer quality

| Metric | Pass rate | What it measures |
|---|---|---|
| **Faithfulness** | **100%** | Every claim in the answer is supported by the retrieved context — no hallucination |
| **Contextual precision** | **100%** | Relevant chunks ranked above irrelevant ones |
| **Contextual recall** | **90%** | The retrieved context covers what the reference answer needs |
| **Answer relevancy** | **80%** | The answer actually addresses the question asked |

Faithfulness is the one that matters most for a RAG system: at 100%, the model
is answering *from the documents* rather than from its own weights.

### Latency

| Path | Typical | Notes |
|---|---|---|
| `/query` — retrieve + answer | **~6.0s** | One generation |
| `/agent` — knowledge-base match | **~11.6s** | Fast path: one generation, no routing turn |
| `/agent` — tool call required | **~24.8s** | Routing turn + tool + answer |
| `/health` under 6 concurrent generations | **9ms** | The event loop stays free — handlers run in a threadpool |

### Engineering

| | |
|---|---|
| Agent fast path on KB-answered questions | **2 generations → 1** |
| Boot time (warmup moved off the startup path) | **12ms** |
| Serving image vs. full install | **~2GB smaller** (no torch / sentence-transformers) |
| Tests | **53** unit + **54** API assertions, network-free |

<sub>Sample size is 10 questions — enough to catch a regression, not enough to
publish. Retrieval-level metrics (hit_rate@k, recall@k, MRR) run judge-free via
<code>mlflow/retrievalflow.py</code>.</sub>

---

## How it fits together

```mermaid
flowchart LR
    subgraph Ingest["Index time"]
        D["PDF · DOCX · XLSX"] --> E[extract] --> M[markdown] --> S[sections] --> C[chunk]
    end
    C --> Q[("Qdrant Cloud<br/>dense + sparse + ColBERT")]

    subgraph Serve["Query time"]
        U["Streamlit UI"] --> API["FastAPI"]
        API -->|/query| R[retrieve] --> G[generate]
        API -->|/agent| AG["LangGraph agent"]
        AG --> T1[rag_tool] & T2[sql_tool] & T3[get_weather]
    end

    R --> Q
    AG --> Q
    G --> O(["Ollama<br/>qwen3:4b"])
    AG --> O
    AG --> P[(Postgres<br/>query_history)]

    API -.->|OTLP| L{{"grafana/otel-lgtm<br/>Tempo · Prometheus · Loki · Grafana"}}
```

---

## Quickstart

The fastest path is Docker. You need **Docker Desktop**, **Ollama**, and a free
**Qdrant Cloud** cluster.

**1. Get the model**

```bash
ollama pull qwen3:4b-instruct
```

**2. Configure**

```bash
cp .env.example .env
```

Fill in three things — the rest have working defaults:

| variable | where it comes from |
|---|---|
| `QDRANT_URL` | your cluster page at [cloud.qdrant.io](https://cloud.qdrant.io) |
| `QDRANT_API_KEY` | Qdrant → Data Access Control → API Keys (shown once) |
| `PG_PASSWORD` | your local Postgres — or set `API_ENABLE_AGENT=false` and skip it |

**3. Add your documents**

Drop PDFs, Word docs or spreadsheets straight into `data/raw/`. No subfolders —
they aren't scanned.

**4. Index them**

```bash
pip install -r requirements.txt
python scripts/run_index.py --dry-run     # parse + chunk only, no credentials needed
python scripts/run_index.py               # for real
```

Always start with `--dry-run`. It exercises extraction, heading detection and
chunking without touching the network, so a problem with your documents shows
up in seconds rather than after an upload.

**5. Run everything**

```bash
docker compose up -d --build
```

| what | where |
|---|---|
| Ask questions | <http://localhost:8501> |
| API docs | <http://localhost:8000/docs> |
| Dashboards | <http://localhost:3000> (admin / admin) |

Check it came up cleanly:

```bash
curl http://localhost:8000/ready
```

Every dependency reports separately, so a failure names what's broken instead
of erroring somewhere deeper. First check can take ~30s while Ollama loads.

> **No Docker?** `uvicorn api.main:app` and `streamlit run ui/app.py` work
> exactly the same. See [DOCKER.md](DOCKER.md) for the differences.

<p align="center">
  <img src="docs/images/ui-answer.png" alt="The Streamlit UI answering a question with citations and per-component service health" width="900">
</p>

---

## Seeing what it's doing

One extra container gives you traces, metrics and logs, already wired together:

```bash
docker compose up -d lgtm    # included in the compose file
```

Open Grafana, find a slow request in the logs, click its `trace_id`, and see
exactly where the time went — Qdrant search vs. Ollama generation vs. waiting
for a generation slot.

<p align="center">
  <img src="docs/images/loki-logs.png" alt="Loki showing per-request log lines with latency and tool usage" width="900">
</p>

**→ [OBSERVABILITY.md](OBSERVABILITY.md)** — what's collected, the queries
worth knowing, and the ready-made dashboard.

---

## Testing and evaluation

Three layers, answering three different questions.

**1. Does the code work?** — 53 unit tests plus 54 API assertions, none of
which need network. The notebook's "check the outcome" cells became
assertions, so extraction, chunking, the payload round trip and the vector
store are all covered offline.

```bash
pytest                      # fast, no services required
pytest -m integration       # the ones that need live Qdrant / Ollama / Postgres
```

**2. Is retrieval finding the right chunks?** — deterministic, **no LLM
judge**, so it costs nothing and cannot be rate limited. `hit_rate@k`, `MRR@k`,
`recall@k` and `precision@k` at k in {1,3,5,10,20}, against human-labelled
chunk ids.

```bash
python mlflow/retrievalflow.py --label "baseline"
```

The diagnostic worth knowing: **`recall@20` vs `recall@5`**. The first is what
dense + sparse found at all; the second is what survived the ColBERT rerank. A
large gap means the chunks *are* being retrieved and the reranker is burying
them — a completely different fix from "the chunks were never found".

**3. Are the answers any good?** — four DeepEval scorers through MLflow.

```bash
python mlflow/ragflow.py                  # faithfulness, relevancy, contextual precision/recall
python mlflow/agentflow.py --no-judge     # tool routing — deterministic, free
python mlflow/agentflow.py --judge <name> # Local Model calling as judge eg:ollama:/qwen3:4b-instruct
```

<p align="center">
  <img src="docs/images/mlflow-eval.png" alt="MLflow evaluation run: four scorers across 10 questions" width="880">
</p>

Every run logs parameters, metrics and traces to MLflow, so "did adding those
three papers help?" becomes a comparison rather than a feeling.

**→ [mlflow/README.md](mlflow/README.md)** for the full evaluation story,
including the judge-quota circuit breaker and why the labels come from three
independent retrievers rather than from the current pipeline.

---

## Reliability

Small model, network dependencies, one GPU — things fail. The interesting part
is *which* failures are handled, and how.

**Validation at the edge.** Pydantic schemas reject bad input before any work
happens: `question` is 1–2000 characters, `top_k` is 1–50, `session_id` is
capped. A malformed request costs a 422, not a wasted generation.

**Retries that understand "succeeded but useless".** `call_with_retry` does 3
attempts with exponential backoff — 0.5s, 1s, 2s, capped at 8s, plus up to
0.25s of jitter so parallel callers don't synchronise on a shared service. The
part worth stealing is that it retries on **validation**, not only exceptions:

```python
validate_retrieval(points)       # zero results isn't an error — but it isn't usable
validate_llm_response(response)  # empty content usually means the model was still loading
```

Neither of those raises. Both are worth one more attempt before giving up, and
without this they would sail through as a successful empty answer.

**Degraded answers are a field, not an error.** If generation partly fails, the
response is still a 200 carrying `degraded: true` and `failure_stage`, because
a partial answer with citations beats a 502. That is also why
`rag_answers_total{degraded="true"}` exists as a metric — an error-rate panel
cannot see a successful-looking failure.

**Per-component readiness.** Postgres going down 503s `/agent` while `/query`
stays 200. Failed components are re-checked on `/ready`, rate limited and
lock-guarded, so a dependency that comes back is picked up automatically — no
restart needed.

**Bounded generation.** A semaphore caps concurrent generations; past the
timeout you get a 503 you can act on instead of an invisible queue.

---

<details>
<summary><b>Running it without Docker</b></summary>

```bash
python -m venv venv && source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                               # then fill it in

python scripts/run_index.py                        # index your documents
python scripts/run_query.py "What is hybrid retrieval?"
python scripts/run_agent.py "How many queries have failed so far?" --show-tools

uvicorn api.main:app                               # http://localhost:8000
streamlit run ui/app.py                            # http://localhost:8501
```

To send telemetry from a local run, copy `api/.env.example` to `api/.env` and
start the backend with `docker compose up -d lgtm`.

</details>

<details>
<summary><b>The services behind it</b></summary>

**Qdrant — cloud, required.** The vector store. Free tier is plenty. Create a
cluster, copy the URL and an API key. You don't create the collection yourself;
`run_index.py` does that on first run.

Dense, sparse and ColBERT vectors are all computed **server-side**
(`cloud_inference=True`), which is why serving needs no local embedding model
and no GPU beyond what Ollama uses.

**Ollama — local, required.** The generator. `ollama pull qwen3:4b-instruct`,
then leave it running. Check with `curl http://localhost:11434/api/tags`.

**Postgres — local, optional.** Only `/agent` and query-history logging need
it. `CREATE DATABASE "RAG";` and the table is created automatically. Set
`API_ENABLE_AGENT=false` to skip it entirely — `/query` is unaffected.

</details>

<details>
<summary><b>Project layout</b></summary>

```
src/                 the library — pure functions and lazy factories
  config.py          Config + logger
  models.py          StructuralUnit, Chunk — shared dataclasses
  extraction.py      PDF / DOCX / XLSX → records
  markdown.py        heading detection → Markdown
  sections.py        Markdown → sections, with page numbers
  chunking.py        semantic + fixed chunking
  payload.py         build_payload / format_source
  vector_store.py    client, collection, dedup, upload
  retrieval.py       hybrid search + ColBERT rerank
  generation.py      prompt, LLM, retries, generate_answer
  retry.py           backoff + validators
  history.py         Postgres query_history
  agent.py           tools + LangGraph graph

api/                 FastAPI service — routes, state, logging, telemetry
ui/                  Streamlit front end
scripts/             entry points — run_index, run_query, run_agent, run_api
mlflow/              evaluation flows: retrieval, answer quality, agent
evaluation/          golden datasets and standalone eval scripts
observability/       the Grafana dashboard
tests/               assertions, mostly network-free
data/raw/            put your source documents here
```

**The one rule:** each module only imports from ones above it in that list. If
you ever want to import upward, something is in the wrong file.

</details>

<details>
<summary><b>How citations actually work — and how they once broke</b></summary>

This is the part worth understanding, because it is where the pipeline broke
once and the failure was silent.

Section splitting records the heading chain each section sits under. Chunking
copies it onto every chunk. `payload.py` turns it into the Qdrant payload and,
at query time, back into a citation:

```
chunk.metadata          {"section_path": ["Foundation Models", "Pre-training"],
                         "page_start": 41, "page_end": 42, ...}
        ↓ build_payload
Qdrant payload          {"section_heading": "Foundation Models > Pre-training",
                         "page_label": "pp.41–42", ...}
        ↓ format_source
citation                [Foundation-LLMs.pdf — Foundation Models > Pre-training, pp.41–42]
```

**Qdrant only returns what the payload holds.** An earlier prototype's upload
step wrote five keys and dropped `metadata` entirely, so every citation came
back as `[Unknown section, p.?]` while the metadata sat intact one stage
upstream. `build_payload` and `format_source` now live together in one small
module with one test file, precisely so the write side and the read side can
never disagree again.

`run_index.py` prints "Chunks with no section heading: N/total" before
uploading anything. A non-zero N means heading detection missed, and that is
far cheaper to fix there than to discover in an answer.

</details>

<details>
<summary><b>Design decisions worth knowing</b></summary>

- **No module-level side effects.** `get_client()`, `get_llm()`,
  `get_connection()` and `build_agent()` are lazy singletons, so importing any
  module opens no connections and loads no models. That's what lets `pytest`
  run without Qdrant.
- **Sync handlers, never `async def`.** The whole pipeline is blocking; in an
  async handler one slow generation would stall every concurrent request
  including `/health`. FastAPI runs `def` handlers in a threadpool instead —
  which is why `api/state.py` exists to make the singletons thread-safe.
- **Two slots, on purpose.** One 4b model on a 6GB card doesn't go faster with
  ten concurrent requests, it thrashes.
- **Readiness is per component.** A single boolean would pull a healthy
  retrieval endpoint out of a load balancer for a dependency it never touches.
- **The agent takes a fast path.** When the automatic knowledge-base search
  clearly matches the question, the answer comes straight from those documents
  — skipping a routing turn the model didn't need. `AGENT_RAG_FAST_PATH=0`
  disables it.
- **`num_ctx` is set explicitly** (default 16384). Ollama defaults it to 4096
  whatever the model supports, and `num_predict` comes out of that budget
  rather than on top of it.
- **`rag_tool` returns the answer plus citations, not the whole result dict.**
  In an agent loop every tool return is permanent context; returning the raw
  chunks would carry thousands of tokens through every later turn.

</details>

---

## Documentation

| | |
|---|---|
| [DOCKER.md](DOCKER.md) | Containers, troubleshooting, publishing images |
| [OBSERVABILITY.md](OBSERVABILITY.md) | Traces, metrics, logs and the dashboard |
| [mlflow/README.md](mlflow/README.md) | Evaluation — retrieval, answer quality, agent |
| [observability/README.md](observability/README.md) | The Grafana dashboard, and why there's so little config |
