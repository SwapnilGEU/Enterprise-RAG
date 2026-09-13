# Enterprise RAG Pipeline

PDF / DOCX / XLSX → structured Markdown → sections → chunks → Qdrant →
hybrid retrieval + rerank → grounded answer → tool-using agent.

Modularised from `Prototype4.ipynb`. Every module maps to a numbered notebook
section, noted in its docstring.

## Layout

```
src/                 the library — pure functions and lazy factories
  config.py          Config + logger                            (§2)
  models.py          StructuralUnit, Chunk — shared dataclasses
  extraction.py      PDF / DOCX / XLSX → records                (§3-5)
  markdown.py        heading detection → Markdown               (§6)
  sections.py        Markdown → sections, with page numbers     (§7)
  chunking.py        semantic + fixed chunking                  (§8-9)
  payload.py         build_payload / format_source              (§12.2, §14)
  vector_store.py    client, collection, dedup, upload          (§10-12)
  retrieval.py       hybrid search + ColBERT rerank             (§13)
  generation.py      prompt, LLM, retries, generate_answer      (§14-17)
  history.py         Postgres query_history                     (§18)
  agent.py           tools + LangGraph graph                    (§19-20)

scripts/             entry points — the notebook's linear flow
tests/               the "check the outcome" cells, as assertions
evaluation/          golden datasets + eval scripts (MLflow goes here)
notebooks/           thin notebook that imports from src/
data/raw/            put your source documents here
```

**The one rule:** each module only imports from ones above it in that list. If
you ever want to import upward, something is in the wrong file.

## Setup

```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # then fill in QDRANT_API_KEY and PG_PASSWORD
```

You also need, running locally: **Ollama** (`ollama pull qwen3:4b-instruct`) and
**Postgres** (only for the agent). Qdrant is cloud.

## Use

```bash
# 1. Index — run whenever data/raw/ changes
python scripts/run_index.py --dry-run      # parse + chunk only, no network, no credentials
python scripts/run_index.py                # for real

# 2. Ask (needs Qdrant + Ollama)
python scripts/run_query.py "What is hybrid retrieval?"
python scripts/run_query.py "..." --show-context

# 3. Agent (also needs Postgres)
python scripts/run_agent.py "How many queries have failed so far?" --show-tools

# 4. Evaluate
python evaluation/eval_rag.py --limit 3
python evaluation/eval_agent.py

# 5. Test
pytest                      # fast, no network
pytest -m integration       # the ones needing live services
```

Start with `--dry-run`. It exercises extraction, Markdown conversion, section
splitting and chunking without needing a single credential, so if something is
wrong with your documents you find out in seconds.

## How the citations work

This is the part worth understanding, because it is where the pipeline broke
once and the failure was silent.

Section splitting (`sections.py`) records the heading chain each section sits
under. Chunking (`chunking.py`) copies it onto every chunk. `payload.py` turns
it into the Qdrant payload and, at query time, back into a citation:

```
chunk.metadata          {"section_path": ["Foundation Models", "Pre-training"],
                         "page_start": 41, "page_end": 42, ...}
        ↓ build_payload
Qdrant payload          {"section_heading": "Foundation Models > Pre-training",
                         "page_label": "pp.41–42", ...}
        ↓ format_source
citation                [Foundation-LLMs.pdf — Foundation Models > Pre-training, pp.41–42]
```

**Qdrant only returns what the payload holds.** In Prototype 3.3 the upload
step wrote five keys and dropped `metadata` entirely, so every citation came
back as `[Unknown section, p.?]` while the metadata sat intact one stage
upstream. `build_payload` and `format_source` live together in one small module
with one test file (`tests/test_payload.py`) precisely so the write side and the
read side can never disagree again.

`run_index.py` reports "Chunks with no section heading: N/total" before it
uploads anything. A non-zero N means heading detection missed, and it is far
cheaper to fix there than to discover it in an answer.

## Notes for what comes next

- **No module-level side effects.** `get_client()`, `get_llm()`,
  `get_connection()`, `build_agent()` and `get_embedder()` are all lazy
  singletons, so importing any module opens no connections and loads no models.
  That is what lets `pytest` run without Qdrant and what will let a FastAPI
  worker import `generation.py` without dragging in pymupdf.
- **Two lifecycles, two dependency sets.** Index-time needs pymupdf / openpyxl /
  sentence-transformers (heavy); query-time needs qdrant-client + langchain
  (light). `requirements.txt` marks the split — when you get to Docker, that is
  where the two images divide.
- **`num_ctx` is set explicitly** (`config.ollama_num_ctx`, default 16384).
  Ollama defaults it to 4096 whatever the model supports, and `num_predict`
  comes out of that budget, not on top of it.
- **`rag_tool` returns the answer plus citations, not the whole result dict.**
  In an agent loop every tool return is permanent context; returning the
  retrieved chunks would carry thousands of tokens through every later turn.
