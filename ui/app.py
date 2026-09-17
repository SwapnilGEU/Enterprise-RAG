"""Streamlit front end for the Enterprise RAG API.

    # terminal 1
    uvicorn api.main:app --reload
    # terminal 2
    streamlit run ui/app.py

Talks to the API over HTTP and nothing else — no `src/` imports, no Qdrant key,
no Ollama, no database. That is what lets it be a separate image with a tiny
dependency set (streamlit, requests, pandas) and be pointed at a container later
by changing one environment variable.

    API_BASE_URL=http://api:8000 streamlit run ui/app.py

Queueing
--------
Ask and it sends — there is no separate "add" step. The queue exists only so a
second question typed while the first is still answering is **held rather than
lost**. It answers in order, one at a time, and nothing is dropped or
interrupted.

Mechanically: each submission appends to a pending list, and the runner handles
**one item per Streamlit rerun**. Looping over the whole list inside a single
run would block the script, so Streamlit would paint nothing until the last
answer landed and the page would just sit there looking broken. One item then
`st.rerun()` means every answer appears the moment it arrives.

Sequential rather than parallel because the API caps concurrent generations at
`API_MAX_CONCURRENT_GENERATIONS` (default 2) and behind that is one 4b model on
a 6GB card. Firing several at once does not finish sooner — the extras block on
the server's semaphore and, past `API_GENERATION_QUEUE_TIMEOUT`, turn into
503s. Parallelism here would relocate the queue and add a failure mode.
"""

from __future__ import annotations

import io
import os
import time
import uuid
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

# The one seam between local and Docker. Compose sets this to the service name
# (http://api:8000); hardcoding localhost would make the container talk to
# itself and fail with a confusing connection refused.
API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

# Must comfortably exceed the server's worst case: up to 120s waiting for a
# generation slot, plus the generation itself. `requests` defaults to no timeout
# at all, which would freeze the browser tab forever with no way out.
REQUEST_TIMEOUT = int(os.environ.get("UI_REQUEST_TIMEOUT", "300"))
HEALTH_TIMEOUT = 5

CSV_COLUMNS = [
    "timestamp", "endpoint", "question", "answer", "citations", "tools_used",
    "kb_prefetched", "latency_ms", "degraded", "failure_stage", "status",
    "error", "request_id",
]


# --- pure helpers (no Streamlit, so they are testable) ----------------------


def rows_to_csv(rows: list[dict]) -> str:
    """Chat rows -> CSV text. Exports exactly what is on screen.

    `request_id` earns its column: the API returns it in the body and stamps it
    on every JSON log line, so a bad answer in this file can be traced straight
    back to its log entry — and, once tracing is wired up, to its trace.
    """
    frame = pd.DataFrame(rows, columns=CSV_COLUMNS)
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue()


def build_row(endpoint: str, question: str, payload: dict, status: str,
              error: str | None, latency_ms: float) -> dict:
    """One chat/CSV row. Same shape whether the call succeeded or not, so a
    failed query still exports rather than vanishing from the record."""
    sources = payload.get("sources") or []
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "endpoint": endpoint,
        "question": question,
        "answer": payload.get("answer", ""),
        "citations": " | ".join(s.get("citation", "") for s in sources),
        "tools_used": ", ".join(payload.get("tools_used") or []),
        "kb_prefetched": bool(payload.get("retrieval_prefetched", False)),
        "latency_ms": payload.get("latency_ms", round(latency_ms, 1)),
        "degraded": bool(payload.get("degraded", False)),
        "failure_stage": payload.get("failure_stage") or "",
        "status": status,
        "error": error or "",
        "request_id": payload.get("request_id", ""),
        "_sources": sources,          # underscore keys stay out of the CSV
    }


def call_api(base_url: str, endpoint: str, question: str, *, top_k: int | None,
             include_context: bool, session_id: str, timeout: int) -> tuple[dict, str, str | None]:
    """POST one question. Returns (payload, status, error).

    Never raises: a transport failure is a row in the chat like any other, so
    one bad query does not abort a batch of twenty.

    A 503 is retried once after a short pause — from this API it means "no
    generation slot", i.e. busy rather than broken. Anything else is recorded
    as-is; retrying a 422 or a 502 just wastes the user's time.
    """
    url = f"{base_url}/{endpoint.lstrip('/')}"
    if endpoint.strip("/") == "agent":
        body = {"question": question, "session_id": session_id}
    else:
        body = {"question": question, "include_context": include_context}
        if top_k:
            body["top_k"] = top_k

    # Correlates this click with the API's logs. Once OpenTelemetry is wired up,
    # a `traceparent` header here is what would make one trace span both
    # services; X-Request-ID is the same idea with no dependencies.
    headers = {"X-Request-ID": uuid.uuid4().hex}

    for attempt in (1, 2):
        try:
            response = requests.post(url, json=body, headers=headers, timeout=timeout)
        except requests.exceptions.Timeout:
            return {}, "error", f"timed out after {timeout}s"
        except requests.exceptions.ConnectionError:
            return {}, "error", f"cannot reach {base_url} — is the API running?"
        except Exception as exc:  # noqa: BLE001
            return {}, "error", f"{type(exc).__name__}: {exc}"

        if response.status_code == 200:
            return response.json(), "ok", None

        detail = _detail_of(response)
        if response.status_code == 503 and attempt == 1:
            time.sleep(2)
            continue
        return {}, "error", f"HTTP {response.status_code}: {detail}"

    return {}, "error", "unreachable"


def _detail_of(response) -> str:
    try:
        data = response.json()
    except Exception:  # noqa: BLE001
        return response.text[:300]
    detail = data.get("detail", data)
    return str(detail)[:400]


def fetch_ready(base_url: str) -> tuple[bool, dict, str | None]:
    """GET /ready. 503 is a normal answer here, not a failure — it carries the
    per-component detail, which is the whole point of that endpoint."""
    try:
        response = requests.get(f"{base_url}/ready", timeout=HEALTH_TIMEOUT)
    except Exception as exc:  # noqa: BLE001
        return False, {}, f"{type(exc).__name__}: {exc}"

    try:
        data = response.json()
    except Exception:  # noqa: BLE001
        return False, {}, f"HTTP {response.status_code}"

    if response.status_code == 503:
        return False, data.get("detail", data), None
    return bool(data.get("ready")), data, None


# --- state ------------------------------------------------------------------


def init_state() -> None:
    defaults = {
        "chat": [],            # completed rows
        "queue": [],           # pending questions
        "running": False,      # batch in progress
        "cancel": False,       # stop after the current item
        "confirm_clear": False,
        "session_id": f"ui-{uuid.uuid4().hex[:8]}",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


# --- app --------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Enterprise RAG", page_icon="📚", layout="wide")
    init_state()

    # ---- sidebar ----------------------------------------------------------
    with st.sidebar:
        st.title("Enterprise RAG")

        base_url = st.text_input("API base URL", value=API_BASE_URL)
        endpoint = st.radio(
            "Endpoint", ["query", "agent"],
            captions=["Retrieval + answer", "LangGraph agent with tools"],
            horizontal=True,
        )
        top_k = st.slider("top_k", 1, 20, 5, help="Chunks to retrieve. /query only.")
        include_context = st.checkbox(
            "Include chunk text", value=False,
            help="Returns the retrieved text, not just citations. /query only.",
        )

        st.divider()
        st.subheader("Service health")
        if st.button("Refresh", use_container_width=True):
            st.rerun()

        ready, data, error = fetch_ready(base_url)
        if error:
            st.error(f"Cannot reach the API\n\n{error}")
        elif ready:
            st.success("ready")
        else:
            st.warning("not ready")

        components = (data or {}).get("components", {})
        for name, info in components.items():
            if info.get("ready"):
                st.caption(f"✅ {name}")
            else:
                required = "required" if info.get("required") else "optional"
                st.caption(f"❌ {name} ({required})")
                if info.get("error"):
                    st.caption(f"　　{info['error'][:110]}")
        for missing in (data or {}).get("missing_settings", []) or []:
            st.caption(f"⚠️ unset: {missing}")

        st.divider()
        st.caption(f"session: {st.session_state.session_id}")

    # ---- chat history ------------------------------------------------------
    st.subheader("Conversation")
    if not st.session_state.chat:
        st.info("No questions yet. Add one below, then Send.")

    for row in st.session_state.chat:
        with st.chat_message("user"):
            st.write(row["question"])
        with st.chat_message("assistant"):
            if row["status"] != "ok":
                st.error(row["error"])
            else:
                st.write(row["answer"])
                if row["degraded"]:
                    st.warning(
                        f"Degraded answer — the pipeline failed at: {row['failure_stage'] or 'unknown'}. "
                        "Retrieval may not have contributed."
                    )
                # The pre-search is shown apart from the tools the model chose,
                # so "tools:" keeps meaning "what the agent decided" instead of
                # reading rag_tool on literally every answer.
                marks = []
                if row.get("kb_prefetched"):
                    marks.append("knowledge base pre-searched")
                if row["tools_used"]:
                    marks.append(f"tools chosen: {row['tools_used']}")
                elif row["endpoint"] == "agent":
                    marks.append("no tool chosen")
                if marks:
                    st.caption(" · ".join(marks))
                sources = row.get("_sources") or []
                if sources:
                    with st.expander(f"{len(sources)} source(s)"):
                        for source in sources:
                            st.markdown(f"**{source.get('citation','')}**")
                            if source.get("score") is not None:
                                st.caption(f"score {source['score']:.4f}")
                            if source.get("text"):
                                st.text(source["text"][:1500])
            st.caption(
                f"{row['endpoint']} · {row['latency_ms']} ms · {row['timestamp']} · {row['request_id'][:8]}"
            )

    # Filled in during processing, below — declared here so progress appears in
    # the right place on the page rather than at the bottom.
    status_slot = st.empty()

    # ---- pending ------------------------------------------------------------
    # Only shown when something is actually waiting. With one-at-a-time sending
    # a visible empty queue is just furniture.
    pending = st.session_state.queue
    if pending:
        waiting = ", ".join(f"“{q[:40]}”" for q in pending[:3])
        more = f" (+{len(pending) - 3} more)" if len(pending) > 3 else ""
        st.caption(f"⏳ {len(pending)} waiting: {waiting}{more}")

    # Never disabled. Streamlit holds a submission made while the script is busy
    # and delivers it on the next run, where it lands on the queue — which is
    # exactly the "typed a second question by mistake" case: it waits its turn
    # instead of being dropped or interrupting the one in flight.
    question = st.chat_input("Ask a question…")
    if question and question.strip():
        st.session_state.queue.append(question.strip())
        st.rerun()

    # ---- controls ----------------------------------------------------------
    controls = st.columns(3)

    if controls[0].button(
        "Stop", use_container_width=True, disabled=not pending,
        help="Drops what is still waiting. The question already in flight finishes — "
             "a blocking request cannot be cancelled.",
    ):
        st.session_state.queue = []
        st.session_state.cancel = True
        st.rerun()

    # Two clicks, because one stray click should not destroy a conversation
    # that has not been exported yet.
    if not st.session_state.confirm_clear:
        if controls[1].button("Clear chat", use_container_width=True,
                              disabled=not st.session_state.chat):
            st.session_state.confirm_clear = True
            st.rerun()
    else:
        if controls[1].button("Really clear?", type="secondary", use_container_width=True):
            st.session_state.chat = []
            st.session_state.confirm_clear = False
            st.rerun()

    controls[2].download_button(
        "Export CSV",
        data=rows_to_csv(st.session_state.chat) if st.session_state.chat else "",
        file_name=f"rag-chat-{datetime.now():%Y%m%d-%H%M%S}.csv",
        mime="text/csv",
        use_container_width=True,
        disabled=not st.session_state.chat,
    )

    if st.session_state.confirm_clear:
        st.caption("Click again to confirm — export first if you want to keep this.")

    # ---- the queue runner --------------------------------------------------
    # Last in the script, so everything above has already rendered: the user
    # sees the answers so far while this one is in flight. Anything on the queue
    # runs — no separate Send, which is the whole point of the change.
    if st.session_state.queue:
        current = st.session_state.queue.pop(0)
        remaining = len(st.session_state.queue)
        status_slot.info(
            f"Answering: {current}" + (f"  ·  {remaining} waiting" if remaining else "")
        )

        started = time.perf_counter()
        payload, status, error = call_api(
            base_url, endpoint, current,
            top_k=top_k, include_context=include_context,
            session_id=st.session_state.session_id, timeout=REQUEST_TIMEOUT,
        )
        elapsed = (time.perf_counter() - started) * 1000
        st.session_state.chat.append(
            build_row(endpoint, current, payload, status, error, elapsed)
        )

        # Stop clears the queue, so the loop ends naturally after this one.
        st.session_state.cancel = False
        st.rerun()


if __name__ == "__main__":
    main()
