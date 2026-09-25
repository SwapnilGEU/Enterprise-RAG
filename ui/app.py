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
import json
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
    "kb_prefetched", "latency_ms", "ttft_ms", "degraded", "failure_stage", "status",
    "error", "request_id", "steps",
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
              error: str | None, latency_ms: float, steps: list[str] | None = None) -> dict:
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
        # None from the non-streaming /agent, which does not report it.
        "_kb_matched": payload.get("kb_matched"),
        # True only when the knowledge base searched and had NO answer. This,
        # not _kb_matched, decides the "not from your documents" notice:
        # kb_matched is a word-overlap guess and misses paraphrases, while a
        # refusal from the knowledge base itself is a fact.
        "_kb_refused": payload.get("kb_refused"),
        "latency_ms": payload.get("latency_ms", round(latency_ms, 1)),
        "ttft_ms": payload.get("time_to_first_token_ms"),
        "degraded": bool(payload.get("degraded", False)),
        "failure_stage": payload.get("failure_stage") or "",
        "status": status,
        "error": error or "",
        "request_id": payload.get("request_id", ""),
        "steps": " → ".join(steps or []),
        "_sources": sources,          # underscore keys stay out of the CSV
        "_steps": list(steps or []),
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


def stream_api(base_url: str, endpoint: str, question: str, *, include_context: bool,
               session_id: str, timeout: int):
    """POST one question to the streaming endpoint and yield its events.

    The API sends newline-delimited JSON — one event per line — and this yields
    each as a dict the moment its line arrives:

        {"type": "status", "message": "🔎 Searching the knowledge base…"}
        {"type": "token",  "text": "Hybrid "}
        {"type": "reset"}                   agent only: clear this turn's text
        {"type": "done",   ...the same fields /query or /agent return...}
        {"type": "error",  "detail": "..."}

    Replaced call_api() for the chat: `requests.post(...)` without
    `stream=True` reads the WHOLE body before returning, so even a streaming
    server would look frozen until the last token. `stream=True` plus
    iter_lines() is what lets each token through as it arrives.

    Never raises, like call_api(): transport problems become an error event.
    Falls back to the non-streaming endpoint on a 404, so this UI still works
    against an API from before streaming existed.
    """
    url = f"{base_url}/{endpoint.strip('/')}/stream"
    if endpoint.strip("/") == "agent":
        body = {"question": question, "session_id": session_id}
    else:
        body = {"question": question, "include_context": include_context}
    headers = {"X-Request-ID": uuid.uuid4().hex}

    for attempt in (1, 2):
        try:
            # (connect, read) — the read timeout is the longest allowed GAP
            # between bytes, not the whole answer, which is what you want for
            # a stream.
            response = requests.post(url, json=body, headers=headers, stream=True,
                                     timeout=(10, timeout))
        except requests.exceptions.Timeout:
            yield {"type": "error", "detail": f"timed out after {timeout}s"}
            return
        except requests.exceptions.ConnectionError:
            yield {"type": "error", "detail": f"cannot reach {base_url} — is the API running?"}
            return
        except Exception as exc:  # noqa: BLE001
            yield {"type": "error", "detail": f"{type(exc).__name__}: {exc}"}
            return

        if response.status_code == 200:
            with response:
                try:
                    # Bytes, not decode_unicode: NDJSON has no charset header,
                    # and json.loads() reads UTF-8 bytes directly.
                    for line in response.iter_lines():
                        if not line:
                            continue
                        try:
                            yield json.loads(line)
                        except ValueError:
                            continue
                except requests.exceptions.RequestException as exc:
                    yield {"type": "error", "detail": f"stream interrupted: {type(exc).__name__}"}
            return

        if response.status_code == 404 and endpoint.strip("/") in ("query", "agent"):
            response.close()
            payload, status, error = call_api(
                base_url, endpoint, question, top_k=None, include_context=include_context,
                session_id=session_id, timeout=timeout,
            )
            if status != "ok":
                yield {"type": "error", "detail": error}
                return
            yield {"type": "status", "message": "ℹ️ This API has no streaming endpoint — showing the full answer"}
            yield {"type": "token", "text": payload.get("answer", "")}
            yield {"type": "done", **payload}
            return

        detail = _detail_of(response)
        response.close()
        if response.status_code == 503 and attempt == 1:
            time.sleep(2)
            continue
        yield {"type": "error", "detail": f"HTTP {response.status_code}: {detail}"}
        return


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
        "in_flight": None,     # the question being answered right now
        "running": False,      # batch in progress
        "cancel": False,       # stop after the current item
        "confirm_clear": False,
        "session_id": f"ui-{uuid.uuid4().hex[:8]}",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


# --- app --------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Enterprise RAG", page_icon="🤖", layout="wide")
    init_state()

    # ---- sidebar ----------------------------------------------------------
    with st.sidebar:
        st.title("⚙️ Settings")

        base_url = st.text_input("API base URL", value=API_BASE_URL)
        endpoint = st.radio(
            "Endpoint", ["query", "agent"],
            captions=["Retrieval + answer", "LangGraph agent with tools"],
            horizontal=True,
        )
        include_context = st.checkbox(
            "Include chunk text", value=False,
            help="Returns the retrieved text, not just citations. /query only.",
        )

        st.divider()
        st.subheader("🩺 Service health")
        if st.button("🔄 Refresh", use_container_width=True):
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
        st.subheader("🗂️ Chat tools")

        # Two clicks, because one stray click should not destroy a conversation
        # that has not been exported yet.
        if not st.session_state.confirm_clear:
            if st.button("🧹 Clear chat", use_container_width=True,
                         disabled=not st.session_state.chat):
                st.session_state.confirm_clear = True
                st.rerun()
        else:
            if st.button("⚠️ Really clear?", type="secondary", use_container_width=True):
                st.session_state.chat = []
                st.session_state.confirm_clear = False
                st.rerun()
            st.caption("Click again to confirm — export first if you want to keep this.")

        st.download_button(
            "📥 Export CSV",
            data=rows_to_csv(st.session_state.chat) if st.session_state.chat else "",
            file_name=f"rag-chat-{datetime.now():%Y%m%d-%H%M%S}.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=not st.session_state.chat,
        )

        st.divider()
        st.caption(f"session: {st.session_state.session_id}")

    # ---- chat history ------------------------------------------------------
    st.subheader("💬 Hey! Want help?")
    if not st.session_state.chat:
        st.info("👋 Nothing asked yet — type a question at the bottom to get started.")

    for row in st.session_state.chat:
        with st.chat_message("user"):
            st.write(row["question"])
        with st.chat_message("assistant"):
            steps = row.get("_steps") or []
            if steps:
                with st.expander(f"🧭 {len(steps)} step(s)"):
                    for step in steps:
                        st.caption(step)
            if row["status"] == "error":
                st.error(row["error"])
            else:
                st.markdown(row["answer"])
                if row["status"] == "stopped":
                    st.warning("⏹️ Stopped — this answer is incomplete.")
                if row["degraded"]:
                    st.warning(
                        f"Degraded answer — the pipeline failed at: {row['failure_stage'] or 'unknown'}. "
                        "Retrieval may not have contributed."
                    )
                # The pre-search is shown apart from the tools the model chose,
                # so "tools chosen:" keeps meaning "what the agent decided"
                # instead of reading rag_tool on literally every answer.
                #
                # Collapsed into one phrase when the pre-search is the whole
                # story. "knowledge base pre-searched · no tool chosen" sitting
                # under a cited, knowledge-base-grounded answer reads as though
                # the knowledge base went unused, when that is precisely the
                # case where it did all the work and the model needed nothing
                # further. Both facts are still reported — just not as two
                # clauses that look like they disagree.
                marks = []
                if row.get("kb_prefetched") and not row["tools_used"] and row.get("_kb_refused"):
                    # The knowledge base searched and had no answer, and no
                    # tool ran, so this is the model's general knowledge.
                    marks.append("generated by the model — not retrieved from the knowledge base")
                elif row.get("kb_prefetched") and not row["tools_used"]:
                    marks.append("answered from the pre-searched knowledge base")
                else:
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
            first = f" · first token {row['ttft_ms']} ms" if row.get("ttft_ms") is not None else ""
            st.caption(
                f"{row['endpoint']} · {row['latency_ms']} ms{first} · {row['timestamp']} · {row['request_id'][:8]}"
            )

    # Filled in during processing, below — declared here so progress appears in
    # the right place on the page rather than at the bottom.
    live_slot = st.container()

    # ---- pending ------------------------------------------------------------
    # Only shown when something is actually waiting. With one-at-a-time sending
    # a visible empty queue is just furniture.
    pending = st.session_state.queue

    # Busy means "a question is already on its way to an answer" — either one is
    # mid-request (in_flight, set by the runner below) or one is queued and this
    # very run is about to start it. Both must count: the runner pops from the
    # queue AFTER the widgets are drawn, so gating on in_flight alone would
    # leave the box enabled for exactly the run that does the work.
    busy = bool(pending) or st.session_state.in_flight is not None

    if pending:
        waiting = ", ".join(f"“{q[:40]}”" for q in pending[:3])
        more = f" (+{len(pending) - 3} more)" if len(pending) > 3 else ""
        st.caption(f"⏳ {len(pending)} waiting: {waiting}{more}")

    # ---- ask, with Stop beside it, pinned to the bottom --------------------
    # st.chat_input pins itself to the bottom of the viewport only while it is a
    # direct child of the main container. Putting it in a column — which is what
    # lets Stop sit beside it — makes it render inline instead, so the whole row
    # goes inside st.bottom to get the pinning back. st.bottom is the public API
    # for this as of Streamlit 1.60; before that it was st._bottom, which is why
    # requirements-ui.txt asks for 1.60.
    with st.bottom:
        ask_col, stop_col = st.columns([0.9, 0.1], vertical_alignment="bottom")

        # Disabled while a question is in flight. The widgets are drawn before
        # the queue runner blocks, so this renders disabled straight away and
        # stays that way for the whole wait — which is the frame the user looks
        # at for the next 15-20 seconds.
        #
        # The queue below is kept even so. Streamlit can still deliver a
        # submission made in the instant before the disable reaches the browser,
        # and when it does, the queue is what stops it being dropped.
        with ask_col:
            question = st.chat_input(
                "Answering — one moment…" if busy else "Ask a question…",
                disabled=busy,
            )

        with stop_col:
            if st.button(
                "🛑 Stop", use_container_width=True, disabled=not busy,
                help="Stops the answer being written now (keeping what has arrived) "
                     "and drops anything still waiting.",
            ):
                st.session_state.queue = []
                st.session_state.cancel = True
                st.rerun()

    # Handled after both widgets are drawn, so Stop is on the page before the
    # rerun that a new question triggers.
    if question and question.strip():
        st.session_state.queue.append(question.strip())
        st.rerun()

    # ---- the queue runner --------------------------------------------------
    # Last in the script, so everything above has already rendered: the user
    # sees the answers so far while this one is in flight. Anything on the queue
    # runs — no separate Send, which is the whole point of the change.
    if st.session_state.queue:
        current = st.session_state.queue.pop(0)
        remaining = len(st.session_state.queue)

        # Recorded in session state, not just the local `current`, for the whole
        # duration of the call. Between the pop above and the append below the
        # question used to exist ONLY in that local: off the queue, not yet in
        # the chat. Anything that restarted the script in that window took the
        # question with it. Now it survives, and `busy` above can see it.
        st.session_state.in_flight = current

        # Streamed (added 2026-09-25). The old runner called call_api(), which
        # blocked until the API had generated the WHOLE answer and only then
        # drew anything. Now progress lines and tokens are drawn as they
        # arrive, inside a chat bubble at the bottom of the history.
        started = time.perf_counter()
        steps: list[str] = []
        parts: list[str] = []
        payload: dict = {}
        status, error = "ok", None
        recorded = False
        try:
            with live_slot:
                if remaining:
                    st.caption(f"⏳ {remaining} more waiting after this one")
                with st.chat_message("user"):
                    st.markdown(current)
                with st.chat_message("assistant"):
                    progress = st.status("⏳ Sending…", expanded=True)
                    answer_box = st.empty()

                    for event in stream_api(
                        base_url, endpoint, current, include_context=include_context,
                        session_id=st.session_state.session_id, timeout=REQUEST_TIMEOUT,
                    ):
                        kind = event.get("type")
                        if kind == "status":
                            message = str(event.get("message", ""))
                            steps.append(message)
                            progress.write(message)
                            progress.update(label=message)
                        elif kind == "token":
                            parts.append(str(event.get("text", "")))
                            # The trailing block is a cursor, so it is obvious
                            # the answer is still being written.
                            answer_box.markdown("".join(parts) + " ▌")
                        elif kind == "reset":
                            # The agent started writing, then chose a tool.
                            parts.clear()
                            answer_box.empty()
                        elif kind == "done":
                            payload = event
                            # The server's final text wins: the agent tidies
                            # its answer at the end (one top source appended,
                            # any model-written citations removed).
                            if event.get("answer"):
                                parts[:] = [str(event["answer"])]
                        elif kind == "error":
                            status, error = "error", str(event.get("detail", "unknown error"))

                    answer_box.markdown("".join(parts))
                    progress.update(
                        label="✅ Done" if status == "ok" else "❌ Failed",
                        state="complete" if status == "ok" else "error",
                        expanded=False,
                    )

            if status == "ok" and not payload:
                status, error = "error", "The stream ended without a final answer."
            if payload and not payload.get("answer"):
                payload["answer"] = "".join(parts)

            elapsed = (time.perf_counter() - started) * 1000
            st.session_state.chat.append(
                build_row(endpoint, current, payload, status, error, elapsed, steps=steps)
            )
            recorded = True
        finally:
            # Reached without a row when Stop was pressed: Streamlit restarts
            # the script mid-stream, which abandons the HTTP request (the API
            # sees the disconnect and frees its generation slot). Keep what had
            # already arrived rather than losing the question entirely.
            #
            # Also releases the input box if anything above raised — without
            # this one unexpected exception locks it for the whole session.
            if not recorded:
                elapsed = (time.perf_counter() - started) * 1000
                st.session_state.chat.append(
                    build_row(endpoint, current, {"answer": "".join(parts)}, "stopped",
                              "Stopped before the answer finished.", elapsed, steps=steps)
                )
            st.session_state.in_flight = None

        # Stop clears the queue, so the loop ends naturally after this one.
        st.session_state.cancel = False
        st.rerun()


if __name__ == "__main__":
    main()
