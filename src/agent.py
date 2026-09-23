"""Tools and the LangGraph agent — notebook Sections 19 and 20.

Graph shape:  retrieve_first -> agent -> (tools -> agent)* -> log_to_db -> END

The loop back from tools to agent is what lets the model chain calls (read a
SQL result, then decide it needs another query) rather than being limited to
one tool per turn. Logging is a graph node, not a wrapper, so every path
through the graph gets logged.

Everything here is built by factory functions — `import src.agent` opens no
connections and loads no models.

Why `retrieve_first` exists (added 2026-09-17)
----------------------------------------------
The system prompt below used to say the model MUST call `rag_tool` before
answering a technical question. `qwen3:4b-instruct` did not reliably obey it:
asked "what is rag", it answered from its own weights and then offered — "I can
search the knowledge base for more detailed insights, would you like me to?" —
while the knowledge base held the answer all along. Instruction-following at 4b
is not strong enough to carry a rule that matters this much.

So the knowledge base is no longer consulted at the model's discretion. The
graph retrieves **first, always**, and hands the result to the model as a
completed `rag_tool` call before it gets its first turn. The model still has
every tool available afterwards, so it can follow up with SQL or weather — it
simply no longer gets to skip the knowledge base.

Two consequences worth knowing:

* Every question now pays one retrieval, including "what is 2+2". On this stack
  that is a second or two against a hosted Qdrant. `AGENT_RAG_FIRST=0` turns it
  off and restores the model's discretion.
* It changes what `mlflow/agentflow.py` measures. `ToolCorrectness` is subset
  based, so a case expecting `get_weather` still passes when `rag_tool` also
  ran — but the `direct_llm` rows in `evaluation/datasets/tool_routing.json`
  expect *no* tool at all and will now score 0. That is an honest result rather
  than a regression: with rag-first there is no toolless path. Either evaluate
  with `AGENT_RAG_FIRST=0` to measure the model's own routing, or update those
  rows to expect `rag_tool`.
"""

import operator
import os
import re
import time
import uuid
from typing import Annotated, Mapping, Sequence, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool

from src.config import CONFIG, Config, logger
from src.generation import generate_answer, get_llm
from src.history import save_query_history
from src.usage import llm_usage, total_tokens_per_sec
from src.payload import format_source


# Tool-call ids from the automatic pre-search carry this prefix, so everything
# downstream can tell "the graph did this on its own" apart from "the model
# decided to do this". Without the distinction `tools_used` reads `rag_tool` on
# every single answer — which is noise in the UI, and would quietly make
# agentflow's ToolCorrectness grade the plumbing instead of the model.
PREFETCH_CALL_PREFIX = "ragfirst-"


def is_prefetch_call(call: Mapping[str, object]) -> bool:
    return str(call.get("id", "")).startswith(PREFETCH_CALL_PREFIX)


def _rag_first_enabled() -> bool:
    return os.environ.get("AGENT_RAG_FIRST", "1").strip().lower() not in ("0", "false", "no", "off")


def _fast_path_enabled() -> bool:
    """Answer directly when the pre-search clearly matched, skipping routing.

    Off (`AGENT_RAG_FAST_PATH=0`) restores the previous behaviour: every
    question goes through the tool-bound model, whatever the overlap said.
    """
    return os.environ.get("AGENT_RAG_FAST_PATH", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


# --------------------------------------------------------------------------
# Is the automatic search actually about the question? (added 2026-09-18)
# --------------------------------------------------------------------------
# The pre-search note used to open with "MAY BE COMPLETELY IRRELEVANT ... if it
# does not answer the question, ignore it". That sentence is read on EVERY
# question, including the ones the knowledge base answers perfectly, and a 4b
# model took the invitation: asked something covered by the documents it would
# answer from its own weights and leave the retrieved passages unused.
#
# So the note is now conditional. When the search looks like it matched, the
# model is told plainly to answer from it; only when it does not are the
# documents waved off. The judgement below is deliberately not the ColBERT
# score: MaxSim scores are unnormalised sums over query tokens, so their range
# shifts with query length and a hardcoded threshold would be a guess. Content
# word overlap between the question and the CITATIONS (document names and
# section headings — never the generated answer, which echoes the question back
# and would score high even when nothing matched) needs no calibration and is
# obvious to debug from the log line it writes.
#
# It fails safe. A match it misses falls back to the old cautious wording, which
# is exactly the behaviour that shipped before.

_STOPWORDS = frozenset("""
a an and are as at be by can could do does for from give has have how i in into
is it its list me my of on or please should show so tell than that the their
then there these this to under was were what when where which who why will with
would you your about explain describe current right now
""".split())


def _content_words(text: str) -> set[str]:
    """Lowercase alphanumeric words that carry topic signal."""
    return {
        word
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if len(word) > 2 and word not in _STOPWORDS
    }


def prefetch_overlap(question: str, tool_output: str) -> float:
    """Fraction of the question's content words that appear in the citations.

    `rag_tool` formats its result as "<answer>\n\nSources:\n  <citations>", so
    everything after "Sources:" is document names and heading chains. 0.0 when
    there is nothing to compare — which routes to the cautious wording.
    """
    asked = _content_words(question)
    if not asked:
        return 0.0
    _, _, citations = tool_output.partition("Sources:")
    cited = _content_words(citations)
    if not cited:
        return 0.0
    return len(asked & cited) / len(asked)


# A third of the question's content words turning up in the headings. Tuned to
# separate "what is hybrid retrieval" (the headings say hybrid and retrieval)
# from "what is the weather in Delhi" (the headings say neither). Raise it if
# the agent starts trusting loose matches; the log line below reports the
# number for every question, so tune from real traffic rather than guessing.
RAG_PREFETCH_MATCH_MIN = float(os.environ.get("AGENT_RAG_MATCH_MIN", "0.34"))


# WMO weather interpretation codes, which is what Open-Meteo returns instead of
# a human-readable condition. https://open-meteo.com/en/docs
_WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    56: "Light freezing drizzle", 57: "Dense freezing drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Light freezing rain", 67: "Heavy freezing rain",
    71: "Slight snowfall", 73: "Moderate snowfall", 75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}


# Rewritten 2026-09-17 after the first version broke tool routing. It opened
# with "the knowledge base has ALREADY been searched ... base your answer on
# it", and a 4b model took that as "answer from the knowledge base or give up":
# asked for the weather in Prayagraj it replied "I don't have the current
# weather information" without ever calling `get_weather`.
#
# Two lessons are baked in below. Put the routing rules BEFORE the note about
# the pre-search, so the first thing read is what to call rather than what has
# already happened. And name the exact failure — "never tell the user you lack
# information before calling the tool that would provide it" — because a small
# model follows a concrete prohibition far better than a general principle.
AGENT_SYSTEM_PROMPT = SystemMessage(
    content="""You are a technical assistant with access to these tools:

- `get_weather` — current weather for a city.
- `sql_tool` — the query_history database: past questions, counts, latency, failures.
- `rag_tool` — the document knowledge base: machine learning, NLP, LLMs, RAG.

ROUTING — decide this first, every time:
- Asking about weather, temperature or conditions anywhere? Call `get_weather`.
- Asking about past queries, usage, latency, counts or failures? Call `sql_tool`.
- Asking about machine learning, NLP, LLMs or the documents? Use the knowledge
  base result already in the conversation.

NEVER tell the user you do not have information before calling the tool that
would provide it. If the question is about weather, call `get_weather` — do not
say you lack weather data. The same goes for every other tool.

A knowledge base search for this question has already been run automatically,
and its result appears above as a `rag_tool` result. A note attached to it says
whether the documents that came back match the question. Read that note first.

THE KNOWLEDGE BASE OUTRANKS YOUR OWN MEMORY. When its result covers the
question, answer FROM IT and cite its sources. Do not answer a machine
learning, NLP, LLM or RAG question from memory when the search result addresses
it — your memory is general, and the exact definitions, figures and names the
user is asking for are in those documents.

When the question is about weather or the query-history database, the automatic
search is unrelated to it — ignore the documents and call the tool that fits.

Do not offer to search the knowledge base; that has already happened. Never ask
the user whether they would like you to look something up — look it up, or
answer. You may call `rag_tool` again with different phrasing if you think the
automatic search missed.

When you answer from the knowledge base result, cite its sources."""
)

# Used when AGENT_RAG_FIRST=0 — the model is back in charge of reaching for the
# knowledge base, so it needs telling.
AGENT_SYSTEM_PROMPT_NO_PREFETCH = SystemMessage(
    content="""You are a technical assistant with access to specialized tools.
For any factual, technical, or conceptual questions about machine learning, NLP, or LLMs,
you MUST call `rag_tool` to check the knowledge base before answering.
Do not answer technical questions purely from your own memory."""
)


DIRECT_ANSWER_PROMPT = SystemMessage(
    content="""You are a technical assistant. Answer the question using ONLY the \
documents provided below.

- Cite the sources exactly as they appear in the documents.
- If the documents do not fully answer the question, say what is missing rather \
than filling the gap from memory.
- Be concise and direct. Do not describe your process or mention these instructions."""
)


class AgentState(TypedDict, total=False):
    session_id: str
    user_query: str
    start_time: float
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # Set by retrieve_first: did the pre-search actually match the question?
    # Read by the router to decide whether the model needs a turn at all.
    kb_matched: bool


# --------------------------------------------------------------------------
# Tools (Section 19)
# --------------------------------------------------------------------------

def build_tools(config: Config = CONFIG) -> list:
    """Construct the three tools. A function rather than module-level objects so
    that importing this module doesn't connect to Postgres or load the LLM."""
    import requests
    from langchain.agents import create_agent
    from langchain_community.agent_toolkits import SQLDatabaseToolkit
    from langchain_community.utilities import SQLDatabase

    llm = get_llm(config)

    @tool
    def rag_tool(query: str) -> str:
        """Searches the internal knowledge base containing research papers on Machine Learning,
        Foundation LLMs, Transformers, and Retrieval-Augmented Generation.
        ALWAYS invoke this tool for any questions regarding machine learning theory, NLP,
        prompting, or model architectures before generating an answer."""
        result = generate_answer(query, config=config)

        # Return only what the agent needs to reason and cite. Returning the whole
        # result dict would dump `context` (all retrieved chunks) AND `sources`
        # (the same chunk text again inside each payload) into the message
        # history — thousands of tokens per call, carried forward through every
        # subsequent turn of the agent loop, which overflows num_ctx fast.
        if result["degraded"]:
            return f"Knowledge base lookup degraded ({result.get('failure_stage')}): {result['answer']}"

        citations = "\n".join(f"  {format_source(meta)}" for meta in result["sources"])
        return f"{result['answer']}\n\nSources:\n{citations}"

    db = SQLDatabase.from_uri(config.pg_uri)
    sql_agent = create_agent(
        model=llm,
        tools=SQLDatabaseToolkit(db=db, llm=llm).get_tools(),
        system_prompt="You are a read-only SQL assistant querying the query_history table. Never write/alter data.",
    )

    @tool
    def sql_tool(query: str) -> str:
        """Query the PostgreSQL query_history database using natural language."""
        result = sql_agent.invoke({"messages": [{"role": "user", "content": query}]})
        return result["messages"][-1].content

    @tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city, town or place name."""
        # Open-Meteo, in two calls: name -> coordinates, then coordinates ->
        # current conditions. No API key and no account, which is what keeps
        # `.env` free of one more credential for anyone cloning the repo.
        #
        # Replaced wttr.in on 2026-09-18. That service is a scraper behind a
        # strict rate limit: it answers 200 with an HTML error page once you
        # have asked a few times in a row, so `.json()` raised and every
        # failure surfaced as the same opaque "Weather lookup failed".
        try:
            geo = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 1, "language": "en", "format": "json"},
                timeout=10,
            )
            geo.raise_for_status()
            places = geo.json().get("results") or []
            if not places:
                # A real answer, not an error. The model can relay "no such
                # place" to the user; it cannot do anything with a traceback.
                return f"No place found matching '{city}'. Check the spelling, or try adding the country."
            place = places[0]

            forecast = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    # timezone=auto makes `time` the local clock at that place
                    # rather than UTC, which is what "right now" means to
                    # whoever asked.
                    "timezone": "auto",
                    "current": ",".join([
                        "temperature_2m", "apparent_temperature", "relative_humidity_2m",
                        "wind_speed_10m", "precipitation", "weather_code",
                    ]),
                },
                timeout=10,
            )
            forecast.raise_for_status()
            current = forecast.json()["current"]
        except Exception as exc:  # noqa: BLE001
            return f"Weather lookup failed: {type(exc).__name__}: {exc}"

        where = ", ".join(
            part for part in (place.get("name"), place.get("admin1"), place.get("country"))
            if part
        )

        # Built with .get() and skipped when absent, rather than indexed. A
        # KeyError here would escape the try block above and propagate out of
        # the tool into the graph — one renamed upstream field would turn a
        # weather question into a 500 instead of a partial answer.
        parts = [f"Temperature {current['temperature_2m']}°C"] if "temperature_2m" in current else []
        readings = [
            ("apparent_temperature", "feels like {}°C"),
            ("relative_humidity_2m", "humidity {}%"),
            ("wind_speed_10m", "wind {} km/h"),
            ("precipitation", "precipitation {} mm"),
        ]
        parts += [
            template.format(current[key]) for key, template in readings if key in current
        ]

        condition = _WMO_CODES.get(current.get("weather_code"), "Conditions unavailable")
        observed = f" Observed {current['time']} local time." if current.get("time") else ""

        if not parts:
            return f"{where} — no current readings were returned for this location."
        return f"{where} — {condition}. " + ", ".join(parts) + f".{observed}"

    return [rag_tool, sql_tool, get_weather]


# --------------------------------------------------------------------------
# Graph (Section 20)
# --------------------------------------------------------------------------

_agent = None


def build_agent(config: Config = CONFIG):
    """Compile the LangGraph agent, once per process. Building it constructs
    the SQL toolkit, which connects to Postgres, so you want that to happen
    exactly once. A plain singleton — @lru_cache cannot key on Config."""
    global _agent
    if _agent is not None:
        return _agent

    from langgraph.graph import END, START, StateGraph
    from langgraph.prebuilt import ToolNode

    tools = build_tools(config)
    tools_by_name = {t.name: t for t in tools}
    llm_with_tools = get_llm(config).bind_tools(tools)

    plain_llm = get_llm(config)          # no tools bound — see direct_answer_node
    rag_first = _rag_first_enabled()
    fast_path = rag_first and _fast_path_enabled()
    system_prompt = AGENT_SYSTEM_PROMPT if rag_first else AGENT_SYSTEM_PROMPT_NO_PREFETCH

    def retrieve_first_node(state: AgentState):
        """Search the knowledge base before the model gets a turn.

        The result is injected as a genuine `rag_tool` exchange — an AIMessage
        carrying the tool call, then the ToolMessage holding its output. Shaping
        it that way rather than dumping the text into a system message means
        everything downstream that counts tool usage sees the truth: the tool
        really did run, so `log_to_db_node`, the API's `tools_used`, and
        MLflow's TOOL spans all record it without special-casing.

        A failure here is not fatal. If retrieval is down the agent should still
        answer what it can with its other tools, so the error becomes the tool
        result and the model reads it like any other.
        """
        question = state["user_query"]
        call_id = f"{PREFETCH_CALL_PREFIX}{uuid.uuid4().hex[:8]}"

        try:
            output = tools_by_name["rag_tool"].invoke({"query": question})
        except Exception as exc:  # noqa: BLE001
            logger.error(f"retrieve_first: rag_tool failed -- {exc!r}")
            output = f"Knowledge base lookup failed: {exc}"

        # Labelled, because the search was automatic rather than chosen. A
        # hybrid retriever always returns its top k — there is no "no match" —
        # so for a weather question it hands back documents about transformers
        # with no hint that they are unrelated. Saying so in the message is
        # cheaper than hoping a 4b model works it out.
        #
        # The label is conditional (see prefetch_overlap above). Telling the
        # model the documents "may be unrelated" on every single question is
        # what let it skip a knowledge base that had the answer.
        overlap = prefetch_overlap(question, str(output))
        matched = overlap >= RAG_PREFETCH_MATCH_MIN
        logger.info(
            f"retrieve_first: citation overlap {overlap:.2f} "
            f"(threshold {RAG_PREFETCH_MATCH_MIN:.2f}, matched={matched})"
        )

        if matched:
            note = (
                "[These documents are about the subject of the question. Answer from "
                "them and cite the sources — do not answer from your own memory "
                "instead.]"
            )
        else:
            note = (
                "[These are the closest documents found, but they do not obviously "
                "match the question. If the question is about weather or the query "
                "history, they are unrelated — ignore them and call the tool that "
                "fits. If they do answer the question, use them and cite the sources.]"
            )

        output = f"[Automatic knowledge-base search for: {question}]\n{note}\n\n{output}"

        return {
            "kb_matched": matched,
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "rag_tool",
                        "args": {"query": question},
                        "id": call_id,
                        "type": "tool_call",
                    }],
                ),
                ToolMessage(content=str(output), tool_call_id=call_id, name="rag_tool"),
            ]
        }

    def direct_answer_node(state: AgentState):
        """Answer straight from the pre-searched documents. One LLM call, no tools.

        Why this exists (2026-09-20): `/agent` was taking ~20s, and the cause
        was turn count rather than a slow model. On a question the knowledge
        base plainly answers, the tool-bound model still had to take a routing
        turn, and often called `rag_tool` a second time — two or three
        generations for a question whose answer was already retrieved before
        the model was consulted at all.

        `prefetch_overlap` already decides, with no model call, whether the
        documents are about the question. When they clearly are, the routing
        turn is a generation whose outcome is known in advance.

        Two savings, not one. The obvious one is fewer turns. The quieter one
        is that this uses the PLAIN llm rather than `llm_with_tools`: binding
        tools injects all three schemas into the prompt on every turn, which on
        a 4b model is real prefill cost paid for nothing here.

        Cost, stated honestly: a compound question ("explain hybrid retrieval,
        and what is the weather in Delhi") that scores above the overlap
        threshold will be answered from documents alone and never reach
        `get_weather`. `AGENT_RAG_FAST_PATH=0` turns this off.
        """
        context = ""
        for message in reversed(state["messages"]):
            if isinstance(message, ToolMessage):
                context = str(message.content)
                break

        # A clean two-message prompt rather than replaying the tool-call
        # exchange: an unbound model has no business being handed an AIMessage
        # carrying tool_calls, and the shorter prompt is the point of the
        # exercise.
        answer = plain_llm.invoke([
            DIRECT_ANSWER_PROMPT,
            HumanMessage(content=f"{context}\n\nQuestion: {state['user_query']}"),
        ])
        return {"messages": [answer]}

    def route_after_prefetch(state: AgentState):
        if fast_path and state.get("kb_matched"):
            logger.info("retrieve_first: fast path — answering directly, no routing turn")
            return "direct_answer"
        return "agent"

    def agent_node(state: AgentState):
        # Prepend the system prompt each turn so the routing instruction never
        # scrolls out of the model's attention as the message history grows.
        messages = [system_prompt] + list(state["messages"])
        return {"messages": [llm_with_tools.invoke(messages)]}

    def log_to_db_node(state: AgentState):
        """Log the final output and tool usage to Postgres."""
        latency_ms = (time.time() - state["start_time"]) * 1000
        content = state["messages"][-1].content
        final_message = content if isinstance(content, str) else str(content)

        # Only what the model chose. The automatic pre-search is excluded, so
        # this column keeps meaning what it always meant — which tool the agent
        # decided to reach for.
        used_tools = set()
        for msg in state["messages"]:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    if not is_prefetch_call(tc):
                        used_tools.add(tc["name"])

        # Summed over every model turn in the graph (routing + final answer).
        # Not included: the SQL sub-agent's own calls inside sql_tool, which
        # run in a separate graph whose messages never reach this state.
        usage = llm_usage(state["messages"])

        save_query_history(
            session_id=state.get("session_id", "default"),
            user_query=state["user_query"],
            generated_answer=final_message,
            route="agent",
            tool_used=",".join(sorted(used_tools)) if used_tools else "none",
            latency_ms=latency_ms,
            success=True,
            tokens_used=usage["total_tokens"] or None,
            tokens_per_sec=usage["tokens_per_sec"],
            total_tokens_per_sec=total_tokens_per_sec(usage["total_tokens"], latency_ms / 1000),
            config=config,
        )
        return {}

    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return "log_to_db"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("log_to_db", log_to_db_node)

    if rag_first:
        graph.add_node("retrieve_first", retrieve_first_node)
        graph.add_edge(START, "retrieve_first")
        if fast_path:
            graph.add_node("direct_answer", direct_answer_node)
            graph.add_conditional_edges(
                "retrieve_first",
                route_after_prefetch,
                {"direct_answer": "direct_answer", "agent": "agent"},
            )
            # Straight to logging: nothing after a direct answer can call a tool.
            graph.add_edge("direct_answer", "log_to_db")
        else:
            graph.add_edge("retrieve_first", "agent")
    else:
        graph.add_edge(START, "agent")

    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "log_to_db": "log_to_db"})
    graph.add_edge("tools", "agent")
    graph.add_edge("log_to_db", END)

    _agent = graph.compile()
    logger.info(f"Agent graph compiled. rag_first={rag_first} fast_path={fast_path}")
    return _agent


def ask_agent(question: str, session_id: str = "default", config: Config = CONFIG) -> dict:
    """Convenience wrapper — invoke the agent on one question."""
    from langchain_core.messages import HumanMessage

    return build_agent(config).invoke({
        "session_id": session_id,
        "user_query": question,
        "start_time": time.time(),
        "messages": [HumanMessage(content=question)],
    })
