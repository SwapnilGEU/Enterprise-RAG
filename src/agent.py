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
every tool available afterwards, so it can follow up with SQL, arithmetic or
weather — it simply no longer gets to skip the knowledge base.

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
import time
import uuid
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from src.config import CONFIG, Config, logger
from src.generation import generate_answer, get_llm
from src.history import save_query_history
from src.payload import format_source


def _rag_first_enabled() -> bool:
    return os.environ.get("AGENT_RAG_FIRST", "1").strip().lower() not in ("0", "false", "no", "off")


AGENT_SYSTEM_PROMPT = SystemMessage(
    content="""You are a technical assistant with access to specialized tools.

The knowledge base has ALREADY been searched for this question, and the result
is in the conversation above as a `rag_tool` result. Base your answer on it.

If it answers the question, answer from it and cite its sources. Do not offer
to search the knowledge base — that has happened. Never ask the user whether
they would like you to look something up; either look it up or answer.

If it does not answer the question, say so plainly and then use whichever other
tool fits: `sql_tool` for the query history, `calculator` for arithmetic,
`get_weather` for weather. You may call `rag_tool` again with a different
search phrasing if you think the first one missed.

Do not answer technical questions from your own memory when the knowledge base
result covers them."""
)

# Used when AGENT_RAG_FIRST=0 — the model is back in charge of reaching for the
# knowledge base, so it needs telling.
AGENT_SYSTEM_PROMPT_NO_PREFETCH = SystemMessage(
    content="""You are a technical assistant with access to specialized tools.
For any factual, technical, or conceptual questions about machine learning, NLP, or LLMs,
you MUST call `rag_tool` to check the knowledge base before answering.
Do not answer technical questions purely from your own memory."""
)


class AgentState(TypedDict):
    session_id: str
    user_query: str
    start_time: float
    messages: Annotated[Sequence[BaseMessage], operator.add]


# --------------------------------------------------------------------------
# Tools (Section 19)
# --------------------------------------------------------------------------

def build_tools(config: Config = CONFIG) -> list:
    """Construct the four tools. A function rather than module-level objects so
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
    def calculator(expression: str) -> str:
        """Perform mathematical calculations from a valid arithmetic expression."""
        try:
            return str(eval(expression, {"__builtins__": {}}, {}))
        except Exception as exc:
            return f"Calculation error: {exc}"

    @tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        try:
            response = requests.get(f"https://wttr.in/{city}?format=j1", timeout=10)
            data = response.json()["current_condition"][0]
            return f"Temperature: {data['temp_C']}°C, Condition: {data['weatherDesc'][0]['value']}"
        except Exception as exc:
            return f"Weather lookup failed: {exc}"

    return [rag_tool, sql_tool, calculator, get_weather]


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

    rag_first = _rag_first_enabled()
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
        call_id = f"ragfirst-{uuid.uuid4().hex[:8]}"

        try:
            output = tools_by_name["rag_tool"].invoke({"query": question})
        except Exception as exc:  # noqa: BLE001
            logger.error(f"retrieve_first: rag_tool failed -- {exc!r}")
            output = f"Knowledge base lookup failed: {exc}"

        return {
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

        used_tools = set()
        for msg in state["messages"]:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    used_tools.add(tc["name"])

        save_query_history(
            session_id=state.get("session_id", "default"),
            user_query=state["user_query"],
            generated_answer=final_message,
            route="agent",
            tool_used=",".join(sorted(used_tools)) if used_tools else "none",
            latency_ms=latency_ms,
            success=True,
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
        graph.add_edge("retrieve_first", "agent")
    else:
        graph.add_edge(START, "agent")

    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "log_to_db": "log_to_db"})
    graph.add_edge("tools", "agent")
    graph.add_edge("log_to_db", END)

    _agent = graph.compile()
    logger.info(f"Agent graph compiled. rag_first={rag_first}")
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
