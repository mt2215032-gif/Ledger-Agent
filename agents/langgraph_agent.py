"""LangGraph state machine with a real tool-calling cycle.

The graph is agent -> tools -> agent -> ... -> END. The cycle is the point: a
graph wired agent -> END is a single model call with extra ceremony, and cannot
do multi-step reasoning no matter what the state schema says.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from agents.provider import get_chat_model
from agents.tools import TOOL_DESCRIPTION, execute_project_query

SYSTEM_PROMPT = (
    "You are a ledger analyst. Use the execute_project_query tool to retrieve "
    "figures before answering, and quote what it returns rather than estimating."
)


class AgentState(TypedDict):
    """Conversation state. `operator.add` appends each turn instead of replacing."""

    messages: Annotated[list[AnyMessage], operator.add]


@tool
def project_query(query: str) -> str:
    """Run a retrieval query against the project ledger.

    Args:
        query: One of totals, findings, by_date, by_account, all.
    """
    try:
        return execute_project_query(query)
    except ValueError as exc:
        # Returned, not raised: the model can correct itself from an error
        # string, whereas an exception tears down the whole graph run.
        return f"Tool error: {exc}"


project_query.description = TOOL_DESCRIPTION

TOOLS = [project_query]


def build_graph(model: Any = None, provider: str | None = None):
    """Compile the agent/tools cycle.

    `model` is injectable so the graph can be exercised without credentials.
    """
    llm = model if model is not None else get_chat_model(provider)
    llm_with_tools = llm.bind_tools(TOOLS)

    def call_model(state: AgentState) -> dict[str, list[AnyMessage]]:
        messages = state["messages"]
        # Prepend the system turn once, rather than on every cycle.
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]
        return {"messages": [llm_with_tools.invoke(messages)]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(TOOLS))

    workflow.add_edge(START, "agent")
    # tools_condition routes to "tools" when the last message has tool calls,
    # and to END when it does not. This edge is what makes the graph cyclic.
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    return workflow.compile()


def run(question: str, *, model: Any = None, provider: str | None = None) -> str:
    """Run the graph to completion and return the final answer."""
    graph = build_graph(model=model, provider=provider)
    final = graph.invoke({"messages": [HumanMessage(content=question)]})
    return final["messages"][-1].content


if __name__ == "__main__":
    import sys

    from agents.provider import load_dotenv

    load_dotenv()
    print(run(" ".join(sys.argv[1:]) or "Does the ledger balance? Explain briefly."))
