"""CrewAI collaborative team backed by Claude or OpenAI.

CrewAI 1.x does not accept a LangChain chat model on `Agent.llm` -- the field is
typed `str | BaseLLM | None`, so passing `ChatAnthropic(...)` raises a pydantic
ValidationError before any model call. Models are selected here with CrewAI's
own `LLM` class and a provider-prefixed model id.
"""

from __future__ import annotations

from typing import Any

from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import tool

from agents.provider import CLAUDE, get_provider, require_key
from agents.tools import execute_project_query

# CrewAI routes through litellm, which wants "<provider>/<model>".
PROVIDER_PREFIX = {CLAUDE: "anthropic", "openai": "openai"}


def get_crew_llm(provider: str | None = None, model: str | None = None, temperature: float = 0.2) -> LLM:
    """Build a CrewAI LLM for the active provider."""
    config = get_provider(provider, model)
    require_key(config)
    return LLM(
        model=f"{PROVIDER_PREFIX[config.name]}/{config.model}",
        temperature=temperature,
    )


@tool("Ledger query")
def ledger_query(query: str) -> str:
    """Query the project ledger. One of: totals, findings, by_date, by_account, all."""
    try:
        return execute_project_query(query)
    except ValueError as exc:
        return f"Tool error: {exc}"


def build_crew(llm: Any = None, provider: str | None = None, verbose: bool = True) -> Crew:
    """Assemble the research -> write crew.

    `llm` is injectable so the crew can be constructed and inspected in tests
    without credentials.
    """
    model = llm if llm is not None else get_crew_llm(provider)

    researcher = Agent(
        role="Principal Research Analyst",
        goal="Gather and analyse the ledger's reconciliation status and exceptions",
        backstory=(
            "An expert investigator who extracts figures from accounting systems "
            "and summarises what they mean, without ever estimating a number."
        ),
        llm=model,
        tools=[ledger_query],
        verbose=verbose,
    )

    writer = Agent(
        role="Technical Lead & Documentation Writer",
        goal="Turn the research into a clean reconciliation deliverable",
        backstory=(
            "A meticulous writer who translates raw reconciliation output into "
            "notes an accountant can act on."
        ),
        llm=model,
        verbose=verbose,
    )

    task_research = Task(
        description=(
            "Use the ledger query tool to retrieve the totals and the automated "
            "findings. Report the exact figures returned; do not compute your own."
        ),
        expected_output="A structured markdown breakdown of totals and exceptions.",
        agent=researcher,
    )

    task_write = Task(
        description=(
            "Write reconciliation notes from the research findings: summary, "
            "exceptions with row references, and proposed correcting entries."
        ),
        expected_output="Reconciliation notes in markdown.",
        agent=writer,
        context=[task_research],
    )

    return Crew(
        agents=[researcher, writer],
        tasks=[task_research, task_write],
        process=Process.sequential,
        verbose=verbose,
    )


def run(provider: str | None = None) -> str:
    """Execute the crew and return its final output."""
    return str(build_crew(provider=provider).kickoff())


if __name__ == "__main__":
    from agents.provider import load_dotenv

    load_dotenv()
    print(run())
