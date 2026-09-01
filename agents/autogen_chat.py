"""AutoGen conversable multi-agent setup, provider-switchable.

Two notes on the classic template this is based on.

Package: the conversable API (`AssistantAgent`, `UserProxyAgent`) lives in the
`autogen` distribution. `ag2` 1.x is a ground-up rewrite that does not export
these names at all, so `pip install ag2` gives an ImportError here.

Safety: the original passes `code_execution_config={"use_docker": False}`, which
executes model-written code directly on the host with no sandbox. That is off by
default here. The agent gets a registered ledger tool instead, which is enough
for this task and cannot run arbitrary code.
"""

from __future__ import annotations

from typing import Any

from autogen import AssistantAgent, UserProxyAgent

from agents.provider import CLAUDE, get_provider, require_key
from agents.tools import execute_project_query

# AutoGen's api_type strings per provider.
API_TYPE = {CLAUDE: "anthropic", "openai": "openai"}

SYSTEM_MESSAGE = (
    "You are a ledger analyst. Use the execute_project_query tool to retrieve "
    "figures before answering, and quote exactly what it returns. "
    "Reply TERMINATE when the question is fully answered."
)


def build_llm_config(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.3,
) -> dict[str, Any]:
    """Build AutoGen's llm_config for the active provider."""
    config = get_provider(provider, model)
    require_key(config)
    return {
        "config_list": [
            {
                "model": config.model,
                "api_key": config.api_key,
                "api_type": API_TYPE[config.name],
            }
        ],
        "temperature": temperature,
    }


def build_agents(
    llm_config: dict[str, Any] | None = None,
    provider: str | None = None,
    *,
    max_turns: int = 5,
    enable_code_execution: bool = False,
) -> tuple[AssistantAgent, UserProxyAgent]:
    """Create the assistant/proxy pair with the ledger tool registered.

    Set `enable_code_execution=True` only if you understand that the proxy will
    then run model-authored code on this machine.
    """
    config = llm_config if llm_config is not None else build_llm_config(provider)

    assistant = AssistantAgent(
        name="Ledger_Analyst",
        system_message=SYSTEM_MESSAGE,
        llm_config=config,
    )

    code_config: Any = False
    if enable_code_execution:
        code_config = {"work_dir": "workspace", "use_docker": False}

    user_proxy = UserProxyAgent(
        name="Execution_Proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=max_turns,
        is_termination_msg=lambda m: "TERMINATE" in (m.get("content") or ""),
        code_execution_config=code_config,
    )

    # Register the tool on both sides: the assistant advertises the schema, the
    # proxy is what actually executes it.
    @user_proxy.register_for_execution()
    @assistant.register_for_llm(
        name="execute_project_query",
        description=(
            "Query the project ledger. One of: totals, findings, by_date, "
            "by_account, all."
        ),
    )
    def _query(query: str) -> str:
        try:
            return execute_project_query(query)
        except ValueError as exc:
            return f"Tool error: {exc}"

    return assistant, user_proxy


def run(message: str, provider: str | None = None, **kwargs: Any) -> Any:
    """Start the chat and return the resulting conversation."""
    assistant, user_proxy = build_agents(provider=provider, **kwargs)
    return user_proxy.initiate_chat(assistant, message=message)


if __name__ == "__main__":
    import sys

    from agents.provider import load_dotenv

    load_dotenv()
    run(" ".join(sys.argv[1:]) or "Does the ledger balance? Use the tool.")
