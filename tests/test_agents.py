"""Tests for the four agent-framework templates.

No API key is needed. Loops are driven with stub clients that reproduce the
real response shapes, which is what lets the parallel-tool-use and stop_reason
paths be tested at all -- those are exactly the cases a happy-path live call
never exercises.
"""

from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest
from langchain_core.messages import AIMessage

from agents import autogen_chat, crewai_team, langgraph_agent, native_loop, tools
from agents.native_loop import (
    MaxIterationsExceeded,
    OutputTruncated,
    run_claude_agent,
    run_openai_agent,
)
from agents.provider import (
    DEFAULT_MODELS,
    REGISTRY,
    MissingAPIKeyError,
    get_chat_model,
    get_provider,
    list_providers,
)


def block(**kw):
    return NS(**kw)


# ------------------------------------------------------------- provider


def test_claude_is_the_default_provider(monkeypatch):
    monkeypatch.delenv("PROVIDER", raising=False)
    monkeypatch.delenv("MODEL", raising=False)
    assert get_provider().name == "claude"


def test_default_models_are_current():
    # The pasted templates pinned claude-3-5-sonnet-20241022, several
    # generations behind; guard against it creeping back.
    assert DEFAULT_MODELS["claude"] == "claude-opus-5"
    assert "claude-3-5" not in DEFAULT_MODELS["claude"]


def test_provider_switches_from_the_environment(monkeypatch):
    monkeypatch.setenv("PROVIDER", "openai")
    monkeypatch.delenv("MODEL", raising=False)
    assert get_provider().model == "gpt-4o-mini"


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("no-such-provider")


def test_missing_key_raises_before_any_network_call(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError, match="ANTHROPIC_API_KEY"):
        get_chat_model("claude")


# ------------------------------------------------------------- registry


def test_both_proprietary_and_open_weight_sources_are_registered():
    assert list_providers(open_weights=False)  # claude, openai, gemini
    assert list_providers(open_weights=True)   # ollama, groq, together, ...
    assert len(REGISTRY) == len(list_providers())


@pytest.mark.parametrize("key", sorted(REGISTRY))
def test_every_provider_has_a_usable_langchain_integration(key, monkeypatch):
    """Each entry must name a real module and class, not a guessed one."""
    spec = REGISTRY[key]
    if spec.api_key_env:
        monkeypatch.setenv(spec.api_key_env, "test-key")
    model = get_chat_model(key)
    assert type(model).__name__ == spec.langchain_class


@pytest.mark.parametrize("key", sorted(REGISTRY))
def test_every_provider_declares_framework_routing(key):
    spec = REGISTRY[key]
    assert spec.default_model
    assert spec.pip_package
    # Claude is the one provider deliberately not routed through an
    # OpenAI-compatible endpoint; it has a first-class SDK.
    if key != "claude":
        assert spec.openai_base_url, f"{key} needs a base_url for the native loop"


def test_keyless_local_provider_needs_no_credentials(monkeypatch):
    """Ollama runs locally, so it is configured without any API key."""
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    config = get_provider("ollama")
    assert config.needs_key is False
    assert config.configured is True
    assert config.api_key is None


def test_keyed_provider_is_unconfigured_without_its_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert get_provider("groq").configured is False


def test_model_override_beats_the_registry_default(monkeypatch):
    monkeypatch.delenv("MODEL", raising=False)
    assert get_provider("groq", "mixtral-8x7b").model == "mixtral-8x7b"


# ---------------------------------------------------------------- tools


def test_tool_returns_real_reconciliation_figures():
    assert json.loads(tools.execute_project_query("totals")) == {
        "total_debits": 700.0,
        "total_credits": 2000.0,
        "difference": -1300.0,
        "is_balanced": False,
    }


def test_unknown_query_raises():
    with pytest.raises(ValueError, match="Unknown query"):
        tools.execute_project_query("drop table")


def test_both_provider_schemas_share_one_definition():
    anthropic = tools.anthropic_tools()[0]
    openai = tools.openai_tools()[0]["function"]
    assert anthropic["input_schema"] is openai["parameters"]
    assert anthropic["input_schema"]["additionalProperties"] is False


# ------------------------------------------------- native loop: Claude


class StubClaude:
    """Replays a scripted list of responses and records what was sent."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    @property
    def messages(self):
        return self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        index = min(len(self.requests) - 1, len(self.responses) - 1)
        return self.responses[index]


def test_every_parallel_tool_use_block_is_answered_in_one_message():
    """Claude may emit several tool_use blocks per turn; all need results."""
    client = StubClaude(
        [
            NS(
                stop_reason="tool_use",
                content=[
                    block(type="thinking", thinking="..."),
                    block(type="tool_use", id="t1", name=tools.TOOL_NAME,
                          input={"query": "totals"}),
                    block(type="tool_use", id="t2", name=tools.TOOL_NAME,
                          input={"query": "findings"}),
                ],
            ),
            NS(stop_reason="end_turn", content=[block(type="text", text="done")]),
        ]
    )
    result = run_claude_agent("q", client=client)

    followup = client.requests[1]["messages"][2]
    assert followup["role"] == "user"
    assert [b["tool_use_id"] for b in followup["content"]] == ["t1", "t2"]
    assert all(b["type"] == "tool_result" for b in followup["content"])
    assert result.iterations == 2
    assert len(result.tool_calls) == 2


def test_thinking_block_does_not_break_text_extraction():
    """`content[0].text` crashes here; the text block is not first."""
    client = StubClaude(
        [
            NS(
                stop_reason="end_turn",
                content=[
                    block(type="thinking", thinking="reasoning"),
                    block(type="text", text="the answer"),
                ],
            )
        ]
    )
    assert run_claude_agent("q", client=client).text == "the answer"


def test_assistant_turn_is_echoed_back_verbatim():
    """Rebuilding it from text would drop tool_use and thinking blocks."""
    original = [
        block(type="thinking", thinking="..."),
        block(type="tool_use", id="t1", name=tools.TOOL_NAME, input={"query": "totals"}),
    ]
    client = StubClaude(
        [
            NS(stop_reason="tool_use", content=original),
            NS(stop_reason="end_turn", content=[block(type="text", text="ok")]),
        ]
    )
    run_claude_agent("q", client=client)
    assert client.requests[1]["messages"][1]["content"] is original


def test_tool_failure_is_reported_as_an_error_result():
    """A dropped tool_result is a hard API error, so failures must be sent."""
    client = StubClaude(
        [
            NS(
                stop_reason="tool_use",
                content=[
                    block(type="tool_use", id="t1", name=tools.TOOL_NAME,
                          input={"query": "nonsense"})
                ],
            ),
            NS(stop_reason="end_turn", content=[block(type="text", text="recovered")]),
        ]
    )
    result = run_claude_agent("q", client=client)
    sent = client.requests[1]["messages"][2]["content"][0]
    assert sent["is_error"] is True
    assert "Tool error" in sent["content"]
    assert result.text == "recovered"


@pytest.mark.parametrize("stop_reason", ["end_turn", "stop_sequence"])
def test_terminal_stop_reasons_return(stop_reason):
    client = StubClaude(
        [NS(stop_reason=stop_reason, content=[block(type="text", text="x")])]
    )
    assert run_claude_agent("q", client=client).stop_reason == stop_reason


def test_max_tokens_raises_rather_than_returning_a_truncated_answer():
    client = StubClaude(
        [NS(stop_reason="max_tokens", content=[block(type="text", text="half an ans")])]
    )
    with pytest.raises(OutputTruncated):
        run_claude_agent("q", client=client)


def test_truncation_can_be_accepted_explicitly():
    client = StubClaude(
        [NS(stop_reason="max_tokens", content=[block(type="text", text="half")])]
    )
    result = run_claude_agent("q", client=client, allow_truncated=True)
    assert result.stop_reason == "max_tokens"


def test_refusal_returns_and_is_flagged_not_looped_on():
    client = StubClaude(
        [NS(stop_reason="refusal", content=[block(type="text", text="I can't")])]
    )
    result = run_claude_agent("q", client=client)
    assert result.refused


def test_runaway_tool_use_is_capped():
    """An unbounded while-loop against a paid API is the expensive bug."""
    forever = NS(
        stop_reason="tool_use",
        content=[block(type="tool_use", id="t", name=tools.TOOL_NAME,
                       input={"query": "totals"})],
    )
    with pytest.raises(MaxIterationsExceeded):
        run_claude_agent("q", client=StubClaude([forever]), max_iterations=3)


def test_adaptive_thinking_is_requested_not_budget_tokens():
    client = StubClaude(
        [NS(stop_reason="end_turn", content=[block(type="text", text="x")])]
    )
    run_claude_agent("q", client=client)
    thinking = client.requests[0]["thinking"]
    assert thinking == {"type": "adaptive"}
    assert "budget_tokens" not in thinking  # rejected on current models


# ------------------------------------------------- native loop: OpenAI


class StubOpenAI:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses[min(len(self.requests) - 1, len(self.responses) - 1)]


def _openai_response(finish_reason, *, content=None, tool_calls=None):
    message = NS(content=content, tool_calls=tool_calls or [])
    return NS(choices=[NS(finish_reason=finish_reason, message=message)])


def test_openai_loop_answers_every_tool_call():
    call1 = NS(id="c1", function=NS(name=tools.TOOL_NAME,
                                    arguments='{"query": "totals"}'))
    call2 = NS(id="c2", function=NS(name=tools.TOOL_NAME,
                                    arguments='{"query": "findings"}'))
    client = StubOpenAI(
        [
            _openai_response("tool_calls", tool_calls=[call1, call2]),
            _openai_response("stop", content="done"),
        ]
    )
    result = run_openai_agent("q", client=client)

    tool_messages = [m for m in client.requests[1]["messages"]
                     if isinstance(m, dict) and m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["c1", "c2"]
    assert result.text == "done"
    assert len(result.tool_calls) == 2


def test_openai_length_finish_raises():
    client = StubOpenAI([_openai_response("length", content="half")])
    with pytest.raises(OutputTruncated):
        run_openai_agent("q", client=client)


def test_openai_content_filter_is_flagged_as_refusal():
    client = StubOpenAI([_openai_response("content_filter", content="")])
    assert run_openai_agent("q", client=client).refused


# ------------------------------------------------------------ LangGraph


class StubChatModel:
    """Duck-typed chat model: calls the tool once, then answers."""

    def __init__(self):
        self.calls = 0
        self.seen = []

    def bind_tools(self, tools_):
        self.bound = tools_
        return self

    def invoke(self, messages):
        self.calls += 1
        self.seen.append(messages)
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[{"name": "project_query",
                             "args": {"query": "totals"}, "id": "c1"}],
            )
        return AIMessage(content="out of balance by 1,300.00")


def test_graph_has_a_real_agent_tools_cycle():
    """The pasted template wired agent -> END, which cannot loop at all."""
    graph = langgraph_agent.build_graph(model=StubChatModel())
    edges = {(e.source, e.target) for e in graph.get_graph().edges}
    assert ("agent", "tools") in edges
    assert ("tools", "agent") in edges


def test_graph_runs_the_cycle_and_feeds_tool_output_back():
    model = StubChatModel()
    answer = langgraph_agent.run("does it balance?", model=model)
    assert answer == "out of balance by 1,300.00"
    assert model.calls == 2  # agent -> tools -> agent
    tool_messages = [m for m in model.seen[-1]
                     if m.__class__.__name__ == "ToolMessage"]
    assert tool_messages and "700.0" in tool_messages[0].content


# --------------------------------------------------------------- CrewAI


def test_crew_builds_with_crewai_native_llm():
    from crewai import LLM

    crew = crewai_team.build_crew(llm=LLM(model="anthropic/claude-opus-5"),
                                  verbose=False)
    assert len(crew.agents) == 2
    assert len(crew.tasks) == 2
    assert crew.agents[0].llm.model == "claude-opus-5"


def test_crewai_rejects_a_langchain_model():
    """Why build_crew uses crewai.LLM: Agent.llm is str | BaseLLM | None."""
    pydantic = pytest.importorskip("pydantic")
    from crewai import Agent
    from langchain_anthropic import ChatAnthropic

    with pytest.raises(pydantic.ValidationError):
        Agent(role="r", goal="g", backstory="b",
              llm=ChatAnthropic(model="claude-opus-5", api_key="x"))


def test_crew_researcher_has_the_ledger_tool():
    from crewai import LLM

    crew = crewai_team.build_crew(llm=LLM(model="anthropic/claude-opus-5"),
                                  verbose=False)
    assert [t.name for t in crew.agents[0].tools] == ["Ledger query"]


# -------------------------------------------------------------- AutoGen


AUTOGEN_CONFIG = {
    "config_list": [{"model": "claude-opus-5", "api_key": "x",
                     "api_type": "anthropic"}],
    "temperature": 0.3,
}


def test_autogen_registers_the_tool_on_both_agents():
    assistant, proxy = autogen_chat.build_agents(llm_config=AUTOGEN_CONFIG)
    advertised = [t["function"]["name"] for t in assistant.llm_config["tools"]]
    assert tools.TOOL_NAME in advertised          # model can see it
    assert tools.TOOL_NAME in proxy.function_map  # proxy can run it


def test_autogen_code_execution_is_off_by_default():
    """The pasted template runs model-written code on the host unsandboxed."""
    _, proxy = autogen_chat.build_agents(llm_config=AUTOGEN_CONFIG)
    assert proxy._code_execution_config is False


def test_autogen_code_execution_is_opt_in():
    _, proxy = autogen_chat.build_agents(llm_config=AUTOGEN_CONFIG,
                                         enable_code_execution=True)
    assert proxy._code_execution_config["work_dir"] == "workspace"


def test_autogen_tool_executes_against_the_ledger():
    _, proxy = autogen_chat.build_agents(llm_config=AUTOGEN_CONFIG)
    out = json.loads(proxy.function_map[tools.TOOL_NAME](query="totals"))
    assert out["total_debits"] == 700.0


# ------------------------------------------- open-weight provider routing


def test_claude_is_refused_by_the_openai_compatible_loop():
    """Routing Claude through an OpenAI-shaped loop would lose its semantics."""
    with pytest.raises(ValueError, match="run_claude_agent"):
        run_openai_agent("q", provider="claude")


@pytest.mark.parametrize(
    "provider, expected_host",
    [
        ("groq", "api.groq.com"),
        ("together", "api.together.xyz"),
        ("mistral", "api.mistral.ai"),
        ("deepseek", "api.deepseek.com"),
        ("cerebras", "api.cerebras.ai"),
        ("ollama", "localhost:11434"),
    ],
)
def test_open_weight_providers_route_to_their_own_endpoint(provider, expected_host):
    assert expected_host in REGISTRY[provider].openai_base_url


def test_open_weight_provider_runs_through_the_shared_loop():
    """One OpenAI-compatible loop serves every open-weight host."""
    client = StubOpenAI([_openai_response("stop", content="llama says hi")])
    result = run_openai_agent("q", client=client, provider="groq")
    assert result.text == "llama says hi"
    assert client.requests[0]["model"] == REGISTRY["groq"].default_model


def test_autogen_omits_api_key_for_keyless_providers():
    entry = autogen_chat.build_llm_config("ollama")["config_list"][0]
    assert entry["api_type"] == "ollama"
    assert "api_key" not in entry


def test_crewai_prefixes_come_from_the_registry(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    llm = crewai_team.get_crew_llm("groq")
    assert REGISTRY["groq"].crewai_prefix == "groq"
    assert llm is not None
