"""Dependency-free tool-use loops against each vendor's SDK directly.

This is the template with the most correctness surface, because the loop is
hand-written rather than supplied by a framework. Five failure modes in the
naive version are handled explicitly here; see README.md for the rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from agents.provider import CLAUDE, OPENAI, get_provider, require_key
from agents.tools import TOOL_NAME, anthropic_tools, dispatch, openai_tools

DEFAULT_MAX_ITERATIONS = 10
DEFAULT_MAX_TOKENS = 16000

SYSTEM_PROMPT = (
    "You are a ledger analyst. Use the execute_project_query tool to retrieve "
    "figures before answering. Never estimate or recompute totals yourself -- "
    "quote what the tool returns."
)


class AgentLoopError(RuntimeError):
    """Base class for loop failures."""


class MaxIterationsExceeded(AgentLoopError):
    """The loop hit its iteration cap without the model finishing.

    Raised rather than returned: an unbounded `while True` around a paid API is
    the single most expensive bug this template can ship with.
    """


class OutputTruncated(AgentLoopError):
    """The model hit max_tokens mid-answer, so the text is incomplete."""


@dataclass
class AgentResult:
    text: str
    stop_reason: str
    iterations: int
    tool_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    @property
    def refused(self) -> bool:
        return self.stop_reason in ("refusal", "content_filter")


# --------------------------------------------------------------- Anthropic


def _claude_text(content: Sequence[Any]) -> str:
    """Join every text block.

    The naive `content[0].text` breaks as soon as thinking is on, because the
    first block is then a thinking block with no `.text` attribute at all.
    """
    return "\n".join(
        block.text for block in content if getattr(block, "type", None) == "text"
    ).strip()


def run_claude_agent(
    prompt: str,
    *,
    client: Any = None,
    model: str | None = None,
    system: str = SYSTEM_PROMPT,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    thinking: bool = True,
    allow_truncated: bool = False,
) -> AgentResult:
    """Run the Anthropic tool-use loop to completion.

    `client` is injectable so the loop can be tested without credentials.
    """
    config = get_provider(CLAUDE, model)
    if client is None:
        import anthropic

        require_key(config)
        client = anthropic.Anthropic()

    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    calls: list[tuple[str, dict[str, Any]]] = []

    request: dict[str, Any] = {
        "model": config.model,
        "max_tokens": max_tokens,
        "system": system,
        "tools": anthropic_tools(),
    }
    if thinking:
        # Adaptive thinking is the current form; budget_tokens is rejected on
        # current models.
        request["thinking"] = {"type": "adaptive"}

    for iteration in range(1, max_iterations + 1):
        response = client.messages.create(messages=messages, **request)
        stop = response.stop_reason

        if stop == "tool_use":
            # Echo the assistant turn back verbatim. Rebuilding it from text
            # would drop the tool_use and thinking blocks the API needs.
            messages.append({"role": "assistant", "content": response.content})

            # Claude may emit several tool_use blocks in one turn. Every one of
            # them needs a tool_result, and they must all travel in a SINGLE
            # user message -- handling only the first is an API error on the
            # next request, and splitting them trains the model out of
            # parallel calls.
            results: list[dict[str, Any]] = []
            for block in response.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                calls.append((block.name, dict(block.input)))
                result: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                }
                try:
                    result["content"] = dispatch(block.name, dict(block.input))
                except Exception as exc:
                    # Report the failure to the model instead of dropping the
                    # block; a missing tool_result is a hard API error.
                    result["content"] = f"Tool error: {exc}"
                    result["is_error"] = True
                results.append(result)

            messages.append({"role": "user", "content": results})
            continue

        if stop == "pause_turn":
            # Server-side tools paused the turn; resend to resume.
            messages.append({"role": "assistant", "content": response.content})
            continue

        if stop == "max_tokens" and not allow_truncated:
            raise OutputTruncated(
                f"Response hit max_tokens ({max_tokens}) after {iteration} "
                "iteration(s); raise max_tokens or narrow the prompt."
            )

        # end_turn, stop_sequence, refusal, or a truncated response the caller
        # opted to accept. A refusal returns normally with stop_reason set --
        # it is a real model outcome, not a crash -- so check `.refused`.
        return AgentResult(
            text=_claude_text(response.content),
            stop_reason=stop,
            iterations=iteration,
            tool_calls=calls,
        )

    raise MaxIterationsExceeded(
        f"Loop did not finish within {max_iterations} iterations."
    )


# ------------------------------------------------------------------ OpenAI


def run_openai_agent(
    prompt: str,
    *,
    client: Any = None,
    model: str | None = None,
    system: str = SYSTEM_PROMPT,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    allow_truncated: bool = False,
) -> AgentResult:
    """Run the equivalent loop against OpenAI chat completions."""
    import json

    config = get_provider(OPENAI, model)
    if client is None:
        import openai

        require_key(config)
        client = openai.OpenAI()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    calls: list[tuple[str, dict[str, Any]]] = []

    for iteration in range(1, max_iterations + 1):
        response = client.chat.completions.create(
            model=config.model,
            max_tokens=max_tokens,
            tools=openai_tools(),
            messages=messages,
        )
        choice = response.choices[0]
        finish = choice.finish_reason

        if finish == "tool_calls":
            messages.append(choice.message)

            # As with Anthropic, every tool call needs a reply -- OpenAI wants
            # one `tool` message per call id.
            for call in choice.message.tool_calls:
                # Always json.loads the arguments; never string-match them.
                arguments = json.loads(call.function.arguments)
                calls.append((call.function.name, arguments))
                try:
                    content = dispatch(call.function.name, arguments)
                except Exception as exc:
                    content = f"Tool error: {exc}"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": content,
                    }
                )
            continue

        if finish == "length" and not allow_truncated:
            raise OutputTruncated(
                f"Response hit max_tokens ({max_tokens}) after {iteration} "
                "iteration(s); raise max_tokens or narrow the prompt."
            )

        return AgentResult(
            text=(choice.message.content or "").strip(),
            stop_reason=finish,
            iterations=iteration,
            tool_calls=calls,
        )

    raise MaxIterationsExceeded(
        f"Loop did not finish within {max_iterations} iterations."
    )


def run_agent(prompt: str, provider: str | None = None, **kwargs: Any) -> AgentResult:
    """Run the native loop for whichever provider is selected."""
    config = get_provider(provider)
    if config.name == CLAUDE:
        return run_claude_agent(prompt, model=kwargs.pop("model", None), **kwargs)
    return run_openai_agent(prompt, model=kwargs.pop("model", None), **kwargs)


if __name__ == "__main__":
    import sys

    from agents.provider import load_dotenv

    load_dotenv()
    question = " ".join(sys.argv[1:]) or "Does the ledger balance? Explain briefly."
    result = run_agent(question)
    print(f"[{result.stop_reason} after {result.iterations} iteration(s)]")
    print(f"tool calls: {result.tool_calls}")
    print(result.text)
