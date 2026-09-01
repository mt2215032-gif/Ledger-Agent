"""Provider selection shared by every agent template.

Set PROVIDER=claude (default) or PROVIDER=openai. Both paths use each vendor's
own SDK -- no OpenAI-compatible shim pointed at Anthropic, which loses tool-use
and thinking semantics and silently changes behaviour.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

CLAUDE = "claude"
OPENAI = "openai"

# Current model IDs. The Claude default follows Anthropic's current guidance;
# the pasted templates' claude-3-5-sonnet-20241022 is several generations old.
DEFAULT_MODELS = {
    CLAUDE: "claude-opus-5",
    OPENAI: "gpt-4o-mini",
}

ENV_KEYS = {
    CLAUDE: "ANTHROPIC_API_KEY",
    OPENAI: "OPENAI_API_KEY",
}


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader so templates run without an extra dependency.

    Existing environment variables win, which keeps an exported key
    authoritative over a stale file.
    """
    env_path = Path(path)
    if not env_path.is_file():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    model: str
    api_key_env: str

    @property
    def api_key(self) -> str | None:
        return os.getenv(self.api_key_env) or None

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


def get_provider(name: str | None = None, model: str | None = None) -> ProviderConfig:
    """Resolve the active provider from the argument, then the environment."""
    resolved = (name or os.getenv("PROVIDER") or CLAUDE).strip().lower()
    if resolved not in DEFAULT_MODELS:
        raise ValueError(
            f"Unknown provider {resolved!r}. Expected one of: {', '.join(DEFAULT_MODELS)}"
        )
    return ProviderConfig(
        name=resolved,
        model=model or os.getenv("MODEL") or DEFAULT_MODELS[resolved],
        api_key_env=ENV_KEYS[resolved],
    )


class MissingAPIKeyError(RuntimeError):
    """Raised when a live call is attempted with no credentials for the provider."""


def require_key(config: ProviderConfig) -> None:
    if not config.configured:
        raise MissingAPIKeyError(
            f"{config.api_key_env} is not set, which {config.name} needs for a live call. "
            f"Export it or put it in .env."
        )


def get_chat_model(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
):
    """Return a LangChain chat model for the active provider.

    Used by the LangGraph and CrewAI templates, which are both built on
    LangChain model objects. Imports are deferred so that selecting one provider
    does not require the other's package to be installed.
    """
    config = get_provider(provider, model)
    require_key(config)

    if config.name == CLAUDE:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=config.model, temperature=temperature)

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=config.model, temperature=temperature)
