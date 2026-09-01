"""Provider registry shared by every agent template.

Selects the model backend from `PROVIDER` (and optionally `MODEL`). Both
proprietary APIs and open-weight sources are registered; each entry records
what the four frameworks need to reach it, so the templates never hardcode a
vendor.

Every provider is reached through its own SDK or LangChain integration. The one
deliberate shortcut is the native loop, where the open-weight hosts are driven
through the OpenAI SDK against their own `base_url` -- that is their documented
OpenAI-compatible endpoint, not a shim bolted onto a different vendor.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Back-compat aliases for the two providers the templates started with.
CLAUDE = "claude"
OPENAI = "openai"


@dataclass(frozen=True)
class Provider:
    """Everything the four frameworks need to reach one model source."""

    key: str
    label: str
    default_model: str
    open_weights: bool
    # LangChain integration (LangGraph template)
    langchain_module: str
    langchain_class: str
    pip_package: str
    # None means "no API key required" (a local runtime such as Ollama).
    api_key_env: str | None = None
    # CrewAI model prefix; None means CrewAI cannot reach this provider.
    crewai_prefix: str | None = None
    # AutoGen api_type; None means AutoGen has no client for it.
    autogen_api_type: str | None = None
    # OpenAI-compatible endpoint, used by the native loop. None for Claude,
    # which has a first-class SDK of its own.
    openai_base_url: str | None = None
    notes: str = ""


#: Registered model sources, proprietary first, then open-weight.
REGISTRY: dict[str, Provider] = {
    "claude": Provider(
        key="claude",
        label="Anthropic Claude",
        default_model="claude-opus-5",
        open_weights=False,
        langchain_module="langchain_anthropic",
        langchain_class="ChatAnthropic",
        pip_package="langchain-anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        crewai_prefix="anthropic",
        autogen_api_type="anthropic",
        openai_base_url=None,  # uses the Anthropic SDK natively
        notes="Native loop uses the anthropic SDK, not an OpenAI-compatible path.",
    ),
    "openai": Provider(
        key="openai",
        label="OpenAI",
        default_model="gpt-4o-mini",
        open_weights=False,
        langchain_module="langchain_openai",
        langchain_class="ChatOpenAI",
        pip_package="langchain-openai",
        api_key_env="OPENAI_API_KEY",
        crewai_prefix="openai",
        autogen_api_type="openai",
        openai_base_url="https://api.openai.com/v1",
    ),
    "gemini": Provider(
        key="gemini",
        label="Google Gemini",
        default_model="gemini-2.0-flash",
        open_weights=False,
        langchain_module="langchain_google_genai",
        langchain_class="ChatGoogleGenerativeAI",
        pip_package="langchain-google-genai",
        api_key_env="GOOGLE_API_KEY",
        crewai_prefix="gemini",
        autogen_api_type="google",
        openai_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        notes="CrewAI needs the crewai[google-genai] extra.",
    ),
    # ---------------------------------------------------------- open weights
    "ollama": Provider(
        key="ollama",
        label="Ollama (local, open weights)",
        default_model="llama3.1",
        open_weights=True,
        langchain_module="langchain_ollama",
        langchain_class="ChatOllama",
        pip_package="langchain-ollama",
        api_key_env=None,  # runs locally; no credential
        crewai_prefix="ollama",
        autogen_api_type="ollama",
        openai_base_url="http://localhost:11434/v1",
        notes="Needs a local Ollama server: `ollama serve` and `ollama pull llama3.1`.",
    ),
    "groq": Provider(
        key="groq",
        label="Groq (open-weight models)",
        default_model="llama-3.3-70b-versatile",
        open_weights=True,
        langchain_module="langchain_groq",
        langchain_class="ChatGroq",
        pip_package="langchain-groq",
        api_key_env="GROQ_API_KEY",
        crewai_prefix="groq",
        autogen_api_type="groq",
        openai_base_url="https://api.groq.com/openai/v1",
        notes="CrewAI reaches Groq through the crewai[litellm] extra.",
    ),
    "together": Provider(
        key="together",
        label="Together AI (open-weight models)",
        default_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        open_weights=True,
        langchain_module="langchain_together",
        langchain_class="ChatTogether",
        pip_package="langchain-together",
        api_key_env="TOGETHER_API_KEY",
        crewai_prefix="together_ai",
        autogen_api_type="together",
        openai_base_url="https://api.together.xyz/v1",
        notes="CrewAI reaches Together through the crewai[litellm] extra.",
    ),
    "mistral": Provider(
        key="mistral",
        label="Mistral (open-weight models)",
        default_model="mistral-large-latest",
        open_weights=True,
        langchain_module="langchain_mistralai",
        langchain_class="ChatMistralAI",
        pip_package="langchain-mistralai",
        api_key_env="MISTRAL_API_KEY",
        crewai_prefix="mistral",
        autogen_api_type="mistral",
        openai_base_url="https://api.mistral.ai/v1",
        notes="CrewAI reaches Mistral through the crewai[litellm] extra.",
    ),
    "deepseek": Provider(
        key="deepseek",
        label="DeepSeek (open-weight models)",
        default_model="deepseek-chat",
        open_weights=True,
        langchain_module="langchain_deepseek",
        langchain_class="ChatDeepSeek",
        pip_package="langchain-deepseek",
        api_key_env="DEEPSEEK_API_KEY",
        crewai_prefix="deepseek",
        autogen_api_type="deepseek",
        openai_base_url="https://api.deepseek.com",
    ),
    "cerebras": Provider(
        key="cerebras",
        label="Cerebras (open-weight models)",
        default_model="llama-3.3-70b",
        open_weights=True,
        langchain_module="langchain_cerebras",
        langchain_class="ChatCerebras",
        pip_package="langchain-cerebras",
        api_key_env="CEREBRAS_API_KEY",
        crewai_prefix="cerebras",
        autogen_api_type="cerebras",
        openai_base_url="https://api.cerebras.ai/v1",
    ),
}

#: Kept so existing callers and tests can read the default per provider.
DEFAULT_MODELS = {key: p.default_model for key, p in REGISTRY.items()}
ENV_KEYS = {key: p.api_key_env for key, p in REGISTRY.items()}


def list_providers(open_weights: bool | None = None) -> list[Provider]:
    """All registered providers, optionally filtered to open-weight sources."""
    providers = list(REGISTRY.values())
    if open_weights is None:
        return providers
    return [p for p in providers if p.open_weights is open_weights]


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
    """A provider resolved against the environment, with a chosen model."""

    provider: Provider
    model: str

    @property
    def name(self) -> str:
        return self.provider.key

    @property
    def api_key_env(self) -> str | None:
        return self.provider.api_key_env

    @property
    def api_key(self) -> str | None:
        if self.provider.api_key_env is None:
            return None
        return os.getenv(self.provider.api_key_env) or None

    @property
    def needs_key(self) -> bool:
        return self.provider.api_key_env is not None

    @property
    def configured(self) -> bool:
        """True when this provider can be called.

        A keyless provider such as Ollama is always considered configured --
        whether the local server is actually running is a connection error at
        call time, not a configuration question.
        """
        return not self.needs_key or bool(self.api_key)


class MissingAPIKeyError(RuntimeError):
    """Raised when a live call is attempted with no credentials."""


class ProviderNotInstalledError(RuntimeError):
    """Raised when the provider's integration package is missing."""


def get_provider(name: str | None = None, model: str | None = None) -> ProviderConfig:
    """Resolve the active provider from the argument, then the environment."""
    resolved = (name or os.getenv("PROVIDER") or CLAUDE).strip().lower()
    if resolved not in REGISTRY:
        raise ValueError(
            f"Unknown provider {resolved!r}. Registered: {', '.join(sorted(REGISTRY))}"
        )
    provider = REGISTRY[resolved]
    return ProviderConfig(
        provider=provider,
        model=model or os.getenv("MODEL") or provider.default_model,
    )


def require_key(config: ProviderConfig) -> None:
    if not config.configured:
        raise MissingAPIKeyError(
            f"{config.api_key_env} is not set, which {config.provider.label} needs "
            "for a live call. Export it or put it in .env."
        )


def get_chat_model(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    **kwargs,
):
    """Return a LangChain chat model for the active provider.

    Imports are deferred so selecting one provider never requires the other
    eight packages to be installed.
    """
    config = get_provider(provider, model)
    require_key(config)
    spec = config.provider

    try:
        module = __import__(spec.langchain_module, fromlist=[spec.langchain_class])
    except ImportError as exc:
        raise ProviderNotInstalledError(
            f"{spec.label} needs the {spec.pip_package} package: "
            f"pip install {spec.pip_package}"
        ) from exc

    chat_class = getattr(module, spec.langchain_class)
    return chat_class(model=config.model, temperature=temperature, **kwargs)
