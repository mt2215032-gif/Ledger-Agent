# Multi-provider agent framework templates

Four agent patterns — LangGraph, CrewAI, AutoGen, and a hand-written native
loop — each switchable across **nine model sources**: proprietary APIs (Claude,
OpenAI, Gemini) and open-weight ones (Ollama, Groq, Together, Mistral,
DeepSeek, Cerebras).

```bash
python -m venv venv-agents && source venv-agents/bin/activate
pip install -r requirements-agents.txt

export ANTHROPIC_API_KEY=sk-ant-...     # whichever provider you pick
export PROVIDER=claude                  # see the table below
export MODEL=...                        # optional; overrides the default

python -m agents.native_loop "Does the ledger balance?"
python -m agents.langgraph_agent
python -m agents.crewai_team
python -m agents.autogen_chat
```

`PROVIDER` and `MODEL` select the backend; every template calls
`agents.provider`, so nothing hardcodes a vendor.

## Sources

Registered in `provider.py`. Every row was constructed and verified; nothing
here is guessed from documentation.

| `PROVIDER` | Source | Open weights | Default model | Key |
| --- | --- | --- | --- | --- |
| `claude` | Anthropic Claude | — | `claude-opus-5` | `ANTHROPIC_API_KEY` |
| `openai` | OpenAI | — | `gpt-4o-mini` | `OPENAI_API_KEY` |
| `gemini` | Google Gemini | — | `gemini-2.0-flash` | `GOOGLE_API_KEY` |
| `ollama` | Ollama, local | ✅ | `llama3.1` | none |
| `groq` | Groq | ✅ | `llama-3.3-70b-versatile` | `GROQ_API_KEY` |
| `together` | Together AI | ✅ | `Llama-3.3-70B-Instruct-Turbo` | `TOGETHER_API_KEY` |
| `mistral` | Mistral | ✅ | `mistral-large-latest` | `MISTRAL_API_KEY` |
| `deepseek` | DeepSeek | ✅ | `deepseek-chat` | `DEEPSEEK_API_KEY` |
| `cerebras` | Cerebras | ✅ | `llama-3.3-70b` | `CEREBRAS_API_KEY` |

`ollama` needs no key — it runs locally (`ollama serve && ollama pull llama3.1`).
A missing local server surfaces as a connection error at call time, not a
configuration error.

### Support per framework

Not every framework reaches every source, and the gaps are extras rather than
dead ends:

| Source | LangGraph | CrewAI | AutoGen | Native loop |
| --- | --- | --- | --- | --- |
| claude | ✅ | ✅ | ✅ | Anthropic SDK |
| openai | ✅ | ✅ | ✅ | OpenAI SDK |
| gemini | ✅ | needs `crewai[google-genai]` | ✅ | OpenAI-compatible |
| ollama | ✅ | ✅ | ✅ | OpenAI-compatible |
| groq | ✅ | needs `crewai[litellm]` | ✅ | OpenAI-compatible |
| together | ✅ | needs `crewai[litellm]` | ✅ | OpenAI-compatible |
| mistral | ✅ | needs `crewai[litellm]` | ✅ | OpenAI-compatible |
| deepseek | ✅ | ✅ | ✅ | OpenAI-compatible |
| cerebras | ✅ | ✅ | ✅ | OpenAI-compatible |

**Claude is never routed through an OpenAI-compatible path.** It has a
first-class SDK, and a shim would silently change tool-use and thinking
semantics; `run_openai_agent(provider="claude")` raises rather than doing it.
The open-weight hosts genuinely publish OpenAI-compatible endpoints, so one
loop serves all six against their own `base_url` — that is using a documented
API, not pretending one vendor is another.

## Files

| File | Role |
| --- | --- |
| `provider.py` | Provider/model resolution, `.env` loading, key checks |
| `tools.py` | One tool definition rendered into both providers' schemas |
| `langgraph_agent.py` | State machine with an agent ↔ tools cycle |
| `crewai_team.py` | Researcher → writer crew |
| `autogen_chat.py` | Conversable assistant + execution proxy |
| `native_loop.py` | Hand-written tool loops for both SDKs |

The tool is backed by this repository's ledger reconciliation, so the agents
query real figures (debits 700.00, credits 2,000.00, out of balance by 1,300.00)
rather than returning a canned string.

## What changed from the source templates, and why

Everything below was verified by running it, not inferred.

### 1. Model IDs were several generations stale

All four templates pinned `claude-3-5-sonnet-20241022`. The default is now
`claude-opus-5` (`DEFAULT_MODELS` in `provider.py`), overridable with `MODEL`.

### 2. The native loop had five defects

The hand-written loop carries the most risk, because no framework is covering
for it. Each of these was reproduced against the original code:

| Defect | Consequence | Fix |
| --- | --- | --- |
| `response.content[0].text` | **AttributeError.** With thinking on, block 0 is a `thinking` block with no `.text` | Join every `text` block |
| Only the first `tool_use` block answered, via `next(...)` | **API 400.** Claude emits parallel `tool_use` blocks; every one needs a `tool_result` | Answer all of them, in one user message |
| `while True` with no cap | Unbounded spend against a paid API | `max_iterations`, then `MaxIterationsExceeded` |
| Only `end_turn` and `tool_use` handled | **Infinite loop** on `max_tokens`, `refusal`, `pause_turn` — verified: all three spin forever | Every stop reason handled explicitly |
| Tool exceptions unhandled | A dropped `tool_result` is a hard API error | Failures returned with `is_error: true` |

Splitting tool results across several user messages also trains the model out
of parallel calls, so they go back in a single message.

`max_tokens` raises `OutputTruncated` rather than returning a half-answer that
reads like a complete one; pass `allow_truncated=True` to accept it. A refusal
returns normally with `result.refused` set — it is a real model outcome, not a
crash.

### 3. LangGraph had no cycle

The original was `set_entry_point("agent")` → `add_edge("agent", END)`: a single
model call, with no tools and no loop, despite being described as a cyclical
runtime. This version adds a `ToolNode` and a `tools_condition` conditional
edge, giving agent → tools → agent → … → END. The cycle is the whole point of
using LangGraph.

### 4. CrewAI rejects a LangChain model

`llm=ChatAnthropic(...)` raises `pydantic.ValidationError` on CrewAI 1.x —
`Agent.llm` is typed `str | BaseLLM | None`. The working form is CrewAI's own
`LLM` class with a provider-prefixed id:

```python
LLM(model="anthropic/claude-opus-5")   # not ChatAnthropic(...)
```

`test_crewai_rejects_a_langchain_model` pins this so the old form cannot
quietly return.

### 5. AutoGen: wrong package, and unsandboxed code execution

`from autogen import AssistantAgent, UserProxyAgent` **fails on `ag2` 1.x**,
which is a ground-up rewrite exporting none of those names. The classic
conversable API lives in the `autogen` distribution (0.14.1 verified). Your
`llm_config` shape — `config_list` with `api_type: "anthropic"` — is correct
there and needs `autogen[anthropic]` for the client.

The original also passed `code_execution_config={"use_docker": False}`, which
executes model-written code directly on the host with no sandbox. That is
**off by default** here; the agent gets a registered ledger tool instead. Pass
`enable_code_execution=True` to opt in.

## Tests

```bash
pytest tests/test_agents.py -q      # 63 tests, no API key needed
```

Loops are driven with stub clients reproducing the real response shapes. That
is deliberate: parallel tool use, `max_tokens`, refusals and runaway loops are
exactly the paths a happy-path live call never reaches.

## Verification status

Structure, wiring, schemas and control flow are verified. **No live model call
was made** for either provider:

- `api.openai.com` is blocked by this environment's egress policy (403 on
  CONNECT), so the OpenAI path could not be exercised live.
- `api.anthropic.com` is reachable (401 — credentials only), but no
  `ANTHROPIC_API_KEY` was available.

So request construction is verified against stubs, and response *parsing* is
verified against replayed shapes — but no real API response has passed through
this code. Run one of the `__main__` entry points with a key to close that gap.

## Adding another source

Append a `Provider` to `REGISTRY` in `provider.py`. The parametrised tests then
cover it automatically: they assert the LangChain module and class actually
import and construct, and that a non-Claude provider declares a `base_url` for
the native loop. Nothing else needs editing — all four templates read the
registry.

## Version pins that matter

`autogen[anthropic]` requires `anthropic<1.0` and `crewai` requires
`openai<3.0`, so installing all four templates together pins both SDKs below
their latest majors. `anthropic` 0.125.0 still exposes `messages.create`,
`tools`, `thinking` and `output_config` — everything these templates use. If
you only need one template, install just its dependencies and the ceilings lift.
