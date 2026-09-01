# Ledger Analysis Agent 📒

An AI agent that reconciles double-entry ledger data, flags imbalances and
posting errors, and explains them conversationally.

Built for the GPT Ledger Agent lab: ingest a ledger as a Pandas DataFrame, use
GPT to summarise it and detect imbalances, explain the discrepancies, draft
reconciliation notes, and chat with the ledger through a Streamlit UI.

## The one design decision worth knowing

**The model never does the arithmetic.** Every figure — totals, per-date
subtotals, per-account movements, the list of exceptions — is computed in Pandas
by `ledger_reconcile.py` and passed to the model as an authoritative fact sheet.
GPT's job is to explain those findings and propose corrections.

A language model that adds up a column will eventually get it wrong, and on a
finance tool a plausible-looking wrong total is worse than no answer. Splitting
the work this way also means the reconciliation is unit-testable and the app
stays useful with no API key at all.

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Set a key to enable the written analysis and chat:

```bash
export OPENAI_API_KEY="sk-..."    # or copy .env.example to .env
```

## Run

```bash
streamlit run app.py
```

Or use it from the command line and from Python:

```bash
python ledger_agent.py            # prints the ledger and its analysis
```

```python
from ledger_agent import run_ledger_analysis
df, analysis = run_ledger_analysis()
```

## What the app does

The sidebar picks a ledger — the built-in sample, a corrected version, or your
own CSV with `Date, Description, Account, Debit, Credit` columns. Totals stay
visible above the tabs and update as you edit.

| Tab | What it gives you |
| --- | --- |
| **Ledger** | An editable grid, live totals, and every automated finding with the rows it refers to |
| **Analysis** | A full written analysis, plus month-end reconciliation notes you can download |
| **Chat** | Questions about the ledger, answered against the pinned reconciliation facts |

### Without an API key

Reconciliation, findings and the ledger view work exactly the same. The Analysis
tab reports the findings without a generated narrative, and the chat tab is
disabled with a note explaining why. Nothing crashes and nothing silently
degrades.

## The sample ledger

The included sample is deliberately broken, so there is something to find:

| Row | Date | Description | Account | Debit | Credit |
| --- | --- | --- | --- | ---: | ---: |
| 0 | 2025-07-01 | Customer Payment | Accounts Receivable | 0 | 500 |
| 1 | 2025-07-01 | Revenue Recorded | Revenue | 500 | 0 |
| 2 | 2025-07-02 | Office Supplies | Office Supplies | 100 | 0 |
| 3 | 2025-07-02 | Cash Paid | Cash | 100 | 0 |
| 4 | 2025-07-03 | Consulting Income | Revenue | 0 | 1500 |

Debits total 700 against credits of 2,000, and the agent reports:

- **LEDGER_IMBALANCE** — credits exceed debits by 1,300.
- **DATE_IMBALANCE** — 2025-07-02 posts two debits with no credit (out by 200);
  "Cash Paid" should credit Cash.
- **DATE_IMBALANCE** — 2025-07-03 credits Revenue 1,500 with no offsetting debit.
- **UNEXPECTED_DEBIT** — row 1 debits Revenue; recording revenue credits it.

`get_balanced_ledger()` returns the same business events posted correctly, and
produces no findings.

## Checks

| Code | Severity | Fires when |
| --- | --- | --- |
| `LEDGER_IMBALANCE` | error | Total debits ≠ total credits |
| `DATE_IMBALANCE` | error | A single date's entries don't balance |
| `NEGATIVE_AMOUNT` | error | A debit or credit is negative |
| `EMPTY_ENTRY` | warning | A row posts neither a debit nor a credit |
| `DEBIT_AND_CREDIT` | warning | One row carries both sides |
| `UNEXPECTED_DEBIT` | warning | A revenue, liability or equity account is debited |

Two deliberate omissions:

- **Crediting an asset or expense account is not flagged.** Every cash payment
  credits Cash. Flagging it would fire on most correct ledgers. Debiting revenue
  is genuinely uncommon, so only that direction is checked.
- **Per-account figures are period movements, not closing balances.** No opening
  balances are loaded, so a net credit on Cash means it decreased over the
  period, not that the account is overdrawn — and nothing concludes otherwise.

## Files

| File | Role |
| --- | --- |
| `ledger_data.py` | Sample ledgers and the column contract, with input validation |
| `ledger_reconcile.py` | All reconciliation maths and the checks. No LLM involved |
| `ledger_prompt.py` | Prompt templates for analysis, notes and chat |
| `ledger_agent.py` | Wires reconciliation to the model, with an offline fallback |
| `app.py` | Streamlit UI |
| `tests/` | 39 tests over the reconciliation logic and the app |

## Tests

```bash
pytest -q
```

39 tests, no API key required. The reconciliation logic is tested directly; the
model-backed paths use a stub chat model, which verifies that the computed
totals actually reach the prompt without asserting anything about generated
prose.

## Notes on the lab code

Two things in the lab handout no longer run on current LangChain, and the code
here uses the supported equivalents:

| Lab handout | Used here | Why |
| --- | --- | --- |
| `from langchain.chat_models import ChatOpenAI` | `from langchain_openai import ChatOpenAI` | Moved to its own package; the old path raises `ImportError` |
| `from langchain.prompts import PromptTemplate` | `from langchain_core.prompts import PromptTemplate` | `langchain.prompts` no longer exists |
| `llm.predict(prompt)` | `llm.invoke(prompt).content` | `.predict()` was removed in LangChain 1.x |

`run_ledger_analysis()` keeps the handout's signature and still returns
`(df, response)`, so anything written against the lab's Step 4 still works.

Verified against langchain 1.3.18, langchain-core 1.6.1, langchain-openai 1.6.0,
openai 3.6.0, pandas 3.0.5, streamlit 1.62.0, Python 3.11.
