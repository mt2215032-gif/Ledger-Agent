"""GPT-backed analysis over a reconciled ledger.

The division of labour matters here: `ledger_reconcile` computes every figure,
this module asks the language model to explain them. If no API key is
configured the module falls back to a deterministic report built from the same
findings, so the lab is fully demonstrable offline -- just without the prose.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import pandas as pd
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from ledger_data import get_ledger
from ledger_prompt import chat_template, ledger_template, notes_template
from ledger_reconcile import ReconciliationReport, format_ledger_table, reconcile

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_TEMPERATURE = 0.2


class MissingAPIKeyError(RuntimeError):
    """Raised when a live model call is attempted with no API key configured."""


def api_key_configured() -> bool:
    """True when an OpenAI key is present in the environment."""
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


def get_llm(model: str | None = None, temperature: float = DEFAULT_TEMPERATURE):
    """Build the chat model.

    Imported lazily so the rest of the project -- reconciliation, tests, the
    offline report -- stays importable on a machine with no OpenAI credentials.
    """
    if not api_key_configured():
        raise MissingAPIKeyError(
            "OPENAI_API_KEY is not set. Export it, or use offline mode, which "
            "reports the same findings without the generated narrative."
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model or DEFAULT_MODEL, temperature=temperature)


@dataclass
class AnalysisResult:
    """A reconciled ledger plus whatever narrative was produced for it."""

    df: pd.DataFrame
    report: ReconciliationReport
    narrative: str
    used_llm: bool

    def __iter__(self):
        """Unpack as `df, narrative`, matching the lab's two-value return."""
        return iter((self.df, self.narrative))


def offline_narrative(report: ReconciliationReport) -> str:
    """Build a narrative from the computed findings, with no model call.

    Deliberately plain: it states what the checks found and stops there, so it
    is never mistaken for the model's analysis.
    """
    lines = ["## Reconciliation summary (offline mode -- no model call)", ""]

    if report.is_balanced:
        lines.append(
            f"The ledger balances. Total debits and total credits are both "
            f"{report.total_debits:,.2f}."
        )
    else:
        heavier = "Credits" if report.difference < 0 else "Debits"
        lines.append(
            f"**The ledger does not balance.** Total debits are "
            f"{report.total_debits:,.2f} against total credits of "
            f"{report.total_credits:,.2f}. {heavier} exceed the other side by "
            f"{abs(report.difference):,.2f}."
        )

    if report.errors:
        lines += ["", "### Errors", ""]
        lines += [f"- {f.message}" + _rows_suffix(f.rows) for f in report.errors]

    if report.warnings:
        lines += ["", "### Warnings", ""]
        lines += [f"- {f.message}" + _rows_suffix(f.rows) for f in report.warnings]

    if not report.findings:
        lines += ["", "All automated checks passed. No exceptions to report."]

    lines += [
        "",
        "_Set `OPENAI_API_KEY` to have the agent explain these findings, propose "
        "correcting journal entries and answer follow-up questions._",
    ]
    return "\n".join(lines)


def _rows_suffix(rows: Sequence[int]) -> str:
    return f" (row(s) {', '.join(str(r) for r in rows)})" if rows else ""


def analyze_ledger(
    df: pd.DataFrame | None = None,
    *,
    model: str | None = None,
    offline: bool = False,
) -> AnalysisResult:
    """Reconcile `df`, then narrate the result.

    Falls back to the offline report when `offline` is set or no key is
    configured, so this never raises just because credentials are absent.
    """
    ledger = get_ledger() if df is None else df
    report = reconcile(ledger)

    if offline or not api_key_configured():
        return AnalysisResult(ledger, report, offline_narrative(report), used_llm=False)

    prompt = ledger_template.format(
        ledger_table=format_ledger_table(ledger),
        reconciliation_facts=report.to_facts(),
    )
    response = get_llm(model).invoke(prompt)
    return AnalysisResult(ledger, report, response.content, used_llm=True)


def run_ledger_analysis(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, str]:
    """Return `(df, analysis_text)` -- the lab's Step 4 entry point."""
    result = analyze_ledger(df)
    return result.df, result.narrative


def generate_reconciliation_notes(
    df: pd.DataFrame | None = None,
    *,
    model: str | None = None,
    offline: bool = False,
) -> str:
    """Produce month-end working-paper notes for the ledger."""
    ledger = get_ledger() if df is None else df
    report = reconcile(ledger)

    if offline or not api_key_configured():
        return offline_narrative(report)

    prompt = notes_template.format(
        ledger_table=format_ledger_table(ledger),
        reconciliation_facts=report.to_facts(),
    )
    return get_llm(model).invoke(prompt).content


def _to_messages(history: Sequence[tuple[str, str]]) -> list[BaseMessage]:
    """Convert `(role, text)` pairs into LangChain messages."""
    messages: list[BaseMessage] = []
    for role, text in history:
        if role in ("user", "human"):
            messages.append(HumanMessage(content=text))
        else:
            messages.append(AIMessage(content=text))
    return messages


def chat_with_ledger(
    question: str,
    df: pd.DataFrame | None = None,
    history: Sequence[tuple[str, str]] = (),
    *,
    model: str | None = None,
) -> str:
    """Answer a question about the ledger, with the reconciliation pinned in context.

    `history` is the prior turns as `(role, text)` pairs, oldest first.
    """
    ledger = get_ledger() if df is None else df
    report = reconcile(ledger)

    if not api_key_configured():
        raise MissingAPIKeyError(
            "Chat needs a live model. Set OPENAI_API_KEY to ask questions; the "
            "Analyze tab still works offline."
        )

    chain = chat_template | get_llm(model)
    response = chain.invoke(
        {
            "ledger_table": format_ledger_table(ledger),
            "reconciliation_facts": report.to_facts(),
            "history": _to_messages(history),
            "question": question,
        }
    )
    return response.content


if __name__ == "__main__":
    result = analyze_ledger()
    print(format_ledger_table(result.df))
    print()
    print(result.narrative)
