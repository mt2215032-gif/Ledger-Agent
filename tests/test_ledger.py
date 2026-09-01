"""Tests for the deterministic half of the agent.

The reconciliation logic is where a wrong answer would actually cost someone
money, so it is tested directly. The model-backed paths are exercised with a
stub chat model: that verifies the prompt formatting and chain wiring without
needing an API key or asserting anything about generated prose.
"""

from __future__ import annotations

import pandas as pd
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import ledger_agent
from ledger_data import get_balanced_ledger, get_ledger, validate_ledger
from ledger_reconcile import (
    TOLERANCE,
    classify_account,
    format_ledger_table,
    reconcile,
)


def codes(report) -> set[str]:
    return {f.code for f in report.findings}


# ------------------------------------------------------------------ totals


def test_sample_ledger_totals_match_the_source_data():
    report = reconcile(get_ledger())
    assert report.total_debits == 700.00
    assert report.total_credits == 2000.00
    assert report.difference == -1300.00
    assert not report.is_balanced


def test_corrected_ledger_balances_and_raises_nothing():
    report = reconcile(get_balanced_ledger())
    assert report.is_balanced
    assert report.difference == 0
    assert report.findings == []


def test_imbalance_is_reported_as_an_error():
    report = reconcile(get_ledger())
    assert "LEDGER_IMBALANCE" in codes(report)
    assert len(report.errors) >= 1


# -------------------------------------------------------------- per-date


def test_each_date_is_checked_independently():
    report = reconcile(get_ledger())
    by_date = report.by_date.set_index("Date")

    # This date balances even though the ledger as a whole does not.
    assert by_date.loc["2025-07-01", "Difference"] == 0
    assert by_date.loc["2025-07-02", "Difference"] == 200.00
    assert by_date.loc["2025-07-03", "Difference"] == -1500.00

    imbalanced = [
        f for f in report.findings if f.code == "DATE_IMBALANCE"
    ]
    assert len(imbalanced) == 2


def test_date_imbalance_cites_the_offending_rows():
    report = reconcile(get_ledger())
    finding = next(f for f in report.findings if f.code == "DATE_IMBALANCE")
    assert finding.rows  # a finding with no rows is not actionable


# --------------------------------------------------------- row hygiene


def test_negative_amounts_are_errors():
    df = pd.DataFrame(
        {
            "Date": ["2025-07-01", "2025-07-01"],
            "Description": ["Bad entry", "Offset"],
            "Account": ["Cash", "Revenue"],
            "Debit": [-100, 0],
            "Credit": [0, -100],
        }
    )
    assert "NEGATIVE_AMOUNT" in codes(reconcile(df))


def test_entry_with_no_amounts_is_flagged():
    df = pd.DataFrame(
        {
            "Date": ["2025-07-01"],
            "Description": ["Placeholder"],
            "Account": ["Cash"],
            "Debit": [0],
            "Credit": [0],
        }
    )
    assert "EMPTY_ENTRY" in codes(reconcile(df))


def test_entry_with_both_sides_on_one_line_is_flagged():
    df = pd.DataFrame(
        {
            "Date": ["2025-07-01"],
            "Description": ["Ambiguous"],
            "Account": ["Cash"],
            "Debit": [100],
            "Credit": [100],
        }
    )
    assert "DEBIT_AND_CREDIT" in codes(reconcile(df))


# ------------------------------------------------- posting conventions


def test_debit_to_a_revenue_account_is_flagged():
    report = reconcile(get_ledger())
    finding = next(f for f in report.findings if f.code == "UNEXPECTED_DEBIT")
    # Row 1 is "Revenue Recorded", posted as a debit to Revenue.
    assert finding.rows == (1,)
    assert finding.severity == "warning"


def test_crediting_an_asset_is_not_flagged():
    """Paying cash credits Cash. That is routine and must not raise a finding."""
    df = pd.DataFrame(
        {
            "Date": ["2025-07-02", "2025-07-02"],
            "Description": ["Office Supplies", "Cash Paid"],
            "Account": ["Office Supplies", "Cash"],
            "Debit": [100, 0],
            "Credit": [0, 100],
        }
    )
    report = reconcile(df)
    assert report.is_balanced
    assert report.findings == []


@pytest.mark.parametrize(
    "account, expected",
    [
        ("Cash", "asset"),
        ("Accounts Receivable", "asset"),
        ("Accounts Payable", "liability"),
        ("Revenue", "revenue"),
        ("Consulting Income", "revenue"),
        ("Office Supplies", "expense"),
        ("Retained Earnings", "equity"),
        ("Zorblax Fund", "unknown"),
    ],
)
def test_account_classification(account, expected):
    assert classify_account(account) == expected


def test_unknown_accounts_produce_no_convention_findings():
    df = pd.DataFrame(
        {
            "Date": ["2025-07-01", "2025-07-01"],
            "Description": ["Mystery in", "Mystery out"],
            "Account": ["Zorblax Fund", "Widget Pool"],
            "Debit": [50, 0],
            "Credit": [0, 50],
        }
    )
    assert reconcile(df).findings == []


# -------------------------------------------------------------- validation


def test_missing_column_is_rejected():
    df = get_ledger().drop(columns=["Credit"])
    with pytest.raises(ValueError, match="Credit"):
        validate_ledger(df)


def test_non_numeric_amount_is_rejected():
    # Built with a text amount in place, the way a messy CSV would parse.
    df = pd.DataFrame(
        {
            "Date": ["2025-07-01"],
            "Description": ["Typo in the amount"],
            "Account": ["Cash"],
            "Debit": ["five hundred"],
            "Credit": [0],
        }
    )
    with pytest.raises(ValueError, match="Non-numeric"):
        validate_ledger(df)


def test_numeric_strings_are_accepted():
    df = get_ledger()
    df["Debit"] = df["Debit"].astype(str)
    assert reconcile(df).total_debits == 700.00


def test_float_noise_stays_within_tolerance():
    """0.1 + 0.2 style drift must not be reported as an imbalance."""
    df = pd.DataFrame(
        {
            "Date": ["2025-07-01"] * 3,
            "Description": ["a", "b", "c"],
            "Account": ["Cash", "Cash", "Revenue"],
            "Debit": [0.1, 0.2, 0],
            "Credit": [0, 0, 0.3],
        }
    )
    report = reconcile(df)
    assert abs(report.difference) < TOLERANCE
    assert report.is_balanced


# ------------------------------------------------------------- rendering


def test_fact_sheet_carries_the_real_totals():
    facts = reconcile(get_ledger()).to_facts()
    assert "700.00" in facts
    assert "2,000.00" in facts
    assert "OUT OF BALANCE" in facts


def test_table_includes_row_numbers_findings_refer_to():
    table = format_ledger_table(get_ledger())
    assert "Row" in table
    assert "Consulting Income" in table


# ------------------------------------------------------- agent plumbing


def test_offline_analysis_needs_no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = ledger_agent.analyze_ledger()
    assert not result.used_llm
    assert "does not balance" in result.narrative
    assert "1,300.00" in result.narrative


def test_offline_result_unpacks_like_the_lab_signature(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    df, narrative = ledger_agent.run_ledger_analysis()
    assert len(df) == 5
    assert isinstance(narrative, str)


def test_chat_without_a_key_raises_a_clear_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ledger_agent.MissingAPIKeyError, match="OPENAI_API_KEY"):
        ledger_agent.chat_with_ledger("does this balance?")


def test_live_analysis_path_sends_the_computed_totals(monkeypatch):
    """The prompt reaching the model must contain the Pandas-computed figures."""
    seen: list[str] = []

    class RecordingModel(FakeListChatModel):
        def invoke(self, input, config=None, **kwargs):
            seen.append(str(input))
            return super().invoke(input, config, **kwargs)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        ledger_agent,
        "get_llm",
        lambda *a, **k: RecordingModel(responses=["Analysis text."]),
    )

    result = ledger_agent.analyze_ledger()
    assert result.used_llm
    assert result.narrative == "Analysis text."
    assert "700.00" in seen[0] and "2,000.00" in seen[0]
    assert "Consulting Income" in seen[0]


def test_chat_chain_composes_and_carries_history(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        ledger_agent,
        "get_llm",
        lambda *a, **k: FakeListChatModel(responses=["Because a credit is missing."]),
    )

    answer = ledger_agent.chat_with_ledger(
        "Why doesn't 2025-07-03 balance?",
        history=[("user", "hello"), ("assistant", "hi")],
    )
    assert answer == "Because a credit is missing."


def test_notes_generation_uses_the_notes_template(monkeypatch):
    seen: list[str] = []

    class RecordingModel(FakeListChatModel):
        def invoke(self, input, config=None, **kwargs):
            seen.append(str(input))
            return super().invoke(input, config, **kwargs)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        ledger_agent,
        "get_llm",
        lambda *a, **k: RecordingModel(responses=["Notes."]),
    )

    assert ledger_agent.generate_reconciliation_notes() == "Notes."
    assert "working paper" in seen[0]
