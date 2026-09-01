"""Deterministic reconciliation checks over a ledger DataFrame.

Every number an analyst would act on is computed here, in Pandas, and only
*then* handed to the language model to narrate. A model that does its own
arithmetic will eventually add 700 and 1500 and get 2100 on a bad day; on a
finance tool that is not an acceptable failure mode. The LLM's job in this
project is explanation, not calculation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from ledger_data import ACCOUNT, CREDIT, DATE, DEBIT, DESCRIPTION, validate_ledger

# Currency is carried as float for Pandas' sake, so compare with a tolerance of
# half a cent rather than testing exact equality.
TOLERANCE = 0.005

Severity = Literal["error", "warning"]

# Normal balances by account type. Assets and expenses increase on the debit
# side; revenue, liabilities and equity increase on the credit side.
DEBIT_NORMAL = {"asset", "expense"}
CREDIT_NORMAL = {"revenue", "liability", "equity"}

# Keyword -> account type. Checked longest-first so "accounts payable" wins
# over "accounts receivable"-style prefix collisions.
ACCOUNT_TYPE_KEYWORDS: dict[str, str] = {
    "accounts receivable": "asset",
    "accounts payable": "liability",
    "retained earnings": "equity",
    "notes payable": "liability",
    "prepaid": "asset",
    "inventory": "asset",
    "equipment": "asset",
    "supplies": "expense",
    "revenue": "revenue",
    "income": "revenue",
    "sales": "revenue",
    "expense": "expense",
    "rent": "expense",
    "salaries": "expense",
    "payroll": "expense",
    "wages": "expense",
    "utilities": "expense",
    "payable": "liability",
    "receivable": "asset",
    "capital": "equity",
    "equity": "equity",
    "bank": "asset",
    "cash": "asset",
}


def classify_account(account: str) -> str:
    """Map an account name to asset/liability/equity/revenue/expense.

    Returns "unknown" when nothing matches. Unknown accounts are skipped by the
    convention check rather than guessed at, so an unfamiliar chart of accounts
    produces no findings instead of wrong ones.
    """
    name = str(account).strip().lower()
    for keyword in sorted(ACCOUNT_TYPE_KEYWORDS, key=len, reverse=True):
        if keyword in name:
            return ACCOUNT_TYPE_KEYWORDS[keyword]
    return "unknown"


@dataclass(frozen=True)
class Finding:
    """One thing wrong with the ledger."""

    code: str
    severity: Severity
    message: str
    rows: tuple[int, ...] = ()

    def __str__(self) -> str:
        where = f" (row(s) {', '.join(str(r) for r in self.rows)})" if self.rows else ""
        return f"[{self.severity.upper()}] {self.code}: {self.message}{where}"


@dataclass
class ReconciliationReport:
    """Everything the checks found, plus the totals they were derived from."""

    total_debits: float
    total_credits: float
    difference: float  # debits - credits
    is_balanced: bool
    by_date: pd.DataFrame
    by_account: pd.DataFrame
    findings: list[Finding] = field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    def to_facts(self) -> str:
        """Render the report as the plain-text fact sheet given to the model.

        The prompt gets these pre-computed lines instead of raw numbers to add
        up, which is what keeps the narration anchored to real figures.
        """
        status = "BALANCED" if self.is_balanced else "OUT OF BALANCE"
        lines = [
            f"Overall status: {status}",
            f"Total debits:  {self.total_debits:,.2f}",
            f"Total credits: {self.total_credits:,.2f}",
            f"Difference (debits - credits): {self.difference:,.2f}",
            "",
            "Per-date totals (each date's journal entries should balance):",
        ]
        for row in self.by_date.itertuples(index=False):
            flag = "OK" if abs(row.Difference) < TOLERANCE else "IMBALANCED"
            lines.append(
                f"  {row.Date}: debits {row.Debit:,.2f}, credits {row.Credit:,.2f}, "
                f"difference {row.Difference:,.2f} [{flag}]"
            )

        lines += [
            "",
            "Per-account net movement for this period (debits - credits).",
            "These are movements, not closing balances: no opening balances are loaded,",
            "so a net credit on an asset means it decreased, not that it is overdrawn.",
        ]
        for row in self.by_account.itertuples(index=False):
            lines.append(
                f"  {row.Account} ({row.Type}): debits {row.Debit:,.2f}, "
                f"credits {row.Credit:,.2f}, net {row.Net:,.2f}"
            )

        lines += ["", f"Automated findings ({len(self.findings)}):"]
        if self.findings:
            lines += [f"  {finding}" for finding in self.findings]
        else:
            lines.append("  None. All automated checks passed.")

        return "\n".join(lines)


def _summarise_by_date(df: pd.DataFrame) -> pd.DataFrame:
    by_date = (
        df.groupby(DATE, as_index=False)[[DEBIT, CREDIT]]
        .sum()
        .sort_values(DATE, ignore_index=True)
    )
    by_date["Difference"] = (by_date[DEBIT] - by_date[CREDIT]).round(2)
    return by_date


def _summarise_by_account(df: pd.DataFrame) -> pd.DataFrame:
    by_account = (
        df.groupby(ACCOUNT, as_index=False)[[DEBIT, CREDIT]]
        .sum()
        .sort_values(ACCOUNT, ignore_index=True)
    )
    by_account["Net"] = (by_account[DEBIT] - by_account[CREDIT]).round(2)
    by_account["Type"] = by_account[ACCOUNT].map(classify_account)
    return by_account


def _check_row_hygiene(df: pd.DataFrame) -> list[Finding]:
    """Per-row sanity checks that do not depend on any total."""
    findings: list[Finding] = []

    negative = df[(df[DEBIT] < 0) | (df[CREDIT] < 0)]
    if not negative.empty:
        findings.append(
            Finding(
                code="NEGATIVE_AMOUNT",
                severity="error",
                message=(
                    "Negative debit or credit amounts found. Reverse the entry with an "
                    "opposite-side posting instead of using a negative number."
                ),
                rows=tuple(negative.index),
            )
        )

    empty = df[(df[DEBIT].abs() < TOLERANCE) & (df[CREDIT].abs() < TOLERANCE)]
    if not empty.empty:
        findings.append(
            Finding(
                code="EMPTY_ENTRY",
                severity="warning",
                message="Entry has neither a debit nor a credit amount and posts nothing.",
                rows=tuple(empty.index),
            )
        )

    both = df[(df[DEBIT].abs() >= TOLERANCE) & (df[CREDIT].abs() >= TOLERANCE)]
    if not both.empty:
        findings.append(
            Finding(
                code="DEBIT_AND_CREDIT",
                severity="warning",
                message=(
                    "Entry carries both a debit and a credit on the same line, which is "
                    "ambiguous. Split it into separate lines."
                ),
                rows=tuple(both.index),
            )
        )

    return findings


def _check_posting_conventions(df: pd.DataFrame) -> list[Finding]:
    """Flag entries posted against an account's normal balance.

    Only one direction of this check carries signal. Crediting an asset or
    expense account is routine -- every cash payment does it, as does every
    refund and depreciation entry -- so flagging it would fire on most correct
    ledgers. Debiting a revenue, liability or equity account is genuinely
    uncommon and shows up mainly on reversals and contra entries, so it is
    worth an analyst's attention. It stays a warning rather than an error
    because those reversals are legitimate.
    """
    suspect: list[int] = []
    for idx, row in df.iterrows():
        if classify_account(row[ACCOUNT]) in CREDIT_NORMAL and row[DEBIT] >= TOLERANCE:
            suspect.append(idx)

    if not suspect:
        return []

    names = ", ".join(sorted({str(df.at[i, ACCOUNT]) for i in suspect}))
    return [
        Finding(
            code="UNEXPECTED_DEBIT",
            severity="warning",
            message=(
                f"Debit posted to a credit-normal account ({names}). Recording revenue "
                "credits the account, so confirm this is a reversal, refund or "
                "correction rather than a reversed entry."
            ),
            rows=tuple(suspect),
        )
    ]


def reconcile(df: pd.DataFrame) -> ReconciliationReport:
    """Run every check over `df` and return the full report."""
    clean = validate_ledger(df)

    total_debits = round(float(clean[DEBIT].sum()), 2)
    total_credits = round(float(clean[CREDIT].sum()), 2)
    difference = round(total_debits - total_credits, 2)
    is_balanced = abs(difference) < TOLERANCE

    by_date = _summarise_by_date(clean)
    by_account = _summarise_by_account(clean)

    findings: list[Finding] = []
    if not is_balanced:
        heavier = "credits" if difference < 0 else "debits"
        findings.append(
            Finding(
                code="LEDGER_IMBALANCE",
                severity="error",
                message=(
                    f"Total debits ({total_debits:,.2f}) do not equal total credits "
                    f"({total_credits:,.2f}); {heavier} exceed the other side by "
                    f"{abs(difference):,.2f}."
                ),
            )
        )

    for row in by_date.itertuples(index=False):
        if abs(row.Difference) >= TOLERANCE:
            heavier = "credits" if row.Difference < 0 else "debits"
            affected = clean.index[clean[DATE] == row.Date]
            findings.append(
                Finding(
                    code="DATE_IMBALANCE",
                    severity="error",
                    message=(
                        f"Entries dated {row.Date} do not balance: debits {row.Debit:,.2f} "
                        f"vs credits {row.Credit:,.2f}, with {heavier} exceeding the other "
                        f"side by {abs(row.Difference):,.2f}. A missing offsetting entry is "
                        "the likeliest cause."
                    ),
                    rows=tuple(affected),
                )
            )

    findings += _check_row_hygiene(clean)
    findings += _check_posting_conventions(clean)

    return ReconciliationReport(
        total_debits=total_debits,
        total_credits=total_credits,
        difference=difference,
        is_balanced=is_balanced,
        by_date=by_date,
        by_account=by_account,
        findings=findings,
    )


def format_ledger_table(df: pd.DataFrame) -> str:
    """Render the ledger as a fixed-width table with the row numbers findings cite."""
    display = validate_ledger(df).copy()
    display.insert(0, "Row", display.index)
    display[DEBIT] = display[DEBIT].map("{:,.2f}".format)
    display[CREDIT] = display[CREDIT].map("{:,.2f}".format)
    return display[["Row", DATE, DESCRIPTION, ACCOUNT, DEBIT, CREDIT]].to_string(index=False)
