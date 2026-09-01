"""Sample ledger data for the Ledger Analysis Agent.

The default ledger is intentionally *broken*: it contains the imbalances and
posting-convention errors that the agent is meant to find. A corrected version
is provided alongside it so reconciliation notes have something to aim at.
"""

from __future__ import annotations

import pandas as pd

# Column contract shared by every module that touches a ledger.
DATE = "Date"
DESCRIPTION = "Description"
ACCOUNT = "Account"
DEBIT = "Debit"
CREDIT = "Credit"

REQUIRED_COLUMNS = [DATE, DESCRIPTION, ACCOUNT, DEBIT, CREDIT]


def get_ledger() -> pd.DataFrame:
    """Return the sample ledger from the lab.

    Known defects, kept deliberately so the agent has something to report:

    * 2025-07-01 balances, but the entries are reversed. A customer payment
      should *debit* Accounts Receivable's contra side and *credit* it, while
      revenue is credited, not debited.
    * 2025-07-02 posts two debits (Office Supplies, Cash) with no credit, so
      the day is out of balance by 200. "Cash Paid" should be a credit.
    * 2025-07-03 credits Revenue 1500 with no offsetting debit.
    """
    data = {
        DATE: [
            "2025-07-01",
            "2025-07-01",
            "2025-07-02",
            "2025-07-02",
            "2025-07-03",
        ],
        DESCRIPTION: [
            "Customer Payment",
            "Revenue Recorded",
            "Office Supplies",
            "Cash Paid",
            "Consulting Income",
        ],
        ACCOUNT: [
            "Accounts Receivable",
            "Revenue",
            "Office Supplies",
            "Cash",
            "Revenue",
        ],
        DEBIT: [0, 500, 100, 100, 0],
        CREDIT: [500, 0, 0, 0, 1500],
    }
    return pd.DataFrame(data)


def get_balanced_ledger() -> pd.DataFrame:
    """Return the same business events, posted correctly.

    Useful as a "known good" contrast when demonstrating the agent, and as a
    fixture for tests that need a ledger with no findings.
    """
    data = {
        DATE: [
            "2025-07-01",
            "2025-07-01",
            "2025-07-02",
            "2025-07-02",
            "2025-07-03",
            "2025-07-03",
        ],
        DESCRIPTION: [
            "Customer Payment",
            "Revenue Recorded",
            "Office Supplies",
            "Cash Paid",
            "Consulting Income",
            "Consulting Income",
        ],
        ACCOUNT: [
            "Accounts Receivable",
            "Revenue",
            "Office Supplies",
            "Cash",
            "Accounts Receivable",
            "Revenue",
        ],
        DEBIT: [500, 0, 100, 0, 1500, 0],
        CREDIT: [0, 500, 0, 100, 0, 1500],
    }
    return pd.DataFrame(data)


def validate_ledger(df: pd.DataFrame) -> pd.DataFrame:
    """Check the frame has the expected shape, and return it with numeric amounts.

    Raises ValueError on a missing column or an unparseable amount, so callers
    fail loudly rather than reconciling silently-wrong numbers.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Ledger is missing required column(s): {', '.join(missing)}. "
            f"Expected: {', '.join(REQUIRED_COLUMNS)}"
        )

    out = df.copy()
    for column in (DEBIT, CREDIT):
        coerced = pd.to_numeric(out[column], errors="coerce")
        bad = coerced.isna() & out[column].notna()
        if bad.any():
            rows = ", ".join(str(i) for i in out.index[bad])
            raise ValueError(f"Non-numeric value in '{column}' at row(s): {rows}")
        out[column] = coerced.fillna(0.0).astype(float)

    return out
