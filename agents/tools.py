"""One tool definition, rendered into each provider's schema.

The tool is backed by the ledger reconciliation in this repository, so the
agent templates do something real and verifiable rather than returning a
canned string.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ledger_data import get_ledger  # noqa: E402
from ledger_reconcile import reconcile  # noqa: E402

TOOL_NAME = "execute_project_query"
TOOL_DESCRIPTION = (
    "Run a retrieval query against the project ledger. Supported queries: "
    "'totals' (debit and credit totals), 'findings' (automated reconciliation "
    "exceptions), 'by_date' (per-date balance), 'by_account' (per-account "
    "movement), or 'all' for the full fact sheet."
)

# JSON Schema shared by both providers. `additionalProperties: false` plus a
# populated `required` is what strict tool use needs on the Anthropic side.
INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "One of: totals, findings, by_date, by_account, all.",
            "enum": ["totals", "findings", "by_date", "by_account", "all"],
        }
    },
    "required": ["query"],
    "additionalProperties": False,
}


def anthropic_tools() -> list[dict[str, Any]]:
    """Tool list in Anthropic's Messages API shape."""
    return [
        {
            "name": TOOL_NAME,
            "description": TOOL_DESCRIPTION,
            "input_schema": INPUT_SCHEMA,
        }
    ]


def openai_tools() -> list[dict[str, Any]]:
    """Tool list in OpenAI's chat-completions shape."""
    return [
        {
            "type": "function",
            "function": {
                "name": TOOL_NAME,
                "description": TOOL_DESCRIPTION,
                "parameters": INPUT_SCHEMA,
            },
        }
    ]


def execute_project_query(query: str) -> str:
    """Run the query against the ledger and return a text result.

    Raises ValueError on an unknown query so the caller can return a
    tool_result marked as an error instead of a plausible-looking wrong answer.
    """
    report = reconcile(get_ledger())

    if query == "totals":
        return json.dumps(
            {
                "total_debits": report.total_debits,
                "total_credits": report.total_credits,
                "difference": report.difference,
                "is_balanced": report.is_balanced,
            }
        )
    if query == "findings":
        return json.dumps(
            [
                {
                    "code": f.code,
                    "severity": f.severity,
                    "message": f.message,
                    "rows": list(f.rows),
                }
                for f in report.findings
            ]
        )
    if query == "by_date":
        return report.by_date.to_string(index=False)
    if query == "by_account":
        return report.by_account.to_string(index=False)
    if query == "all":
        return report.to_facts()

    raise ValueError(
        f"Unknown query {query!r}. Expected one of: "
        "totals, findings, by_date, by_account, all."
    )


def dispatch(name: str, arguments: dict[str, Any]) -> str:
    """Route a tool call by name. Unknown names raise, rather than silently no-op."""
    if name != TOOL_NAME:
        raise ValueError(f"Unknown tool {name!r}")
    return execute_project_query(**arguments)
