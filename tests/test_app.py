"""Smoke tests for the Streamlit UI.

`AppTest` runs app.py the way Streamlit does, so an exception raised anywhere in
the script -- a bad column reference, a renamed helper -- fails here rather than
in front of a user. It cannot drive the canvas-based data editor, so the widgets
it can reach are the ones asserted on.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

# AppTest resolves relative paths against this file, so point at the repo root.
APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def app(monkeypatch):
    # Force the offline path so no test can reach for a real API key.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return AppTest.from_file(APP, default_timeout=60).run()


def test_app_runs_without_exceptions(app):
    assert not app.exception


def test_headline_metrics_show_the_computed_totals(app):
    values = {m.label: m.value for m in app.metric}
    assert values["Total debits"] == "700.00"
    assert values["Total credits"] == "2,000.00"
    assert values["Difference"] == "-1,300.00"


def test_all_three_tabs_are_present(app):
    assert len(app.tabs) == 3


def test_findings_are_surfaced_in_the_ui(app):
    text = " ".join(str(e.value) for e in app.error)
    assert "LEDGER_IMBALANCE" in text
    assert "DATE_IMBALANCE" in text


def test_switching_to_the_corrected_ledger_balances(app):
    app.radio[0].set_value("Corrected ledger (balances)").run()
    assert not app.exception
    values = {m.label: m.value for m in app.metric}
    assert values["Total debits"] == "2,100.00"
    assert values["Total credits"] == "2,100.00"
    assert values["Difference"] == "0.00"


def test_analyze_button_produces_a_report_offline(app):
    next(b for b in app.button if b.label == "Analyze ledger").click().run()
    assert not app.exception
    markdown = " ".join(m.value for m in app.markdown)
    assert "Reconciliation summary" in markdown
    assert "1,300.00" in markdown


def test_notes_button_produces_notes_offline(app):
    next(b for b in app.button if b.label == "Draft reconciliation notes").click().run()
    assert not app.exception
    assert any("Reconciliation summary" in m.value for m in app.markdown)


def test_chat_is_gated_without_an_api_key(app):
    assert any("Chat needs a live model" in str(i.value) for i in app.info)
