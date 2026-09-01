"""Streamlit front end for the Ledger Analysis Agent.

Three tabs: inspect and edit the ledger, run a full analysis, and chat with it.
The reconciliation figures shown in the header are computed in Pandas and update
as you edit, independently of any model call.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ledger_agent import (
    MissingAPIKeyError,
    analyze_ledger,
    api_key_configured,
    chat_with_ledger,
    generate_reconciliation_notes,
)
from ledger_data import get_balanced_ledger, get_ledger, validate_ledger
from ledger_reconcile import reconcile

st.set_page_config(page_title="Ledger Analysis Agent", page_icon="📒", layout="wide")

SAMPLES = {
    "Sample ledger (contains errors)": get_ledger,
    "Corrected ledger (balances)": get_balanced_ledger,
}


def load_ledger() -> pd.DataFrame:
    """Resolve the ledger from the sidebar controls, falling back to the sample."""
    choice = st.session_state.get("source", next(iter(SAMPLES)))
    upload = st.session_state.get("upload")

    if choice == "Upload a CSV" and upload is not None:
        try:
            return validate_ledger(pd.read_csv(upload))
        except Exception as exc:  # surfaced to the user rather than crashing the app
            st.sidebar.error(f"Could not read that CSV: {exc}")
            return get_ledger()

    return SAMPLES.get(choice, get_ledger)()


st.title("Ledger Analysis Agent 📒")
st.caption(
    "Debits and credits are reconciled in Pandas; the model explains the findings "
    "rather than computing them."
)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")

    st.radio(
        "Ledger source",
        [*SAMPLES, "Upload a CSV"],
        key="source",
    )
    if st.session_state.get("source") == "Upload a CSV":
        st.file_uploader(
            "CSV with Date, Description, Account, Debit, Credit",
            type="csv",
            key="upload",
        )

    st.text_input("Model", value="gpt-4o-mini", key="model")

    has_key = api_key_configured()
    if has_key:
        st.success("OPENAI_API_KEY detected")
    else:
        st.warning(
            "No OPENAI_API_KEY found. Reconciliation still runs; the written "
            "analysis and chat need a key."
        )

    offline = st.checkbox(
        "Offline mode (skip model calls)",
        value=not has_key,
        disabled=not has_key,
        help="Reports the computed findings without generating a narrative.",
        key="offline",
    )

    if st.button("Reset conversation"):
        st.session_state.messages = []
        st.rerun()

ledger = load_ledger()

# ------------------------------------------------------------- ledger tabs
# Reserved before the tabs so the totals render above them, but filled after the
# data editor has run, so they reflect any edits made this pass.
metrics_slot = st.container()

tab_ledger, tab_analysis, tab_chat = st.tabs(["🧾 Ledger", "🔍 Analysis", "💬 Chat"])

with tab_ledger:
    st.subheader("Ledger data")
    st.caption("Edit any cell to see the reconciliation update immediately.")
    edited = st.data_editor(
        ledger,
        num_rows="dynamic",
        width="stretch",
        key="editor",
    )
    try:
        ledger = validate_ledger(edited)
    except ValueError as exc:
        st.error(str(exc))
        ledger = load_ledger()

report = reconcile(ledger)

# Live totals, visible from every tab so the numbers are always in view.
with metrics_slot:
    left, middle, right = st.columns(3)
    left.metric("Total debits", f"{report.total_debits:,.2f}")
    middle.metric("Total credits", f"{report.total_credits:,.2f}")
    right.metric(
        "Difference",
        f"{report.difference:,.2f}",
        delta="Balanced" if report.is_balanced else "Out of balance",
        delta_color="normal" if report.is_balanced else "inverse",
    )

with tab_ledger:
    if report.is_balanced:
        st.success("Debits equal credits.")
    else:
        st.error(
            f"Debits and credits differ by {abs(report.difference):,.2f}."
        )

    st.subheader("Automated checks")
    if report.findings:
        for finding in report.findings:
            box = st.error if finding.severity == "error" else st.warning
            rows = (
                f"  \nRows: {', '.join(str(r) for r in finding.rows)}"
                if finding.rows
                else ""
            )
            box(f"**{finding.code}** — {finding.message}{rows}")
    else:
        st.success("All automated checks passed.")

    col_date, col_account = st.columns(2)
    with col_date:
        st.subheader("By date")
        st.dataframe(report.by_date, width="stretch", hide_index=True)
    with col_account:
        st.subheader("By account")
        st.caption("Period movement, not closing balance.")
        st.dataframe(report.by_account, width="stretch", hide_index=True)

with tab_analysis:
    st.subheader("Ledger analysis")
    run_col, notes_col = st.columns(2)
    run_analysis = run_col.button("Analyze ledger", width="stretch", type="primary")
    run_notes = notes_col.button("Draft reconciliation notes", width="stretch")

    if run_analysis:
        with st.spinner("Analyzing…"):
            result = analyze_ledger(
                ledger,
                model=st.session_state.get("model") or None,
                offline=st.session_state.get("offline", False),
            )
        if not result.used_llm:
            st.info("Offline mode — findings shown without a generated narrative.")
        st.markdown(result.narrative)

    if run_notes:
        with st.spinner("Drafting notes…"):
            notes = generate_reconciliation_notes(
                ledger,
                model=st.session_state.get("model") or None,
                offline=st.session_state.get("offline", False),
            )
        st.markdown(notes)
        st.download_button(
            "Download notes (Markdown)",
            data=notes,
            file_name="reconciliation_notes.md",
            mime="text/markdown",
        )

    with st.expander("Facts given to the model"):
        st.code(report.to_facts(), language="text")

with tab_chat:
    st.subheader("Chat with the ledger")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not api_key_configured():
        st.info("Chat needs a live model. Set OPENAI_API_KEY and restart the app.")

    for role, text in st.session_state.messages:
        with st.chat_message(role):
            st.markdown(text)

    question = st.chat_input(
        "Ask about the ledger — e.g. why doesn't 2025-07-03 balance?",
        disabled=not api_key_configured(),
    )
    if question:
        st.session_state.messages.append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking…"):
                    answer = chat_with_ledger(
                        question,
                        ledger,
                        history=st.session_state.messages[:-1],
                        model=st.session_state.get("model") or None,
                    )
                st.markdown(answer)
                st.session_state.messages.append(("assistant", answer))
            except MissingAPIKeyError as exc:
                st.error(str(exc))
            except Exception as exc:  # keep a bad model call from killing the app
                st.error(f"The model call failed: {exc}")
