import streamlit as st

from frontend.api_client import call_api, call_api_files
from frontend.lease_inputs import (
    has_lease_input,
    lease_input,
    lease_input_file_part,
    uploaded_file_part,
)
from frontend.renderers import render_compare_response, render_summary_response


def render_summarise_tab() -> None:
    lease_input_value = lease_input(
        key_prefix="summarise",
        label="Lease text",
        upload_label="Upload lease text",
    )

    submitted = st.button("Summarise Lease", type="primary", use_container_width=True)
    if not submitted:
        return

    if not has_lease_input(lease_input_value):
        st.error("Paste lease text or upload a .txt, .pdf, or .docx file before summarising.")
        return

    with st.spinner("Summarising lease..."):
        if lease_input_value["source"] == "file":
            response = call_api_files(
                "/summarise",
                {"file": uploaded_file_part(lease_input_value["file"])},
            )
        else:
            response = call_api("/summarise-text", {"lease_text": lease_input_value["text"]})

    if response is None:
        return

    render_summary_response(response)


def render_compare_tab() -> None:
    left, right = st.columns(2)

    with left:
        lease_a = lease_input(
            key_prefix="compare_a",
            label="Lease A",
            upload_label="Upload Lease A",
        )

    with right:
        lease_b = lease_input(
            key_prefix="compare_b",
            label="Lease B",
            upload_label="Upload Lease B",
        )

    submitted = st.button("Compare Leases", type="primary", use_container_width=True)
    if not submitted:
        return

    if not has_lease_input(lease_a) or not has_lease_input(lease_b):
        st.error("Provide both leases using pasted text or .txt, .pdf, or .docx upload.")
        return

    with st.spinner("Comparing leases..."):
        if lease_a["source"] == "text" and lease_b["source"] == "text":
            response = call_api(
                "/compare-text",
                {"lease_a": lease_a["text"], "lease_b": lease_b["text"]},
            )
        else:
            response = call_api_files(
                "/compare",
                {
                    "lease_a": lease_input_file_part(lease_a, "lease_a.txt"),
                    "lease_b": lease_input_file_part(lease_b, "lease_b.txt"),
                },
            )

    if response is None:
        return

    render_compare_response(response)
