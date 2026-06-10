import streamlit as st

from frontend.api_client import call_api, call_api_files, call_api_get
from frontend.lease_inputs import (
    has_lease_input,
    lease_input,
    lease_input_file_part,
    uploaded_file_part,
)
from frontend.renderers import render_compare_response, render_summary_response


S3LeaseOption = dict[str, object]


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


def render_s3_leases_tab() -> None:
    leases_response = call_api_get("/s3/leases")
    if leases_response is None:
        return

    leases = _s3_lease_options(leases_response)
    if not leases:
        st.info("No S3 lease files were returned.")
        return

    st.subheader("Summarise from S3")
    selected_lease = st.selectbox(
        "Lease",
        leases,
        format_func=_s3_lease_label,
        key="s3_summarise_lease",
    )

    submitted_summary = st.button(
        "Summarise S3 Lease",
        type="primary",
        use_container_width=True,
    )
    if submitted_summary:
        with st.spinner("Summarising S3 lease..."):
            response = call_api("/summarise-s3", {"key": selected_lease["key"]})
        if response is not None:
            render_summary_response(response, heading="S3 Summary")

    st.subheader("Compare from S3")
    left, right = st.columns(2)
    with left:
        lease_a = st.selectbox(
            "Lease A",
            leases,
            format_func=_s3_lease_label,
            key="s3_compare_lease_a",
        )
    with right:
        lease_b = st.selectbox(
            "Lease B",
            leases,
            format_func=_s3_lease_label,
            index=1 if len(leases) > 1 else 0,
            key="s3_compare_lease_b",
        )

    submitted_compare = st.button(
        "Compare S3 Leases",
        type="primary",
        use_container_width=True,
    )
    if not submitted_compare:
        return

    with st.spinner("Comparing S3 leases..."):
        response = call_api(
            "/compare-s3",
            {
                "lease_a_key": lease_a["key"],
                "lease_b_key": lease_b["key"],
            },
        )

    if response is None:
        return

    render_compare_response(response)


def _s3_lease_options(response: object) -> list[S3LeaseOption]:
    if not isinstance(response, list):
        return []
    return [
        lease
        for lease in response
        if isinstance(lease, dict) and isinstance(lease.get("key"), str)
    ]


def _s3_lease_label(lease: S3LeaseOption) -> str:
    key = str(lease.get("key", ""))
    filename = str(lease.get("filename") or key.rsplit("/", 1)[-1] or key)
    return f"{filename} ({key})"
