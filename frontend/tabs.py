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
        st.error(
            "Paste lease text or upload a .txt, .pdf, or .docx file before summarising."
        )
        return

    with st.spinner("Summarising lease..."):
        if lease_input_value["source"] == "file":
            response = call_api_files(
                "/summarise",
                {"file": uploaded_file_part(lease_input_value["file"])},
            )
        else:
            response = call_api(
                "/summarise-text",
                {"lease_text": lease_input_value["text"]},
            )

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


def render_s3_summarise_tab() -> None:
    leases = _load_s3_leases()
    if leases is None:
        return

    if not leases:
        st.info("No S3 lease files were returned.")
        return

    source = st.radio(
        "Source",
        ["S3 files", "Indexed leases"],
        horizontal=True,
        key="s3_summarise_source",
    )
    use_indexed_source = source == "Indexed leases"
    selected_lease = st.selectbox(
        "Lease",
        leases,
        format_func=_s3_lease_label,
        key="s3_summarise_lease",
    )

    submitted_summary = st.button(
        "Summarise Indexed Lease" if use_indexed_source else "Summarise S3 Lease",
        type="primary",
        use_container_width=True,
    )
    if submitted_summary:
        endpoint = "/summarise-indexed" if use_indexed_source else "/summarise-s3"
        spinner_text = (
            "Summarising indexed lease..."
            if use_indexed_source
            else "Summarising S3 lease..."
        )
        with st.spinner(spinner_text):
            response = call_api(endpoint, {"key": selected_lease["key"]})
        if response is not None:
            heading = "Indexed Summary" if use_indexed_source else "S3 Summary"
            render_summary_response(response, heading=heading)


def render_s3_compare_tab() -> None:
    leases = _load_s3_leases()
    if leases is None:
        return

    if not leases:
        st.info("No S3 lease files were returned.")
        return

    source = st.radio(
        "Source",
        ["S3 files", "Indexed leases"],
        horizontal=True,
        key="s3_compare_source",
    )
    use_indexed_source = source == "Indexed leases"
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
        "Compare Indexed Leases" if use_indexed_source else "Compare S3 Leases",
        type="primary",
        use_container_width=True,
    )
    if not submitted_compare:
        return

    endpoint = "/compare-indexed" if use_indexed_source else "/compare-s3"
    spinner_text = (
        "Comparing indexed leases..."
        if use_indexed_source
        else "Comparing S3 leases..."
    )
    with st.spinner(spinner_text):
        response = call_api(
            endpoint,
            {
                "lease_a_key": lease_a["key"],
                "lease_b_key": lease_b["key"],
            },
        )

    if response is None:
        return

    render_compare_response(response)


def render_s3_index_tab() -> None:
    status = call_api_get("/rag/status")
    if isinstance(status, dict):
        left, middle, right = st.columns(3)
        with left:
            st.metric("Indexed leases", status.get("indexed_lease_count", 0))
        with middle:
            st.metric("Chunks", status.get("chunk_count", 0))
        with right:
            st.metric("Collection", status.get("collection_name", "lease_chunks"))

        last_indexed_at = status.get("last_indexed_at")
        if last_indexed_at:
            st.caption(f"Last indexed: {last_indexed_at}")

    submitted = st.button("Index S3 Leases", type="primary", use_container_width=True)
    if not submitted:
        return

    with st.spinner("Indexing S3 leases..."):
        response = call_api("/rag/index", {})

    if response is None:
        return

    st.success(
        "Indexed "
        f"{response.get('indexed_lease_count', 0)} leases into "
        f"{response.get('indexed_chunk_count', 0)} chunks."
    )

    skipped_files = response.get("skipped_files") or []
    failed_files = response.get("failed_files") or []
    if skipped_files:
        st.warning("Skipped files: " + ", ".join(skipped_files))
    if failed_files:
        st.warning("Failed files: " + ", ".join(failed_files))


def render_s3_search_tab() -> None:
    question = st.text_input("Question", key="rag_search_question")
    submitted = st.button(
        "Search Indexed Leases",
        type="primary",
        use_container_width=True,
    )
    if not submitted:
        return

    if not question.strip():
        st.error("Enter a question before searching.")
        return

    with st.spinner("Searching indexed leases..."):
        response = call_api("/rag/search", {"question": question, "top_k": 5})

    if response is None:
        return

    matches = response.get("matches") or []
    if not matches:
        st.info("No matching indexed lease text was found.")
        return

    for match in matches:
        score = match.get("score")
        score_text = "" if score is None else f" - score {float(score):.2f}"
        st.markdown(
            f"#### {match.get('filename', match.get('key', 'Lease'))}{score_text}"
        )
        st.caption(str(match.get("key", "")))
        st.write(match.get("snippet", ""))


def render_s3_chat_tab() -> None:
    leases = _load_s3_leases()
    if leases is None:
        return

    history = st.session_state.setdefault("rag_chat_history", [])
    selected_leases = st.multiselect(
        "Leases",
        leases,
        format_func=_s3_lease_label,
        key="rag_chat_lease_keys",
    )
    show_sources = st.toggle("Show sources", value=False, key="rag_chat_show_sources")
    question = st.text_input("Question", key="rag_chat_question")

    left, right = st.columns(2)
    with left:
        submitted = st.button("Send", type="primary", use_container_width=True)
    with right:
        clear_chat = st.button("Clear Chat", use_container_width=True)

    if clear_chat:
        history.clear()

    if submitted:
        if not question.strip():
            st.error("Enter a question before sending.")
        else:
            payload = {
                "question": question,
                "lease_keys": [str(lease["key"]) for lease in selected_leases],
                "history": _chat_history_for_api(history),
                "top_k": 5,
            }
            with st.spinner("Answering from indexed lease text..."):
                response = call_api("/rag/chat", payload)

            if response is not None:
                history.append({"role": "user", "content": question})
                history.append(
                    {
                        "role": "assistant",
                        "content": response.get("answer", ""),
                        "citations": response.get("citations") or [],
                    }
                )

    for item in history:
        speaker = "You" if item.get("role") == "user" else "Assistant"
        st.markdown(f"**{speaker}:** {item.get('content', '')}")
        if show_sources and item.get("role") == "assistant":
            _render_rag_citations(item.get("citations") or [])


def _load_s3_leases() -> list[S3LeaseOption] | None:
    leases_response = call_api_get("/s3/leases")
    if leases_response is None:
        return None
    return _s3_lease_options(leases_response)


def _chat_history_for_api(history: list[object]) -> list[dict[str, str]]:
    api_history: list[dict[str, str]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            api_history.append({"role": role, "content": content})
    return api_history


def _render_rag_citations(citations: list[object]) -> None:
    if not citations:
        return

    st.markdown("#### Sources")
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        st.caption(str(citation.get("key", "")))
        st.write(citation.get("snippet", ""))


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
