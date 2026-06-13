import time

import streamlit as st

from frontend.api_client import call_api, call_api_delete, call_api_files, call_api_get
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
    _, refresh_col = st.columns([6, 1])
    with refresh_col:
        st.button("Refresh", use_container_width=True, key="index_tab_refresh")

    status = call_api_get("/rag/status")
    if isinstance(status, dict):
        left, middle, summary_column, right = st.columns(4)
        with left:
            st.metric("Indexed Leases", status.get("indexed_lease_count", 0))
        with middle:
            st.metric("Total Chunks", status.get("chunk_count", 0))
        with summary_column:
            st.metric("Summaries", status.get("indexed_summary_count", 0))
        with right:
            st.metric("Collections", status.get("collection_name", "per-file"))

        last_indexed_at = status.get("last_indexed_at")
        if last_indexed_at:
            st.caption(f"Last indexed: {last_indexed_at}")

        file_collections = status.get("file_collections") or []
        if file_collections:
            st.subheader("Per-File Collections")
            rows = [
                {
                    "Filename": fc.get("filename", ""),
                    "S3 Key": fc.get("s3_key", ""),
                    "Collection": fc.get("collection_name", ""),
                    "Chunks": fc.get("chunk_count", 0),
                    "Indexed At": (fc.get("indexed_at") or "")[:19].replace("T", " "),
                }
                for fc in file_collections
            ]
            st.dataframe(
                rows,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Filename": st.column_config.TextColumn("Filename", width="medium"),
                    "S3 Key": st.column_config.TextColumn("S3 Key", width="medium"),
                    "Collection": st.column_config.TextColumn("Collection Name", width="large"),
                    "Chunks": st.column_config.NumberColumn("Chunks", width="small"),
                    "Indexed At": st.column_config.TextColumn("Indexed At", width="medium"),
                },
            )
        else:
            st.info("No collections indexed yet. Click 'Index S3 Leases' to start.")

    job_status = call_api_get("/rag/index/status")
    job_is_running = _render_s3_index_job_status(job_status)

    submitted = st.button("Index S3 Leases", type="primary", use_container_width=True)
    if submitted:
        with st.spinner("Starting S3 lease indexing..."):
            job_status = call_api("/rag/index", {})
        job_is_running = _render_s3_index_job_status(job_status)

    if job_is_running:
        _schedule_index_status_refresh()
        return

    if not submitted:
        return


def _render_s3_index_job_status(job_status: object) -> bool:
    if not isinstance(job_status, dict):
        return False

    status = job_status.get("status")
    if status == "idle" or status is None:
        if "indexed_lease_count" in job_status:
            _render_s3_index_result(job_status)
        return False

    if status == "running":
        started_at = job_status.get("started_at")
        suffix = f" Started: {started_at}" if started_at else ""
        st.info(f"Indexing S3 leases is running in the background.{suffix}")
        _render_s3_index_progress(job_status)
        return True

    if status == "failed":
        st.error(f"Indexing failed: {job_status.get('error', 'Unknown error')}")
        return False

    if status == "completed":
        result = job_status.get("result")
        if isinstance(result, dict):
            _render_s3_index_result(result)
        else:
            st.success("Indexing completed.")
        return False

    return False


def _render_s3_index_progress(job_status: dict[str, object]) -> None:
    progress_value = _safe_progress_value(job_status.get("progress_percent"))
    message = str(job_status.get("message") or "Indexing S3 leases...")
    current = int(job_status.get("progress_current") or 0)
    total = int(job_status.get("progress_total") or 0)
    progress_label = message
    if total:
        progress_label = f"{message} ({current}/{total})"
    st.progress(progress_value, text=progress_label)

    current_key = job_status.get("current_key")
    if current_key:
        st.caption(f"Current file: {current_key}")


def _safe_progress_value(value: object) -> float:
    try:
        progress_value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(max(progress_value, 0.0), 1.0)


def _render_s3_index_result(response: dict[str, object]) -> None:
    if response is None:
        return

    st.success(
        "Indexed "
        f"{response.get('indexed_lease_count', 0)} leases into "
        f"{response.get('indexed_chunk_count', 0)} chunks and stored "
        f"{response.get('summarised_lease_count', 0)} summaries."
    )

    skipped_files = response.get("skipped_files") or []
    failed_files = response.get("failed_files") or []
    summary_failed_files = response.get("summary_failed_files") or []
    if skipped_files:
        st.warning("Skipped files: " + ", ".join(skipped_files))
    if failed_files:
        st.warning("Failed files: " + ", ".join(failed_files))
    if summary_failed_files:
        st.warning("Summary failed files: " + ", ".join(summary_failed_files))


def _schedule_index_status_refresh() -> None:
    time.sleep(2)
    if hasattr(st, "rerun"):
        st.rerun()



def render_s3_chat_tab() -> None:
    leases = _load_s3_leases()
    if leases is None:
        return

    history = st.session_state.setdefault("rag_chat_history", [])
    side_panel, chat_panel = st.columns(2)

    with side_panel:
        selected_leases = _render_chat_side_panel(leases)

    with chat_panel:
        show_sources = st.toggle(
            "Show sources",
            value=False,
            key="rag_chat_show_sources",
        )
        chat_window = st.container(height=520, border=True)
        with chat_window:
            _render_chat_history(history, show_sources)

        clear_chat = st.button("Clear History", use_container_width=True)
        if clear_chat:
            _clear_current_chat()
            _rerun_if_available()

        question = st.chat_input(
            "Ask about the indexed leases...",
            key="rag_chat_question",
        )
        if not question:
            return

        cleaned_question = question.strip()
        if not cleaned_question:
            return

        api_history = _chat_history_for_api(history)
        history.append({"role": "user", "content": cleaned_question})

        payload = {
            "question": cleaned_question,
            "lease_keys": [str(lease["key"]) for lease in selected_leases],
            "history": api_history,
            "top_k": 5,
        }
        session_id = st.session_state.get("rag_chat_session_id")
        if isinstance(session_id, str) and session_id:
            payload["session_id"] = session_id

        with chat_window:
            _render_chat_message(history[-1], show_sources)
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                response_placeholder.markdown("_Searching indexed lease text..._")
                response = call_api("/rag/chat", payload)

                if response is None:
                    response_placeholder.empty()
                    return

                if response.get("session_id"):
                    st.session_state["rag_chat_session_id"] = response["session_id"]
                assistant_message = _assistant_message_from_response(response)
                history.append(assistant_message)
                response_placeholder.markdown(str(assistant_message["content"]))
                _render_rag_chat_warnings(assistant_message)
                if show_sources:
                    _render_rag_citations(assistant_message.get("citations") or [])


def _render_chat_side_panel(leases: list[S3LeaseOption]) -> list[S3LeaseOption]:
    st.markdown("#### Leases")
    _apply_pending_lease_selection(leases)
    selected_leases = st.multiselect(
        "Leases",
        leases,
        format_func=_s3_lease_label,
        key="rag_chat_lease_keys",
    )

    st.markdown("#### Saved Chats")
    sessions = _load_chat_sessions()
    if not sessions:
        st.info("No saved chats yet.")
    for session in sessions:
        if not isinstance(session, dict):
            continue
        session_id = str(session.get("session_id") or "")
        if not session_id:
            continue
        if st.button(
            _chat_session_label(session),
            use_container_width=True,
            key=f"load_chat_{session_id}",
        ):
            _load_chat_session(session_id, leases)
            _rerun_if_available()

    if st.button("New Chat", use_container_width=True):
        _clear_current_chat(clear_leases=True)
        _rerun_if_available()

    active_session_id = st.session_state.get("rag_chat_session_id")
    if isinstance(active_session_id, str) and active_session_id:
        if st.button("Delete Saved Chat", use_container_width=True):
            if call_api_delete(f"/rag/chat/sessions/{active_session_id}"):
                _clear_current_chat(clear_leases=True)
                _rerun_if_available()

    return selected_leases


def _load_chat_sessions() -> list[dict[str, object]]:
    response = call_api_get("/rag/chat/sessions")
    if not isinstance(response, dict):
        return []
    sessions = response.get("sessions") or []
    return [session for session in sessions if isinstance(session, dict)]


def _load_chat_session(session_id: str, leases: list[S3LeaseOption]) -> None:
    response = call_api_get(f"/rag/chat/sessions/{session_id}")
    if not isinstance(response, dict):
        return

    st.session_state["rag_chat_session_id"] = response.get("session_id")
    st.session_state["rag_chat_history"] = [
        message
        for message in response.get("messages") or []
        if isinstance(message, dict)
    ]
    loaded_keys = [str(key) for key in response.get("lease_keys") or []]
    st.session_state["rag_chat_pending_lease_keys"] = loaded_keys


def _apply_pending_lease_selection(leases: list[S3LeaseOption]) -> None:
    if "rag_chat_pending_lease_keys" not in st.session_state:
        return
    lease_keys = st.session_state.pop("rag_chat_pending_lease_keys")
    st.session_state["rag_chat_lease_keys"] = _lease_options_for_keys(
        leases,
        [str(key) for key in lease_keys or []],
    )


def _lease_options_for_keys(
    leases: list[S3LeaseOption],
    lease_keys: list[str],
) -> list[S3LeaseOption]:
    key_set = set(lease_keys)
    return [lease for lease in leases if str(lease.get("key")) in key_set]


def _chat_session_label(session: dict[str, object]) -> str:
    title = str(session.get("title") or "Lease chat")
    updated_at = str(session.get("updated_at") or "")
    if updated_at:
        return f"{title} - {updated_at[:19]}"
    return title


def _clear_current_chat(clear_leases: bool = False) -> None:
    st.session_state["rag_chat_history"] = []
    st.session_state.pop("rag_chat_session_id", None)
    st.session_state.pop("rag_chat_pending_lease_keys", None)
    if clear_leases:
        st.session_state["rag_chat_pending_lease_keys"] = []


def _assistant_message_from_response(response: dict[str, object]) -> dict[str, object]:
    assistant_message: dict[str, object] = {
        "role": "assistant",
        "content": response.get("answer", ""),
        "citations": response.get("citations") or [],
    }
    if response.get("verification") is not None:
        assistant_message["verification"] = response.get("verification")
    if response.get("warnings"):
        assistant_message["warnings"] = response.get("warnings") or []
    if response.get("saved_at"):
        assistant_message["created_at"] = response.get("saved_at")
    return assistant_message


<<<<<<< HEAD
            if response is None:
                response_placeholder.empty()
                return

            assistant_message = {
                "role": "assistant",
                "content": response.get("answer", ""),
                "citations": response.get("citations") or [],
            }
            if response.get("verification") is not None:
                assistant_message["verification"] = response.get("verification")
            if response.get("warnings"):
                assistant_message["warnings"] = response.get("warnings") or []
            if response.get("eval") is not None:
                assistant_message["eval"] = response.get("eval")
            history.append(assistant_message)
            response_placeholder.markdown(str(assistant_message["content"]))
            _render_rag_chat_warnings(assistant_message)
            _render_ragas_scores(assistant_message.get("eval"))
            if show_sources:
                _render_rag_citations(assistant_message.get("citations") or [])
=======
def _rerun_if_available() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
>>>>>>> b217404d5777596d29b33f9e5cbae81b9b326d47


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


def _render_chat_history(history: list[object], show_sources: bool) -> None:
    for item in history:
        _render_chat_message(item, show_sources)


def _render_chat_message(item: object, show_sources: bool) -> None:
    if not isinstance(item, dict):
        return

    role = item.get("role")
    content = item.get("content")
    if role not in {"user", "assistant"} or not isinstance(content, str):
        return

    with st.chat_message(role):
        st.markdown(content)
        if role == "assistant":
            _render_rag_chat_warnings(item)
            _render_ragas_scores(item.get("eval"))
            if show_sources:
                _render_rag_citations(item.get("citations") or [])


def _render_rag_chat_warnings(item: dict[str, object]) -> None:
    warnings = item.get("warnings") or []
    if warnings:
        st.warning("\n".join(f"- {warning}" for warning in warnings))
        return

    verification = item.get("verification")
    if isinstance(verification, dict) and verification.get("overall_supported") is False:
        st.warning("The chat answer needs review against the indexed lease context.")


def _render_ragas_scores(eval_data: object) -> None:
    if not isinstance(eval_data, dict):
        return
    metrics = {
        "Faithfulness": eval_data.get("faithfulness"),
        "Relevancy": eval_data.get("answer_relevancy"),
        "Ctx Precision": eval_data.get("context_precision"),
        "Ctx Recall": eval_data.get("context_recall"),
    }
    available = {k: v for k, v in metrics.items() if v is not None}
    if not available:
        return
    with st.expander("RAGAS quality scores", expanded=False):
        cols = st.columns(len(available))
        for col, (label, score) in zip(cols, available.items()):
            col.metric(label, f"{float(score):.2f}")


def _render_rag_citations(citations: list[object]) -> None:
    if not citations:
        return

    st.markdown("#### Sources")
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        source_type = str(citation.get("source_type") or "chunk")
        st.caption(f"{citation.get('key', '')} ({source_type})")
        st.write(citation.get("snippet", ""))


def render_upload_index_tab() -> None:
    st.subheader("Upload & Index a Lease File")
    st.caption("Upload a .txt, .pdf, or .docx file. It will be validated (100+ words), stored in S3, chunked, and added to the vector index as its own collection.")

    uploaded = st.file_uploader(
        "Choose a lease file",
        type=["txt", "pdf", "docx"],
        key="upload_index_file",
    )

    if uploaded is not None:
        file_size_kb = round(uploaded.size / 1024, 1)
        st.info(f"Selected: **{uploaded.name}** ({file_size_kb} KB)")

    submitted = st.button(
        "Upload & Index",
        type="primary",
        use_container_width=True,
        disabled=uploaded is None,
    )

    if not submitted:
        return

    if uploaded is None:
        st.error("Select a file before uploading.")
        return

    with st.spinner(f"Uploading and indexing {uploaded.name}..."):
        response = call_api_files(
            "/upload-and-index",
            {"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")},
        )

    if response is None:
        return

    st.success("File uploaded and indexed successfully!")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Chunks created", response.get("chunk_count", 0))
    with col2:
        st.metric("Word count", response.get("word_count", 0))
    with col3:
        st.metric("Collection", "new")

    st.dataframe(
        [
            {
                "Field": "Filename",
                "Value": response.get("filename", ""),
            },
            {
                "Field": "S3 Key",
                "Value": response.get("s3_key", ""),
            },
            {
                "Field": "Collection Name",
                "Value": response.get("collection_name", ""),
            },
            {
                "Field": "Chunks",
                "Value": str(response.get("chunk_count", 0)),
            },
            {
                "Field": "Word Count",
                "Value": str(response.get("word_count", 0)),
            },
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Field": st.column_config.TextColumn("Field", width="small"),
            "Value": st.column_config.TextColumn("Value", width="large"),
        },
    )


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
