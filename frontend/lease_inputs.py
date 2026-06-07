from typing import Any

import streamlit as st

from frontend.api_client import FilePart


LeaseInputValue = dict[str, Any]


def lease_input(key_prefix: str, label: str, upload_label: str) -> LeaseInputValue:
    input_source = st.radio(
        f"{label} input source",
        ["Upload file", "Paste text"],
        horizontal=True,
        key=f"{key_prefix}_source",
    )

    if input_source == "Upload file":
        uploaded_file = st.file_uploader(
            upload_label,
            type=["txt", "pdf", "docx"],
            key=f"{key_prefix}_upload",
        )
        return {"source": "file", "file": uploaded_file, "text": ""}

    text = st.text_area(label, key=f"{key_prefix}_text", height=280)
    return {"source": "text", "file": None, "text": text}


def has_lease_input(value: LeaseInputValue) -> bool:
    if value["source"] == "file":
        return value["file"] is not None
    return bool(value["text"].strip())


def uploaded_file_part(uploaded_file: Any) -> FilePart:
    media_type = {
        "txt": "text/plain",
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }.get(uploaded_file.name.rsplit(".", 1)[-1].lower(), "application/octet-stream")
    return (uploaded_file.name, uploaded_file.getvalue(), media_type)


def lease_input_file_part(value: LeaseInputValue, fallback_filename: str) -> FilePart:
    if value["source"] == "file":
        return uploaded_file_part(value["file"])
    return (fallback_filename, value["text"].encode("utf-8"), "text/plain")
