from typing import Any

import httpx
import streamlit as st

from frontend.error_utils import extract_error_message
from frontend.settings import API_BASE_URL, REQUEST_TIMEOUT_SECONDS


FilePart = tuple[str, bytes, str]


def call_api_get(path: str) -> list[dict[str, Any]] | dict[str, Any] | None:
    url = f"{API_BASE_URL}{path}"
    try:
        response = httpx.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    except httpx.RequestError as exc:
        st.error(f"Could not reach API at {API_BASE_URL}: {exc}")
        return None

    if response.status_code >= 400:
        render_api_error(response)
        return None

    return response.json()


def call_api(path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    url = f"{API_BASE_URL}{path}"
    try:
        response = httpx.post(url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    except httpx.RequestError as exc:
        st.error(f"Could not reach API at {API_BASE_URL}: {exc}")
        return None

    if response.status_code >= 400:
        render_api_error(response)
        return None

    return response.json()


def call_api_delete(path: str) -> bool:
    url = f"{API_BASE_URL}{path}"
    try:
        response = httpx.delete(url, timeout=REQUEST_TIMEOUT_SECONDS)
    except httpx.RequestError as exc:
        st.error(f"Could not reach API at {API_BASE_URL}: {exc}")
        return False

    if response.status_code >= 400:
        render_api_error(response)
        return False

    return True


def call_api_files(path: str, files: dict[str, FilePart]) -> dict[str, Any] | None:
    url = f"{API_BASE_URL}{path}"
    try:
        response = httpx.post(url, files=files, timeout=REQUEST_TIMEOUT_SECONDS)
    except httpx.RequestError as exc:
        st.error(f"Could not reach API at {API_BASE_URL}: {exc}")
        return None

    if response.status_code >= 400:
        render_api_error(response)
        return None

    return response.json()


def render_api_error(response: httpx.Response) -> None:
    message = extract_error_message(response)
    st.error(f"API returned HTTP {response.status_code}: {message}")
