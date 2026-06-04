from typing import Any

import streamlit as st

from api_client import get_endpoint


def configure_page(title: str) -> None:
    st.set_page_config(page_title=title, page_icon=".", layout="centered")


def render_endpoint_status(path: str) -> tuple[bool, dict[str, Any]]:
    ok, data = get_endpoint(path)

    if ok:
        st.success(f"{path} is available")
    else:
        st.error(f"{path} is unavailable")

    with st.expander("Raw endpoint response", expanded=False):
        st.json(data)

    return ok, data
