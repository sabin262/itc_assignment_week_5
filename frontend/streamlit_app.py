import streamlit as st

from frontend.layout import configure_page
from frontend.tabs import (
    render_compare_tab,
    render_s3_chat_tab,
    render_s3_compare_tab,
    render_s3_index_tab,
    render_s3_search_tab,
    render_s3_summarise_tab,
    render_summarise_tab,
)


def main() -> None:
    configure_page()
    st.title("Smart Lease Summariser")

    page = st.sidebar.radio("Workspace", ["Local Leases", "S3 Leases"])

    if page == "Local Leases":
        summarise_tab, compare_tab = st.tabs(["Summarise", "Compare"])

        with summarise_tab:
            render_summarise_tab()

        with compare_tab:
            render_compare_tab()
    else:
        (
            summarise_tab,
            compare_tab,
            index_tab,
            search_tab,
            chat_tab,
        ) = st.tabs(["Summarise", "Compare", "Index", "Search", "Chat"])

        with summarise_tab:
            render_s3_summarise_tab()

        with compare_tab:
            render_s3_compare_tab()

        with index_tab:
            render_s3_index_tab()

        with search_tab:
            render_s3_search_tab()

        with chat_tab:
            render_s3_chat_tab()


if __name__ == "__main__":
    main()
