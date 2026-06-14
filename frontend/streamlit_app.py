import streamlit as st

from frontend.layout import configure_page
from frontend.tabs import (
    render_compare_tab,
    render_s3_chat_tab,
    render_s3_compare_tab,
    render_s3_index_tab,
    render_s3_summarise_tab,
    render_summarise_tab,
    render_upload_index_tab,
)

WORKSPACE_PAGES = ["Local Leases", "S3 Leases"]


def main() -> None:
    configure_page()
    st.title("Smart Lease Summariser")

    page = _workspace_page_selector()

    if page == "Local Leases":
        summarise_tab, compare_tab = st.tabs(["Summarise", "Compare"])

        with summarise_tab:
            render_summarise_tab()

        with compare_tab:
            render_compare_tab()
    else:
        (
            upload_tab,
            index_tab,
            summarise_tab,
            compare_tab,
            chat_tab,
        ) = st.tabs(["Index & Upload", "Index", "Summarise", "Compare", "Chat"])

        with upload_tab:
            render_upload_index_tab()

        with index_tab:
            render_s3_index_tab()

        with summarise_tab:
            render_s3_summarise_tab()

        with compare_tab:
            render_s3_compare_tab()

        with chat_tab:
            render_s3_chat_tab()


def _workspace_page_selector() -> str:
    current_page = st.session_state.get("workspace_page", WORKSPACE_PAGES[0])
    if current_page not in WORKSPACE_PAGES:
        current_page = WORKSPACE_PAGES[0]

    st.sidebar.markdown("#### Workspace")
    for page in WORKSPACE_PAGES:
        if st.sidebar.button(
            page,
            key=f"workspace_page_{page.lower().replace(' ', '_')}",
            use_container_width=True,
            type="primary" if page == current_page else "secondary",
        ):
            current_page = page
            st.session_state["workspace_page"] = page

    st.session_state["workspace_page"] = current_page
    return str(current_page)


if __name__ == "__main__":
    main()
