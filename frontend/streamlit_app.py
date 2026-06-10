import streamlit as st

from frontend.layout import configure_page
from frontend.tabs import render_compare_tab, render_s3_leases_tab, render_summarise_tab


def main() -> None:
    configure_page()
    st.title("Smart Lease Summariser")

    summarise_tab, compare_tab, s3_tab = st.tabs(["Summarise", "Compare", "S3 Leases"])

    with summarise_tab:
        render_summarise_tab()

    with compare_tab:
        render_compare_tab()

    with s3_tab:
        render_s3_leases_tab()


if __name__ == "__main__":
    main()
