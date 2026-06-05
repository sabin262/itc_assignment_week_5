import streamlit as st

from api_client import post_files_endpoint
from ui import configure_page, render_endpoint_status


configure_page("Compare")

st.title("Compare")
render_endpoint_status("/compare")

with st.form("compare-form"):
    first_file = st.file_uploader("First file", type=["txt", "pdf", "docx"], key="first_file")
    second_file = st.file_uploader("Second file", type=["txt", "pdf", "docx"], key="second_file")
    submitted = st.form_submit_button("Compare")

if submitted:
    if first_file is not None and second_file is not None:
        ok, result = post_files_endpoint(
            "/compare",
            {"first_file": first_file, "second_file": second_file},
        )
        if ok:
            st.subheader("File contents")
            first_column, second_column = st.columns(2)

            with first_column:
                st.text_area(
                    result["first_filename"] or "First file",
                    value=result["first_text"],
                    height=320,
                    disabled=True,
                )

            with second_column:
                st.text_area(
                    result["second_filename"] or "Second file",
                    value=result["second_text"],
                    height=320,
                    disabled=True,
                )

            st.subheader("Result")
            st.write(result["message"])
            st.metric("Length difference", result["length_difference"])
        else:
            st.error("The compare request failed.")
            st.json(result)
    else:
        st.warning("Upload two .txt, .pdf, or .docx files before submitting.")
