import streamlit as st

from api_client import post_files_endpoint
from ui import configure_page, render_endpoint_status


configure_page("Summarise")

st.title("Summarise")
render_endpoint_status("/summarise")

with st.form("summarise-form"):
    uploaded_file = st.file_uploader("Upload file", type=["txt", "pdf", "docx"])
    submitted = st.form_submit_button("Summarise")

if submitted:
    if uploaded_file is not None:
        ok, result = post_files_endpoint("/summarise", {"file": uploaded_file})
        if ok:
            st.subheader("File contents")
            st.text_area(
                "Extracted text",
                value=result["text"],
                height=320,
                disabled=True,
            )
            st.subheader("Summary")
            st.write(result["summary"])
            st.caption(f"Original length: {result['original_length']} characters")
        else:
            st.error("The summarise request failed.")
            st.json(result)
    else:
        st.warning("Upload a .txt, .pdf, or .docx file before submitting.")
