import streamlit as st

from api_client import post_endpoint
from ui import configure_page, render_endpoint_status


configure_page("Summarize")

st.title("Summarize")
render_endpoint_status("/summarize")

with st.form("summarize-form"):
    text = st.text_area("Text", height=220, placeholder="Paste text to summarize.")
    submitted = st.form_submit_button("Summarize")

if submitted:
    if text.strip():
        ok, result = post_endpoint("/summarize", {"text": text})
        if ok:
            st.subheader("Summary")
            st.write(result["summary"])
            st.caption(f"Original length: {result['original_length']} characters")
        else:
            st.error("The summarize request failed.")
            st.json(result)
    else:
        st.warning("Enter text before submitting.")
