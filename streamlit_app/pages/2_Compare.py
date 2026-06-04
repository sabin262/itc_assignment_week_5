import streamlit as st

from api_client import post_endpoint
from ui import configure_page, render_endpoint_status


configure_page("Compare")

st.title("Compare")
render_endpoint_status("/compare")

with st.form("compare-form"):
    first_text = st.text_area("First text", height=160)
    second_text = st.text_area("Second text", height=160)
    submitted = st.form_submit_button("Compare")

if submitted:
    if first_text.strip() and second_text.strip():
        ok, result = post_endpoint(
            "/compare",
            {"first_text": first_text, "second_text": second_text},
        )
        if ok:
            st.subheader("Result")
            st.write(result["message"])
            st.metric("Length difference", result["length_difference"])
        else:
            st.error("The compare request failed.")
            st.json(result)
    else:
        st.warning("Enter both text values before submitting.")
