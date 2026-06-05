import streamlit as st

from ui import configure_page, render_endpoint_status


configure_page("API Status")

st.title("FastAPI Status")
st.write("Use the sidebar to check each FastAPI endpoint from Streamlit.")

render_endpoint_status("/health")

st.subheader("Pages")
st.page_link("pages/1_Summarise.py", label="Summarise")
st.page_link("pages/2_Compare.py", label="Compare")
st.page_link("pages/3_Health.py", label="Health")
