import streamlit as st

from ui import configure_page, render_endpoint_status


configure_page("Health")

st.title("Health")
render_endpoint_status("/health")
