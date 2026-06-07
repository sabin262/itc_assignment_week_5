import streamlit as st


PAGE_STYLES = """
<style>
.block-container {
    padding-top: 1.6rem;
    padding-bottom: 2rem;
    max-width: 1180px;
}
div[data-testid="stMetricValue"] {
    font-size: 1.05rem;
}
.stTextArea textarea {
    min-height: 260px;
}
</style>
"""


def configure_page() -> None:
    st.set_page_config(
        page_title="Smart Lease Summariser",
        layout="wide",
    )
    st.markdown(PAGE_STYLES, unsafe_allow_html=True)
