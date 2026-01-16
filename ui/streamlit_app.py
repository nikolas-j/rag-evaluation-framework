"""Streamlit RAG Evaluation Framework."""

import streamlit as st
from ui.components import (
    config_page,
    knowledge_base_page,
    datasets_page,
    query_page,
    evaluation_page,
    reports_page,
    prompts_page
)
from ui.utils import init_session_state

st.set_page_config(
    page_title="RAG Evaluation Framework",
    page_icon="",
    layout="wide",
)

init_session_state()

st.title("RAG Evaluation Framework")
st.markdown("Interactive platform for testing and evaluating Retrieval-Augmented Generation systems")

page = st.sidebar.selectbox(
    "Navigation",
    [
        "Configuration",
        "Prompt Workspace",
        "Knowledge Base",
        "Test Sets",
        "RAG Query",
        "Evaluation",
        "Run Reports"
    ]
)

st.sidebar.markdown("---")

if page == "Configuration":
    config_page.render()
elif page == "Prompt Workspace":
    prompts_page.render()
elif page == "Knowledge Base":
    knowledge_base_page.render()
elif page == "Test Sets":
    datasets_page.render()
elif page == "RAG Query":
    query_page.render()
elif page == "Evaluation":
    evaluation_page.render()
elif page == "Run Reports":
    reports_page.render()

st.sidebar.markdown("---")
st.sidebar.caption("RAG Evaluation Framework v1.0")
st.sidebar.caption("Built with Streamlit, FastAPI, ChromaDB, OpenAI and LlamaIndex")

