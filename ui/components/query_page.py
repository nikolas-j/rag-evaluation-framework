"""RAG query interface."""

import streamlit as st
from ui.api_client import APIClient


def render():
    """Render RAG query page."""
    st.header("RAG Query Interface")
    
    question = st.text_area(
        "Enter your question:",
        height=100,
        placeholder="e.g., What is the expense policy for hotels?"
    )
    
    if st.button("Submit Query", type="primary") and question:
        try:
            with st.spinner("Processing query..."):
                result = APIClient.query_rag(question, {})
                
                if result:
                    _render_answer(result)
                    _render_sources(result)
                    _render_metadata(result)
        except Exception as e:
            st.error(f"Query failed: {e}")


def _render_answer(result):
    """Render the generated answer."""
    st.subheader("Answer")
    answer = result.get('answer', 'No answer generated')
    st.markdown(f"**{answer}**")


def _render_sources(result):
    """Render retrieved sources."""
    st.subheader("Retrieved Sources")
    sources = result.get('sources', [])
    
    if not sources:
        st.info("No sources retrieved")
        return
    
    for i, source in enumerate(sources, 1):
        with st.expander(f"Source {i}: {source.get('filename', 'Unknown')} (Rank: {source.get('rank', 'N/A')})"):
            st.write(f"**Path:** {source.get('source_path', 'N/A')}")
            st.write(f"**Category:** {source.get('category', 'N/A')}")
            
            if source.get('snippet'):
                st.write("**Snippet:**")
                st.text(source['snippet'])


def _render_metadata(result):
    """Render query metadata."""
    st.subheader("Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        retrieval_time = result.get('retrieval_time_ms', 0)
        st.metric("Retrieval Time", f"{retrieval_time:.0f} ms")
    
    with col2:
        generation_time = result.get('generation_time_ms', 0)
        st.metric("Generation Time", f"{generation_time:.0f} ms")
    
    with col3:
        total_time = retrieval_time + generation_time
        st.metric("Total Time", f"{total_time:.0f} ms")
    
    # Token usage metrics
    prompt_tokens = result.get('prompt_tokens')
    completion_tokens = result.get('completion_tokens')
    total_tokens = result.get('total_tokens')
    
    if any([prompt_tokens, completion_tokens, total_tokens]):
        st.subheader("Token Usage")
        tcol1, tcol2, tcol3 = st.columns(3)
        
        with tcol1:
            st.metric("Prompt Tokens", prompt_tokens if prompt_tokens is not None else "N/A")
        with tcol2:
            st.metric("Completion Tokens", completion_tokens if completion_tokens is not None else "N/A")
        with tcol3:
            st.metric("Total Tokens", total_tokens if total_tokens is not None else "N/A")
    
    # Display active prompt
    config = result.get('config_snapshot', {})
    if config:
        st.subheader("Active Prompt")
        rag_prompt = config.get('rag_system_prompt_title', 'Unknown')
        st.info(f"📝 RAG System Prompt: **{rag_prompt}**")
        
        with st.expander("Configuration Used"):
            st.json(config)
