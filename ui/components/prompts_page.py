"""Prompt workspace for managing versioned prompts."""

import streamlit as st
from ui.api_client import APIClient


def render():
    """Render prompt workspace page."""
    st.header("Prompt Workspace")
    st.markdown("Manage versioned prompts for RAG and evaluation metrics. Edit or create new prompts to customize system behavior.")
    
    # Tab selection
    tab1, tab2 = st.tabs(["RAG Prompts", "Evaluation Prompts"])
    
    with tab1:
        _render_rag_prompts()
    
    with tab2:
        _render_eval_prompts()


def _render_rag_prompts():
    """Render RAG prompts management."""
    st.subheader("RAG System Prompts")
    
    try:
        prompts = APIClient.list_prompts("rag")
    except Exception as e:
        st.error(f"Failed to load RAG prompts: {e}")
        return
    
    if not prompts:
        st.info("No RAG prompts available")
        return
    
    # Display existing prompts
    st.markdown("### Existing Prompts")
    for prompt in prompts:
        with st.expander(f"📝 {prompt['title']}" + (" (Default)" if prompt['title'] == "Default v1.0" else "")):
            st.text_area(
                "Content",
                value=prompt['content'],
                height=200,
                key=f"view_rag_{prompt['title']}",
                disabled=True
            )
            if prompt.get('description'):
                st.caption(f"**Description:** {prompt['description']}")
            st.caption(f"**Created:** {prompt.get('created_at', 'N/A')}")
            if prompt.get('modified_at'):
                st.caption(f"**Modified:** {prompt['modified_at']}")
            
            # Action buttons
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button(f"Use This", key=f"use_rag_{prompt['title']}", type="primary"):
                    try:
                        config = APIClient.load_config()
                        config["rag_system_prompt_title"] = prompt['title']
                        APIClient.save_config(config)
                        st.success(f"✅ Now using '{prompt['title']}' for RAG queries")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to set active prompt: {e}")
            
            with col2:
                if prompt['title'] != "Default v1.0":
                    if st.button(f"Delete", key=f"del_rag_{prompt['title']}"):
                        try:
                            APIClient.delete_prompt("rag", prompt['title'])
                            st.success(f"Deleted {prompt['title']}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to delete: {e}")
    
    # Create new prompt
    st.markdown("### Create New Prompt")
    with st.form("new_rag_prompt"):
        new_title = st.text_input(
            "Title",
            placeholder="e.g., Strict v2.0, Creative v1.5",
            help="Version identifier for this prompt"
        )
        new_description = st.text_input(
            "Description",
            placeholder="e.g., Stricter source citation requirements"
        )
        new_content = st.text_area(
            "Prompt Content",
            height=250,
            placeholder="Enter your system prompt here...\n\nTip: Look at 'Default v1.0' above as an example",
            help="No validation - you have full control. Make sure it works for your use case."
        )
        
        submitted = st.form_submit_button("Save Prompt", type="primary")
        
        if submitted:
            if not new_title or not new_content:
                st.error("Title and content are required")
            else:
                try:
                    APIClient.save_prompt("rag", new_title, new_content, new_description)
                    st.success(f"Saved prompt '{new_title}'")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to save prompt: {e}")


def _render_eval_prompts():
    """Render evaluation prompts management."""
    st.subheader("Evaluation Metric Prompts")
    
    try:
        prompts = APIClient.list_prompts("eval")
        metrics = APIClient.get_available_metrics()
    except Exception as e:
        st.error(f"Failed to load evaluation prompts: {e}")
        return
    
    if not prompts:
        st.info("No evaluation prompts available")
        return
    
    # Group by metric
    prompts_by_metric = {}
    for prompt in prompts:
        metric = prompt.get('metric', 'unknown')
        if metric not in prompts_by_metric:
            prompts_by_metric[metric] = []
        prompts_by_metric[metric].append(prompt)
    
    # Display existing prompts
    st.markdown("### Existing Prompts")
    for metric, metric_prompts in prompts_by_metric.items():
        st.markdown(f"**{metric.replace('_', ' ').title()}**")
        for prompt in metric_prompts:
            with st.expander(f"📝 {prompt['title']}" + (" (Default)" if prompt['title'] == "Default v1.0" else "")):
                st.text_area(
                    "Content",
                    value=prompt['content'],
                    height=250,
                    key=f"view_eval_{metric}_{prompt['title']}",
                    disabled=True
                )
                if prompt.get('description'):
                    st.caption(f"**Description:** {prompt['description']}")
                st.caption(f"**Metric:** {metric}")
                st.caption(f"**Created:** {prompt.get('created_at', 'N/A')}")
                if prompt.get('modified_at'):
                    st.caption(f"**Modified:** {prompt['modified_at']}")
                
                st.caption("**Note:** Prompts should include placeholders like `{question}`, `{expected_answer}`, `{contexts}` - see default for example")
                
                # Action buttons
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button(f"Use This", key=f"use_eval_{metric}_{prompt['title']}", type="primary"):
                        try:
                            config = APIClient.load_config()
                            # Update the specific metric prompt field
                            field_name = f"eval_prompt_{metric}"
                            config[field_name] = prompt['title']
                            APIClient.save_config(config)
                            st.success(f"✅ Now using '{prompt['title']}' for {metric.replace('_', ' ').title()}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to set active prompt: {e}")
                
                with col2:
                    if prompt['title'] != "Default v1.0":
                        if st.button(f"Delete", key=f"del_eval_{metric}_{prompt['title']}"):
                            try:
                                APIClient.delete_prompt("eval", prompt['title'], metric)
                                st.success(f"Deleted {prompt['title']}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to delete: {e}")
    
    # Create new prompt
    st.markdown("### Create New Evaluation Prompt")
    with st.form("new_eval_prompt"):
        metric_select = st.selectbox(
            "Metric",
            metrics,
            help="Which evaluation metric this prompt is for"
        )
        new_title = st.text_input(
            "Title",
            placeholder="e.g., Strict v2.0, Lenient v1.0",
            help="Version identifier for this prompt"
        )
        new_description = st.text_input(
            "Description",
            placeholder="e.g., Stricter relevance requirements"
        )
        new_content = st.text_area(
            "Prompt Content",
            height=300,
            placeholder="Enter your judge prompt here...\n\nTip: Look at the default prompt above for required placeholders",
            help="No validation - ensure you include necessary placeholders like {question}, {contexts}, etc."
        )
        
        submitted = st.form_submit_button("Save Prompt", type="primary")
        
        if submitted:
            if not metric_select or not new_title or not new_content:
                st.error("Metric, title, and content are required")
            else:
                try:
                    APIClient.save_eval_prompt(metric_select, new_title, new_content, new_description)
                    st.success(f"Saved prompt '{new_title}' for {metric_select}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to save prompt: {e}")
