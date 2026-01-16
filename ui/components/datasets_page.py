"""Test dataset management UI."""

import streamlit as st
from ui.api_client import APIClient


def render():
    """Render datasets management page."""
    st.header("Test Set Management")
    
    _render_upload_section()
    _render_datasets_list()


def _render_upload_section():
    """Render dataset upload section."""
    st.subheader("Upload Test Set")
    uploaded_file = st.file_uploader(
        "Upload .json test set",
        type=['json'],
        key="dataset_file_uploader"
    )
    
    if uploaded_file and st.button("Upload Test Set"):
        try:
            with st.spinner("Uploading and validating..."):
                result = APIClient.upload_test_set(uploaded_file)
                st.success(f"Test set '{result.get('name')}' uploaded successfully ({result.get('num_questions')} questions)")
                st.rerun()
        except Exception as e:
            st.error(f"Upload failed: {e}")


def _render_datasets_list():
    """Render list of available datasets."""
    st.subheader("Available Test Sets")
    
    try:
        datasets = APIClient.list_datasets()
        
        if not datasets:
            st.info("No test sets available. Upload a .json test set to get started.")
            return
        
        for dataset in datasets:
            with st.expander(f"📊 {dataset['name']} ({dataset['num_questions']} questions)"):
                st.write(f"**Path:** {dataset['path']}")
                if dataset.get('description'):
                    st.write(f"**Description:** {dataset['description']}")
                st.write(f"**Questions:** {dataset['num_questions']}")
    except Exception as e:
        st.error(f"Failed to load datasets: {e}")
