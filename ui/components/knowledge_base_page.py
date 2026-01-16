"""Knowledge base management UI."""

import time
import streamlit as st
from ui.api_client import APIClient
from ui.utils import render_file_tree


def render():
    """Render knowledge base management page."""
    st.header("Knowledge Base Management")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        _render_upload_section()
        _render_ingestion_section()
    
    with col2:
        _render_kb_files()


def _render_upload_section():
    """Render file upload section."""
    st.subheader("1. Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload .txt files",
        type=['txt'],
        accept_multiple_files=True,
        key="kb_file_uploader"
    )
    
    if uploaded_files:
        st.write(f"Selected {len(uploaded_files)} files:")
        for file in uploaded_files[:5]:
            st.text(f"  - {file.name}")
        if len(uploaded_files) > 5:
            st.text(f"  ... and {len(uploaded_files) - 5} more")
        
        if st.button("Upload Files"):
            try:
                with st.spinner("Uploading..."):
                    result = APIClient.upload_kb_files(uploaded_files)
                    st.success(f"Uploaded {result.get('count', 0)} files")
                    if result.get('errors'):
                        st.warning(f"Some files failed: {', '.join(result['errors'])}")
                    st.rerun()
            except Exception as e:
                st.error(f"Upload failed: {e}")


def _render_ingestion_section():
    """Render ingestion section."""
    st.subheader("2. Build Vector Store")
    
    st.info("💡 **Tip:** Use 'Clear & Rebuild' if you've updated files in knowledge_base to ensure old data is removed.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Start Ingestion", use_container_width=True):
            _run_ingestion()
    
    with col2:
        if st.button("🗑️ Clear & Rebuild", type="secondary", use_container_width=True, help="Delete existing vector store and rebuild from scratch"):
            _run_clear_and_rebuild()


def _run_ingestion():
    """Run ingestion process."""
    try:
        with st.spinner("Starting ingestion..."):
            APIClient.trigger_ingestion()
            st.success("Ingestion started")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            while True:
                status = APIClient.get_ingestion_status()
                status_text.text(f"Status: {status.get('status')} - {status.get('message', '')}")
                progress = status.get('progress', 0)
                progress_bar.progress(min(progress / 100, 1.0))
                
                if status.get('status') in ['completed', 'error', 'idle']:
                    break
                time.sleep(1)
            
            if status.get('status') == 'completed':
                st.success("✅ Ingestion completed successfully")
            elif status.get('status') == 'error':
                st.error(f"Ingestion failed: {status.get('message', 'Unknown error')}")
    except Exception as e:
        st.error(f"Ingestion failed: {e}")


def _run_clear_and_rebuild():
    """Clear vector store and rebuild from scratch."""
    try:
        # Step 1: Clear vector store
        with st.spinner("🗑️ Clearing vector store..."):
            result = APIClient.reset_vector_store()
            st.success(f"✅ {result.get('message', 'Vector store cleared')}")
        
        # Step 2: Run fresh ingestion
        with st.spinner("📥 Starting fresh ingestion..."):
            APIClient.trigger_ingestion()
            st.success("Ingestion started")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            while True:
                status = APIClient.get_ingestion_status()
                status_text.text(f"Status: {status.get('status')} - {status.get('message', '')}")
                progress = status.get('progress', 0)
                progress_bar.progress(min(progress / 100, 1.0))
                
                if status.get('status') in ['completed', 'error', 'idle']:
                    break
                time.sleep(1)
            
            if status.get('status') == 'completed':
                st.success("✅ Vector store rebuilt successfully!")
            elif status.get('status') == 'error':
                st.error(f"Ingestion failed: {status.get('message', 'Unknown error')}")
    except Exception as e:
        st.error(f"Failed to clear and rebuild: {e}")


def _render_kb_files():
    """Render current knowledge base files."""
    st.subheader("Current Knowledge Base Files")
    
    if st.button("Refresh File List"):
        st.rerun()
    
    try:
        tree = APIClient.get_kb_tree()
        files = APIClient.get_kb_files()
        
        if files:
            st.write(f"**Total files:** {len(files)}")
            
            with st.expander("File Tree View", expanded=True):
                render_file_tree(tree)
            
            with st.expander("File List"):
                for file_info in files:
                    st.text(f"📄 {file_info.get('path', '')} ({file_info.get('size', 0) / 1024:.1f} KB)")
        else:
            st.info("No files in knowledge base. Upload .txt files to get started.")
    except Exception as e:
        st.error(f"Failed to load files: {e}")
