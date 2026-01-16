"""Utilities for session state and formatting."""

import streamlit as st
from typing import Dict, Any


def init_session_state():
    """Initialize all required session state variables."""
    if 'config' not in st.session_state:
        st.session_state.config = {}
    if 'eval_running' not in st.session_state:
        st.session_state.eval_running = False
    if 'cancel_eval' not in st.session_state:
        st.session_state.cancel_eval = False


def render_file_tree(node: Dict[str, Any], level: int = 0) -> None:
    """Recursively render file tree."""
    if not node:
        return
    
    indent = "  " * level
    
    if node.get("type") == "directory":
        st.text(f"{indent}📁 {node.get('name', 'Unknown')}")
        for child in node.get("children", []):
            render_file_tree(child, level + 1)
    else:
        size_kb = node.get("size", 0) / 1024
        st.text(f"{indent}📄 {node.get('name', 'Unknown')} ({size_kb:.1f} KB)")
