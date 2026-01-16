"""Evaluation runner UI."""

import time
import streamlit as st
from ui.api_client import APIClient


def render():
    """Render evaluation page."""
    st.header("Run Evaluation")
    
    try:
        datasets = APIClient.list_datasets()
    except Exception as e:
        st.error(f"Failed to load datasets: {e}")
        return
    
    if not datasets:
        st.warning("No test sets available. Please upload a test set first.")
        return
    
    dataset_names = [d['name'] for d in datasets]
    selected_dataset = st.selectbox("Select Test Set", dataset_names)
    
    num_questions = st.number_input(
        "Number of Questions to Evaluate",
        min_value=1,
        max_value=1000,
        value=None,
        placeholder="All questions",
        help="Leave empty to evaluate all questions, or specify number of first questions to use"
    )
    
    run_name = st.text_input(
        "Run Name (optional)",
        placeholder="e.g., baseline_run",
        help="Custom name for this evaluation run"
    )
    
    selected_metrics = _render_metric_selection()
    
    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("Start Evaluation", type="primary", disabled=st.session_state.eval_running)
    with col2:
        cancel_button = st.button("Cancel Evaluation", disabled=not st.session_state.eval_running)
    
    if cancel_button:
        st.session_state.cancel_eval = True
        st.session_state.eval_running = False
        st.warning("Evaluation cancelled")
    
    if start_button and not st.session_state.eval_running:
        _run_evaluation(selected_dataset, selected_metrics, run_name, num_questions)


def _render_metric_selection():
    """Render metric selection UI."""
    st.subheader("Select Metrics")
    
    try:
        available_metrics = APIClient.get_available_metrics()
    except Exception as e:
        st.error(f"Failed to load metrics: {e}")
        return []
    
    if not available_metrics:
        st.warning("Could not load available metrics")
        return []
    
    selected = st.multiselect(
        "Metrics to compute",
        available_metrics,
        default=available_metrics,
        help="Select which evaluation metrics to compute"
    )
    
    return selected


def _run_evaluation(dataset: str, metrics: list, run_name: str, num_questions: int = None):
    """Run evaluation with progress monitoring."""
    if not metrics:
        st.error("Please select at least one metric")
        return
    
    st.session_state.eval_running = True
    st.session_state.cancel_eval = False
    
    try:
        with st.spinner("Starting evaluation..."):
            run_id = APIClient.start_evaluation(dataset, metrics, {}, run_name, num_questions)
            
            if not run_id:
                st.error("Failed to start evaluation")
                st.session_state.eval_running = False
                return
            
            st.success(f"Evaluation started: {run_id}")
            _monitor_progress(run_id)
            
            st.session_state.eval_running = False
            st.success("Evaluation completed successfully!")
            st.info(f"View detailed results in 'Run Reports' page")
    except Exception as e:
        st.error(f"Evaluation failed: {e}")
        st.session_state.eval_running = False


def _monitor_progress(run_id: str):
    """Monitor evaluation progress."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.empty()
    
    while True:
        if st.session_state.cancel_eval:
            status_text.text("Evaluation cancelled by user")
            break
        
        try:
            run_details = APIClient.get_run_details(run_id)
            metadata = run_details.get('metadata', {})
            summary = run_details.get('summary', {})
            results = run_details.get('results', [])
            
            # Get progress from in-memory tracking (available immediately without disk reads)
            current_question = run_details.get('current_question', 0)
            total_questions = run_details.get('total_questions', metadata.get('num_questions', 0))
            current_question_text = run_details.get('current_question_text', '')
            
            if total_questions > 0:
                progress = current_question / total_questions
                progress_bar.progress(min(progress, 1.0))
                
                # Show current question being processed from in-memory tracking
                if current_question_text:
                    # Truncate question if too long
                    question_display = current_question_text[:80] + "..." if len(current_question_text) > 80 else current_question_text
                    status_text.text(f"Evaluating {current_question}/{total_questions}: {question_display}")
                else:
                    status_text.text(f"Evaluating {current_question}/{total_questions}")
            else:
                status_text.text("Starting evaluation...")
            
            if results:
                with results_container.container():
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Questions Processed", len(results))
                    with col2:
                        avg_score = summary.get('average_overall_score', 0)
                        st.metric("Average Score", f"{avg_score:.3f}")
            
            # Check if evaluation is completed
            status = run_details.get('status', 'running')
            if status == 'completed':
                progress_bar.progress(1.0)
                status_text.empty()
                break
            elif total_questions > 0 and current_question >= total_questions:
                progress_bar.progress(1.0)
                status_text.empty()
                break
            
            time.sleep(2)
        except Exception as e:
            st.error(f"Error monitoring progress: {e}")
            break
