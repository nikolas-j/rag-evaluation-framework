"""Evaluation reports viewer."""

import streamlit as st
from ui.api_client import APIClient


def render():
    """Render evaluation reports page."""
    st.header("Evaluation Run Reports")
    
    try:
        runs = APIClient.list_runs()
    except Exception as e:
        st.error(f"Failed to load runs: {e}")
        return
    
    if not runs:
        st.info("No evaluation runs available. Run an evaluation to see results here.")
        return
    
    run_options = [f"{r.get('run_name', r['run_id'][:8])} - {r.get('timestamp', 'N/A')}" for r in runs]
    run_ids = [r['run_id'] for r in runs]
    
    selected_idx = st.selectbox(
        "Select Run",
        range(len(run_options)),
        format_func=lambda i: run_options[i]
    )
    
    if selected_idx is None:
        return
    
    selected_run_id = run_ids[selected_idx]
    
    try:
        run_details = APIClient.get_run_details(selected_run_id)
        
        if run_details:
            _render_summary(run_details)
            _render_config(run_details)
            _render_metric_averages(run_details)
            _render_individual_results(run_details)
    except Exception as e:
        st.error(f"Failed to load run details: {e}")


def _render_summary(run_details):
    """Render run summary."""
    st.subheader("Run Summary")
    summary = run_details.get('summary', {})
    results = run_details.get('results', [])
    metadata = run_details.get('metadata', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Questions", summary.get('total_questions', 0))
    with col2:
        st.metric("Average Score", f"{summary.get('average_overall_score', 0):.3f}")
    with col3:
        st.metric("Questions Processed", len(results))
    with col4:
        # Try dataset from metadata, fallback to run_details directly
        dataset = run_details.get('dataset', metadata.get('dataset_name', 'N/A'))
        # Extract just the filename if it's a path
        if '/' in dataset:
            dataset = dataset.split('/')[-1].replace('.json', '')
        st.metric("Dataset", dataset)
    
    # Performance metrics row
    st.subheader("Average Performance Metrics")
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    
    with perf_col1:
        avg_retrieval = summary.get('average_retrieval_time_ms', 0)
        st.metric("Avg Retrieval Time", f"{avg_retrieval:.1f} ms")
    with perf_col2:
        avg_generation = summary.get('average_generation_time_ms', 0)
        st.metric("Avg Generation Time", f"{avg_generation:.1f} ms")
    with perf_col3:
        avg_total = summary.get('average_total_time_ms', 0)
        st.metric("Avg Total Time", f"{avg_total:.1f} ms")
    
    # Token usage metrics if available
    if summary.get('average_total_tokens') is not None:
        st.subheader("Average Token Usage")
        tok_col1, tok_col2, tok_col3 = st.columns(3)
        
        with tok_col1:
            avg_prompt = summary.get('average_prompt_tokens', 0)
            st.metric("Avg Prompt Tokens", f"{avg_prompt:.0f}")
        with tok_col2:
            avg_completion = summary.get('average_completion_tokens', 0)
            st.metric("Avg Completion Tokens", f"{avg_completion:.0f}")
        with tok_col3:
            avg_total_tok = summary.get('average_total_tokens', 0)
            st.metric("Avg Total Tokens", f"{avg_total_tok:.0f}")
        
        # Calculate price per 1k messages
        try:
            config = APIClient.load_config()
            input_price_per_m = config.get('input_token_price_per_million', 0.150)
            output_price_per_m = config.get('output_token_price_per_million', 0.600)
            
            # Price for 1 message
            price_per_message = (
                (avg_prompt * input_price_per_m / 1_000_000) +
                (avg_completion * output_price_per_m / 1_000_000)
            )
            # Price for 1000 messages
            price_per_1k = price_per_message * 1000
            
            st.subheader("Cost Estimate")
            cost_col1, cost_col2 = st.columns(2)
            with cost_col1:
                st.metric("Price per Message", f"${price_per_message:.4f}")
            with cost_col2:
                st.metric("Price per 1K Messages", f"${price_per_1k:.2f}")
        except Exception as e:
            st.warning(f"Could not calculate pricing: {e}")


def _render_config(run_details):
    """Render configuration as JSON."""
    config = run_details.get('config', {})
    
    if not config:
        st.info("No configuration snapshot available for this run")
        return
    
    # Highlight active prompts
    st.subheader("Active Prompts")
    prompt_col1, prompt_col2 = st.columns(2)
    
    with prompt_col1:
        rag_prompt = config.get('rag_system_prompt_title', 'Unknown')
        st.info(f"📝 **RAG System:** {rag_prompt}")
    
    with prompt_col2:
        st.markdown("**Evaluation Metrics:**")
        # Dynamically display all eval prompt fields in config
        for key, value in config.items():
            if key.startswith("eval_prompt_"):
                metric = key.replace("eval_prompt_", "")
                st.caption(f"• {metric.replace('_', ' ').title()}: {value}")
    
    with st.expander("📋 Full Configuration", expanded=False):
        st.json(config)


def _render_metric_averages(run_details):
    """Render metric averages."""
    st.subheader("Metric Averages")
    metric_avgs = run_details.get('summary', {}).get('metric_averages', {})
    
    if not metric_avgs:
        st.info("No metric averages available")
        return
    
    num_metrics = len(metric_avgs)
    cols = st.columns(min(num_metrics, 5))
    
    for i, (metric, score) in enumerate(metric_avgs.items()):
        col_idx = i % 5
        with cols[col_idx]:
            metric_display = metric.replace('_', ' ').title()
            st.metric(metric_display, f"{score:.3f}")


def _render_individual_results(run_details):
    """Render individual question results."""
    st.subheader("Individual Results")
    results = run_details.get('results', [])
    
    if not results:
        st.info("No individual results available for this run")
        return
    
    for i, result in enumerate(results, 1):
        question_text = result.get('question', '')
        overall_score = result.get('overall_score', 0)
        
        score_color = "🟢" if overall_score >= 0.7 else "🟡" if overall_score >= 0.5 else "🔴"
        
        with st.expander(f"{score_color} Q{i}: {question_text[:80]}... (Score: {overall_score:.3f})"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Question:**")
                st.write(result.get('question', 'N/A'))
                
                st.write("**Generated Answer:**")
                st.write(result.get('answer', 'N/A'))
                
                st.write("**Expected Answer:**")
                st.write(result.get('expected_answer', 'N/A'))
                
                if result.get('contexts'):
                    with st.expander("Retrieved Contexts"):
                        for ctx_idx, ctx in enumerate(result['contexts'], 1):
                            st.text(f"Context {ctx_idx}:")
                            st.text(ctx[:200] + "..." if len(ctx) > 200 else ctx)
            
            with col2:
                st.write("**Overall Score:**")
                st.metric("", f"{overall_score:.3f}")
                
                st.write("**Performance:**")
                retrieval_ms = result.get('retrieval_time_ms', 0)
                generation_ms = result.get('generation_time_ms', 0)
                total_ms = result.get('total_time_ms', 0)
                st.text(f"Retrieval: {retrieval_ms:.1f} ms")
                st.text(f"Generation: {generation_ms:.1f} ms")
                st.text(f"Total: {total_ms:.1f} ms")
                
                # Show token usage if available
                if result.get('total_tokens') is not None:
                    st.write("**Tokens:**")
                    st.text(f"Prompt: {result.get('prompt_tokens', 'N/A')}")
                    st.text(f"Completion: {result.get('completion_tokens', 'N/A')}")
                    st.text(f"Total: {result.get('total_tokens', 'N/A')}")
                
                st.write("**Metric Scores:**")
                metrics = result.get('metrics', {})
                for metric_name, metric_data in metrics.items():
                    score = metric_data.get('score', 0)
                    metric_display = metric_name.replace('_', ' ').title()
                    st.write(f"{metric_display}: {score:.3f}")
                
                if result.get('sources'):
                    st.write("**Sources:**")
                    for source in result['sources']:
                        source_file = source.get('source_path', 'unknown').split('/')[-1]
                        snippet = source.get('snippet', '')[:50]
                        rank = source.get('rank', '?')
                        st.text(f"{rank}. {source_file}")
                        st.caption(f"   {snippet}...")
