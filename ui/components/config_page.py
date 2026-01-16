"""Configuration management UI."""

import json
import streamlit as st
from ui.api_client import APIClient


def render():
    """Render configuration page."""
    st.header("Configuration")
    
    if st.button("Load Current Config"):
        try:
            st.session_state.config = APIClient.load_config()
            st.success("Configuration loaded")
        except Exception as e:
            st.error(f"Failed to load config: {e}")
            return
    
    if not st.session_state.config:
        st.info("Click 'Load Current Config' to view and edit configuration")
        return
    
    config = st.session_state.config
    
    # Prompt Settings
    st.subheader("Prompt Settings")
    try:
        rag_prompts = APIClient.list_prompts("rag")
        eval_prompts = APIClient.list_prompts("eval")
        available_metrics = APIClient.get_available_metrics()
        
        prompt_col1, prompt_col2 = st.columns(2)
        with prompt_col1:
            st.markdown("**RAG System Prompt**")
            rag_prompt_titles = [p["title"] for p in rag_prompts]
            current_rag_prompt = config.get("rag_system_prompt_title", "Default v1.0")
            # Fallback to default if not found
            if current_rag_prompt not in rag_prompt_titles:
                current_rag_prompt = "Default v1.0"
            rag_prompt_idx = rag_prompt_titles.index(current_rag_prompt) if current_rag_prompt in rag_prompt_titles else 0
            rag_system_prompt_title = st.selectbox(
                "Select RAG Prompt",
                rag_prompt_titles,
                index=rag_prompt_idx,
                key="rag_system_prompt_title_input"
            )
        
        with prompt_col2:
            st.markdown("**Evaluation Prompts**")
            # Group eval prompts by metric
            eval_prompts_by_metric = {}
            for p in eval_prompts:
                metric = p.get("metric", "unknown")
                if metric not in eval_prompts_by_metric:
                    eval_prompts_by_metric[metric] = []
                eval_prompts_by_metric[metric].append(p["title"])
            
            # Store selections - dynamically based on available metrics
            eval_prompt_selections = {}
            for metric in available_metrics:
                if metric in eval_prompts_by_metric:
                    titles = eval_prompts_by_metric[metric]
                    current_title = config.get(f"eval_prompt_{metric}", "Default v1.0")
                    # Fallback to default if not found
                    if current_title not in titles:
                        current_title = "Default v1.0"
                    idx = titles.index(current_title) if current_title in titles else 0
                    selected = st.selectbox(
                        f"{metric.replace('_', ' ').title()}",
                        titles,
                        index=idx,
                        key=f"eval_prompt_{metric}_input"
                    )
                    eval_prompt_selections[metric] = selected
    except Exception as e:
        st.warning(f"Could not load prompts: {e}. Using defaults.")
        rag_system_prompt_title = config.get("rag_system_prompt_title", "Default v1.0")
        # Fallback to hardcoded list if API fails
        eval_prompt_selections = {
            "contextual_precision": config.get("eval_prompt_contextual_precision", "Default v1.0"),
            "contextual_relevance": config.get("eval_prompt_contextual_relevance", "Default v1.0"),
            "correctness": config.get("eval_prompt_correctness", "Default v1.0"),
            "faithfulness": config.get("eval_prompt_faithfulness", "Default v1.0"),
        }
    
    # RAG Settings
    st.subheader("RAG Settings")
    rag_col1, rag_col2 = st.columns(2)
    with rag_col1:
        retrieval_strategy = st.selectbox(
            "Retrieval Strategy",
            ["vector"],
            index=0,
            key="retrieval_strategy_input"
        )
        top_k = st.number_input("Top K", min_value=1, max_value=20, value=config.get("top_k", 5), key="top_k_input")
    with rag_col2:
        chunk_size = st.number_input("Chunk Size", min_value=128, max_value=2048, value=config.get("chunk_size", 512), step=64, key="chunk_size_input")
        chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=512, value=config.get("chunk_overlap", 50), step=10, key="chunk_overlap_input")
    
    # Eval Settings
    st.subheader("Eval Settings")
    eval_col1, eval_col2 = st.columns(2)
    with eval_col1:
        include_metric_reasons = st.checkbox(
            "Include Metric Reasons (slower but more informative)",
            value=config.get("include_metric_reasons", False),
            key="include_metric_reasons_input"
        )
        max_contexts_for_eval = st.number_input(
            "Max Contexts for Eval",
            min_value=1, max_value=10, value=config.get("max_contexts_for_eval", 3),
            key="max_contexts_for_eval_input",
            help="Maximum number of retrieved contexts to include in evaluation"
        )
    with eval_col2:
        # Convert dict to JSON string for display, handle empty dict
        default_weights = config.get("overall_score_weights", {})
        weights_str = json.dumps(default_weights) if default_weights else "{}"
        overall_score_weights = st.text_area(
            "Overall Score Weights (JSON)",
            value=weights_str,
            key="overall_score_weights_input",
            help="JSON dict mapping metric names to weights for overall score calculation. Example: {\"contextual_precision\": 0.5, \"contextual_relevance\": 0.5}",
            height=100
        )
    
    # Model Settings
    st.subheader("Model Settings")
    model_col1, model_col2 = st.columns(2)
    with model_col1:
        llm_model = st.text_input("LLM Model", value=config.get("llm_model", "gpt-4o-mini"), key="llm_model_input")
    with model_col2:
        embedding_model = st.text_input("Embedding Model", value=config.get("embedding_model", "text-embedding-3-small"), key="embedding_model_input")
    
    # LLM-as-Judge Settings
    st.subheader("LLM-as-Judge Settings")
    judge_col1, judge_col2 = st.columns(2)
    with judge_col1:
        judge_model = st.text_input(
            "Judge Model",
            value=config.get("judge_model", "gpt-4o-mini"),
            key="judge_model_input",
            help="OpenAI model to use for LLM-as-judge evaluation"
        )
        judge_num_samples = st.number_input(
            "Number of Judge Samples",
            min_value=1, max_value=10, value=config.get("judge_num_samples", 1),
            key="judge_num_samples_input",
            help="Number of times to evaluate each metric and average the scores"
        )
    with judge_col2:
        judge_temperature = st.slider(
            "Judge Temperature",
            min_value=0.0, max_value=2.0, value=config.get("judge_temperature", 0.0),
            step=0.1, key="judge_temperature_input",
            help="Temperature for judge LLM (0.0 = deterministic, higher = more random)"
        )
    
    # Token Pricing Settings
    st.subheader("Token Pricing (per Million)")
    pricing_col1, pricing_col2 = st.columns(2)
    with pricing_col1:
        input_token_price_per_million = st.number_input(
            "Input Token Price (USD/M)",
            min_value=0.0, value=config.get("input_token_price_per_million", 0.150),
            step=0.001, format="%.3f",
            key="input_token_price_per_million_input",
            help="Price per million input tokens in USD (e.g., 0.150 for gpt-4o-mini)"
        )
    with pricing_col2:
        output_token_price_per_million = st.number_input(
            "Output Token Price (USD/M)",
            min_value=0.0, value=config.get("output_token_price_per_million", 0.600),
            step=0.001, format="%.3f",
            key="output_token_price_per_million_input",
            help="Price per million output tokens in USD (e.g., 0.600 for gpt-4o-mini)"
        )
    
    if st.button("Save Configuration", type="primary"):
        try:
            # Parse overall_score_weights JSON
            try:
                weights_dict = json.loads(overall_score_weights) if overall_score_weights.strip() else {}
                if not isinstance(weights_dict, dict):
                    st.error("Overall Score Weights must be a JSON object/dict")
                    return
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON for Overall Score Weights: {e}")
                return
            
            overrides = {
                "llm_model": llm_model,
                "embedding_model": embedding_model,
                "retrieval_strategy": retrieval_strategy,
                "top_k": top_k,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "include_metric_reasons": include_metric_reasons,
                "max_contexts_for_eval": max_contexts_for_eval,
                "overall_score_weights": weights_dict,
                "judge_model": judge_model,
                "judge_num_samples": judge_num_samples,
                "judge_temperature": judge_temperature,
                "rag_system_prompt_title": rag_system_prompt_title,
                "input_token_price_per_million": input_token_price_per_million,
                "output_token_price_per_million": output_token_price_per_million,
            }
            
            # Add eval prompt selections dynamically
            for metric, title in eval_prompt_selections.items():
                overrides[f"eval_prompt_{metric}"] = title
            APIClient.save_config(overrides)
            st.success("Configuration saved successfully")
            st.session_state.config = overrides
        except Exception as e:
            st.error(f"Failed to save configuration: {e}")
