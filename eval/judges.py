"""
Pure functional LLM-as-judge implementations for evaluation metrics.
Handles all judging logic with proper JSON validation and error handling.
"""
import json
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
from rag_app.config import get_settings
from rag_app.prompt_manager import get_prompt_manager
from .prompts import (
    CONTEXTUAL_PRECISION_PROMPT,
    CORRECTNESS_PROMPT,
    FAITHFULNESS_PROMPT,
    CONTEXTUAL_RELEVANCE_PROMPT
)

logger = logging.getLogger(__name__)


def _call_judge_api(
    prompt: str,
    model: str,
    temperature: float,
    max_retries: int = 3
) -> Optional[Dict[str, Any]]:
    """
    Call OpenAI API with JSON output enforcement and retry logic.
    
    Returns:
        Parsed JSON dict on success, None on failure
    """
    client = OpenAI()
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluation assistant. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Validate required fields
            if "score" not in result:
                logger.warning(f"Judge response missing 'score' field (attempt {attempt + 1})")
                continue
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error (attempt {attempt + 1}): {e}")
        except Exception as e:
            logger.error(f"API call error (attempt {attempt + 1}): {e}")
    
    return None


def judge_contextual_precision(
    question: str,
    contexts: List[str],
    expected_answer: str,
    answer: str = "",
    model: Optional[str] = None,
    num_samples: Optional[int] = None,
    temperature: Optional[float] = None,
    include_reasoning: Optional[bool] = None,
    prompt_template: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate contextual precision using OpenAI LLM-as-judge.
    
    All parameters are optional and will be loaded from .env defaults if not provided.
    
    Args:
        question: The question being answered
        contexts: List of retrieved context strings (ordered by ranking)
        expected_answer: The expected/golden answer
        model: OpenAI model to use (default: from settings.judge_model)
        num_samples: Number of samples to average (default: from settings.judge_num_samples)
        temperature: Sampling temperature (default: from settings.judge_temperature)
        include_reasoning: Include verdict reasoning (default: from settings.include_metric_reasons)
        prompt_template: Custom prompt template (default: from prompts.py)
    
    Returns:
        {
            "score": float (0.0-1.0, averaged across samples),
            "verdict": str (explanation, empty if include_reasoning=False),
            "samples": List[float] (individual scores),
            "error": str (present only if evaluation failed)
        }
    """
    
    # Load defaults from settings
    settings = get_settings()
    model = model or settings.judge_model
    num_samples = num_samples if num_samples is not None else settings.judge_num_samples
    temperature = temperature if temperature is not None else settings.judge_temperature
    include_reasoning = include_reasoning if include_reasoning is not None else settings.include_metric_reasons
    
    # Load prompt from library or fallback to default
    if prompt_template is None:
        prompt_manager = get_prompt_manager()
        prompt_title = settings.eval_prompt_contextual_precision
        prompt_template = prompt_manager.get_prompt("eval", prompt_title, metric="contextual_precision")
        if not prompt_template:
            logger.warning(f"Eval prompt '{prompt_title}' not found, using default")
            prompt_template = CONTEXTUAL_PRECISION_PROMPT
    
    # Format contexts
    contexts_formatted = "\n\n".join([
        f"Context {i+1}:\n{ctx}" 
        for i, ctx in enumerate(contexts)
    ])
    
    # Format prompt
    prompt = prompt_template.format(
        question=question,
        expected_answer=expected_answer,
        contexts=contexts_formatted
    )
    
    samples = []
    verdict = ""
    
    # Run multiple judge samples
    for sample_num in range(num_samples):
        result = _call_judge_api(prompt, model, temperature)
        
        if result is None:
            # Judge failed - return error result
            error_msg = f"Judge failed to return valid JSON after retries (sample {sample_num + 1})"
            logger.error(error_msg)
            return {
                "score": 0.0,
                "verdict": "",
                "samples": [],
                "error": error_msg
            }
        
        # Extract score and verdict
        try:
            score = float(result["score"])
            score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
            samples.append(score)
            
            if include_reasoning:
                verdict = result.get("verdict", "")
                
        except (ValueError, TypeError) as e:
            error_msg = f"Invalid score format in judge response: {e}"
            logger.error(error_msg)
            return {
                "score": 0.0,
                "verdict": "",
                "samples": [],
                "error": error_msg
            }
    
    # Calculate average score
    avg_score = sum(samples) / len(samples) if samples else 0.0
    
    return {
        "score": avg_score,
        "verdict": verdict if include_reasoning else "",
        "samples": samples
    }


def judge_correctness(
    question: str,
    contexts: List[str],
    expected_answer: str,
    answer: str,
    model: Optional[str] = None,
    num_samples: Optional[int] = None,
    temperature: Optional[float] = None,
    include_reasoning: Optional[bool] = None,
    prompt_template: Optional[str] = None
) -> Dict[str, Any]:
    """Evaluate correctness of generated answer using OpenAI LLM-as-judge.
    
    Args:
        question: The question being answered
        contexts: List of retrieved context strings (unused for this metric, for consistency)
        expected_answer: The expected/golden answer
        answer: Generated answer to evaluate
        model: OpenAI model to use (default: from settings.judge_model)
        num_samples: Number of samples to average (default: from settings.judge_num_samples)
        temperature: Sampling temperature (default: from settings.judge_temperature)
        include_reasoning: Include verdict reasoning (default: from settings.include_metric_reasons)
        prompt_template: Custom prompt template (default: from prompt library)
    
    Returns:
        Same structure as judge_contextual_precision
    """
    # Load defaults from settings
    settings = get_settings()
    model = model or settings.judge_model
    num_samples = num_samples if num_samples is not None else settings.judge_num_samples
    temperature = temperature if temperature is not None else settings.judge_temperature
    include_reasoning = include_reasoning if include_reasoning is not None else settings.include_metric_reasons
    
    # Load prompt from library or fallback to default
    if prompt_template is None:
        prompt_manager = get_prompt_manager()
        prompt_title = settings.eval_prompt_correctness
        prompt_template = prompt_manager.get_prompt("eval", prompt_title, metric="correctness")
        if not prompt_template:
            logger.warning(f"Eval prompt '{prompt_title}' not found, using default")
            prompt_template = CORRECTNESS_PROMPT
    
    # Format contexts for prompt template
    contexts_formatted = "\n\n".join([
        f"Context {i+1}:\n{ctx}" 
        for i, ctx in enumerate(contexts)
    ])
    
    # Format prompt
    prompt = prompt_template.format(
        question=question,
        expected_answer=expected_answer,
        answer=answer,
        contexts=contexts_formatted
    )
    
    samples = []
    verdict = ""
    
    # Run multiple judge samples
    for sample_num in range(num_samples):
        result = _call_judge_api(prompt, model, temperature)
        
        if result is None:
            error_msg = f"Judge failed to return valid JSON after retries (sample {sample_num + 1})"
            logger.error(error_msg)
            return {
                "score": 0.0,
                "verdict": "",
                "samples": [],
                "error": error_msg
            }
        
        try:
            score = float(result["score"])
            score = max(0.0, min(1.0, score))
            samples.append(score)
            
            if include_reasoning:
                verdict = result.get("verdict", "")
                
        except (ValueError, TypeError) as e:
            error_msg = f"Invalid score format in judge response: {e}"
            logger.error(error_msg)
            return {
                "score": 0.0,
                "verdict": "",
                "samples": [],
                "error": error_msg
            }
    
    avg_score = sum(samples) / len(samples) if samples else 0.0
    
    return {
        "score": avg_score,
        "verdict": verdict if include_reasoning else "",
        "samples": samples
    }


def judge_faithfulness(
    question: str,
    contexts: List[str],
    expected_answer: str,
    answer: str,
    model: Optional[str] = None,
    num_samples: Optional[int] = None,
    temperature: Optional[float] = None,
    include_reasoning: Optional[bool] = None,
    prompt_template: Optional[str] = None
) -> Dict[str, Any]:
    """Evaluate faithfulness of generated answer to contexts using OpenAI LLM-as-judge.
    
    Args:
        question: The question being answered
        contexts: List of retrieved context strings
        expected_answer: The expected/golden answer (unused for this metric, for consistency)
        answer: Generated answer to evaluate
        model: OpenAI model to use (default: from settings.judge_model)
        num_samples: Number of samples to average (default: from settings.judge_num_samples)
        temperature: Sampling temperature (default: from settings.judge_temperature)
        include_reasoning: Include verdict reasoning (default: from settings.include_metric_reasons)
        prompt_template: Custom prompt template (default: from prompt library)
    
    Returns:
        Same structure as judge_contextual_precision
    """
    # Load defaults from settings
    settings = get_settings()
    model = model or settings.judge_model
    num_samples = num_samples if num_samples is not None else settings.judge_num_samples
    temperature = temperature if temperature is not None else settings.judge_temperature
    include_reasoning = include_reasoning if include_reasoning is not None else settings.include_metric_reasons
    
    # Load prompt from library or fallback to default
    if prompt_template is None:
        prompt_manager = get_prompt_manager()
        prompt_title = settings.eval_prompt_faithfulness
        prompt_template = prompt_manager.get_prompt("eval", prompt_title, metric="faithfulness")
        if not prompt_template:
            logger.warning(f"Eval prompt '{prompt_title}' not found, using default")
            prompt_template = FAITHFULNESS_PROMPT
    
    # Format contexts
    contexts_formatted = "\n\n".join([
        f"Context {i+1}:\n{ctx}" 
        for i, ctx in enumerate(contexts)
    ])
    
    # Format prompt
    prompt = prompt_template.format(
        question=question,
        contexts=contexts_formatted,
        answer=answer
    )
    
    samples = []
    verdict = ""
    
    # Run multiple judge samples
    for sample_num in range(num_samples):
        result = _call_judge_api(prompt, model, temperature)
        
        if result is None:
            error_msg = f"Judge failed to return valid JSON after retries (sample {sample_num + 1})"
            logger.error(error_msg)
            return {
                "score": 0.0,
                "verdict": "",
                "samples": [],
                "error": error_msg
            }
        
        try:
            score = float(result["score"])
            score = max(0.0, min(1.0, score))
            samples.append(score)
            
            if include_reasoning:
                verdict = result.get("verdict", "")
                
        except (ValueError, TypeError) as e:
            error_msg = f"Invalid score format in judge response: {e}"
            logger.error(error_msg)
            return {
                "score": 0.0,
                "verdict": "",
                "samples": [],
                "error": error_msg
            }
    
    avg_score = sum(samples) / len(samples) if samples else 0.0
    
    return {
        "score": avg_score,
        "verdict": verdict if include_reasoning else "",
        "samples": samples
    }


def judge_contextual_relevance(
    question: str,
    contexts: List[str],
    expected_answer: str,
    answer: str = "",  # For consistency with other judges
    model: Optional[str] = None,
    num_samples: Optional[int] = None,
    temperature: Optional[float] = None,
    include_reasoning: Optional[bool] = None,
    prompt_template: Optional[str] = None
) -> Dict[str, Any]:
    """Evaluate contextual relevance using OpenAI LLM-as-judge.
    
    Args:
        question: The question being answered
        contexts: List of retrieved context strings
        expected_answer: The expected/golden answer
        answer: Generated answer (unused for this metric, for consistency)
        model: OpenAI model to use (default: from settings.judge_model)
        num_samples: Number of samples to average (default: from settings.judge_num_samples)
        temperature: Sampling temperature (default: from settings.judge_temperature)
        include_reasoning: Include verdict reasoning (default: from settings.include_metric_reasons)
        prompt_template: Custom prompt template (default: from prompt library)
    
    Returns:
        Same structure as judge_contextual_precision
    """
    # Load defaults from settings
    settings = get_settings()
    model = model or settings.judge_model
    num_samples = num_samples if num_samples is not None else settings.judge_num_samples
    temperature = temperature if temperature is not None else settings.judge_temperature
    include_reasoning = include_reasoning if include_reasoning is not None else settings.include_metric_reasons
    
    # Load prompt from library or fallback to default
    if prompt_template is None:
        prompt_manager = get_prompt_manager()
        prompt_title = settings.eval_prompt_contextual_relevance
        prompt_template = prompt_manager.get_prompt("eval", prompt_title, metric="contextual_relevance")
        if not prompt_template:
            logger.warning(f"Eval prompt '{prompt_title}' not found, using default")
            prompt_template = CONTEXTUAL_RELEVANCE_PROMPT
    
    # Format contexts
    contexts_formatted = "\n\n".join([
        f"Context {i+1}:\n{ctx}" 
        for i, ctx in enumerate(contexts)
    ])
    
    # Format prompt
    prompt = prompt_template.format(
        question=question,
        expected_answer=expected_answer,
        contexts=contexts_formatted
    )
    
    samples = []
    verdict = ""
    
    # Run multiple judge samples
    for sample_num in range(num_samples):
        result = _call_judge_api(prompt, model, temperature)
        
        if result is None:
            error_msg = f"Judge failed to return valid JSON after retries (sample {sample_num + 1})"
            logger.error(error_msg)
            return {
                "score": 0.0,
                "verdict": "",
                "samples": [],
                "error": error_msg
            }
        
        try:
            score = float(result["score"])
            score = max(0.0, min(1.0, score))
            samples.append(score)
            
            if include_reasoning:
                verdict = result.get("verdict", "")
                
        except (ValueError, TypeError) as e:
            error_msg = f"Invalid score format in judge response: {e}"
            logger.error(error_msg)
            return {
                "score": 0.0,
                "verdict": "",
                "samples": [],
                "error": error_msg
            }
    
    avg_score = sum(samples) / len(samples) if samples else 0.0
    
    return {
        "score": avg_score,
        "verdict": verdict if include_reasoning else "",
        "samples": samples
    }
