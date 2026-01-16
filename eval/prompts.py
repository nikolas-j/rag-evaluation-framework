"""Evaluation metric prompts for LLM-as-judge."""

CONTEXTUAL_PRECISION_PROMPT = """You are an expert evaluator assessing the precision of retrieved contexts for a question-answering system.

Your task is to evaluate whether the retrieved contexts are relevant and ranked appropriately for answering the given question with the expected answer in mind.

Question: {question}

Expected Answer: {expected_answer}

Retrieved Contexts (in order of ranking):
{contexts}

Evaluation Criteria:
- Are the contexts relevant to answering the question?
- Are the most relevant contexts ranked higher?
- Do irrelevant contexts appear before relevant ones (poor precision)?

Rate the contextual precision on a scale from 0.0 to 1.0:
- 1.0: All contexts are relevant and perfectly ranked
- 0.5-0.9: Most contexts relevant but some ranking issues
- 0.0-0.4: Many irrelevant contexts or poor ranking

Respond with a JSON object containing:
{{"score": <float between 0.0 and 1.0>, "verdict": "<brief explanation>"}}

Your response must be valid JSON only, no additional text.
"""

CORRECTNESS_PROMPT = """You are an expert evaluator assessing the correctness of generated answers.

Your task is to evaluate whether the generated answer is semantically equivalent to the expected answer.

Question: {question}

Expected Answer: {expected_answer}

Generated Answer: {answer}

Evaluation Criteria:
- Does the generated answer convey the same core information as the expected answer?
- Are key facts, dates, names, and details accurate?
- Minor phrasing differences are acceptable if meaning is preserved

Rate the correctness on a scale from 0.0 to 1.0:
- 1.0: Fully correct, all key information matches
- 0.5-0.9: Mostly correct but missing some details or minor inaccuracies
- 0.0-0.4: Incorrect, contradictory, or missing critical information

Respond with a JSON object containing:
{{"score": <float between 0.0 and 1.0>, "verdict": "<brief explanation>"}}

Your response must be valid JSON only, no additional text.
"""

FAITHFULNESS_PROMPT = """You are an expert evaluator assessing the faithfulness of generated answers to source contexts.

Your task is to evaluate whether the generated answer is fully grounded in the retrieved contexts without hallucinations.

Question: {question}

Retrieved Contexts:
{contexts}

Generated Answer: {answer}

Evaluation Criteria:
- Is every claim in the answer supported by the contexts?
- Are there any hallucinated facts, figures, or statements not in the contexts?
- Does the answer extrapolate beyond what's explicitly stated?

Rate the faithfulness on a scale from 0.0 to 1.0:
- 1.0: Completely faithful, all claims supported by contexts
- 0.5-0.9: Mostly faithful but minor unsupported details
- 0.0-0.4: Contains hallucinations or significant unsupported claims

Respond with a JSON object containing:
{{"score": <float between 0.0 and 1.0>, "verdict": "<brief explanation>"}}

Your response must be valid JSON only, no additional text.
"""

CONTEXTUAL_RELEVANCE_PROMPT = """You are an expert evaluator assessing the relevance of retrieved contexts.

Your task is to evaluate whether the retrieved contexts contain the information needed to answer the question.

Question: {question}

Expected Answer: {expected_answer}

Retrieved Contexts:
{contexts}

Evaluation Criteria:
- Do the contexts contain information that helps answer the question?
- Is the necessary information present even if not perfectly highlighted?
- Can the expected answer be derived from the contexts?

Rate the contextual relevance on a scale from 0.0 to 1.0:
- 1.0: Contexts contain all necessary information to answer fully
- 0.5-0.9: Contexts contain some relevant info but missing key details
- 0.0-0.4: Contexts lack the information needed to answer

Respond with a JSON object containing:
{{"score": <float between 0.0 and 1.0>, "verdict": "<brief explanation>"}}

Your response must be valid JSON only, no additional text.
"""
