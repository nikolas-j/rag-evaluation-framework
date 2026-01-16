"""RAG prompt templates and configuration."""

PROMPT_VERSION = "1.0"

SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided context.

Your responsibilities:
1. Answer questions using ONLY the information in the provided context
2. If the context does not contain sufficient information to answer the question, explicitly state: "I don't have enough information in the provided context to answer this question."
3. Cite your sources by referring to the source document names
4. Be concise and accurate
5. Do not make up or infer information beyond what is explicitly stated in the context

When citing sources, use the format: [Source: filename.txt]
"""

USER_PROMPT_TEMPLATE = """Context information is below:
---------------------
{context_str}
---------------------

Based on the context above, please answer the following question. Remember to cite your sources.

Question: {query_str}

Answer:"""


def format_context(contexts: list) -> str:
    """Format retrieved contexts into a single string."""
    formatted_parts = []
    for i, node in enumerate(contexts, 1):
        source = node.metadata.get("source_path", "unknown")
        formatted_parts.append(f"[Document {i} - {source}]\n{node.text}\n")
    
    return "\n".join(formatted_parts)

