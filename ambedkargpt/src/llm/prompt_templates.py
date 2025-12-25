

def build_prompt(query: str, context_chunks: list) -> str:
    """
    Build a grounded prompt using retrieved chunks.
    """

    context_text = "\n\n".join(
        [
            f"[Page {c['page']}]\n{c['text']}"
            for c in context_chunks
        ]
    )

    prompt = f"""
You are an expert assistant answering questions
ONLY using the provided context from Dr. B.R. Ambedkar's writings.

Context:
{context_text}

Question:
{query}

Rules:
- Answer only from the context
- Do NOT hallucinate
- Be concise and factual
- Cite page numbers in the answer

Answer:
"""
    return prompt
