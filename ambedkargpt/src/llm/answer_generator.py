# src/llm/answer_generator.py

from src.llm.llm_client import LLMClient
from src.llm.prompt_templates import build_prompt


class AnswerGenerator:
    """
    Generates final answers using retrieved context + LLM
    """

    def __init__(self, model: str = "llama3"):
        self.llm = LLMClient(model=model)

    def generate_answer(
        self,
        query: str,
        local_results: list,
        global_results: list,
        max_chunks: int = 5,
    ) -> str:
        # Combine local + global results
        combined = local_results + global_results

        # Remove duplicates
        seen = set()
        unique_chunks = []
        for c in combined:
            if c["chunk_id"] not in seen:
                unique_chunks.append(c)
                seen.add(c["chunk_id"])

        # Take top chunks
        selected_chunks = unique_chunks[:max_chunks]

        prompt = build_prompt(query, selected_chunks)
        return self.llm.generate(prompt)
