# src/llm/llm_client.py

import ollama


class LLMClient:
    """
    Wrapper around local Ollama LLM
    """

    def __init__(self, model: str = "llama3"):
        self.model = model

    def generate(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        return response["message"]["content"]
