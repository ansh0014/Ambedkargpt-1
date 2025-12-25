# src/graph/summarizer.py

import pickle
from collections import defaultdict

from src.llm.llm_client import LLMClient


class CommunitySummarizer:
    """
    Summarizes each community using chunk text + LLM.
    """

    def __init__(self, graph_path: str, chunks_path: str, model="mistral:latest"):
        self.graph_path = graph_path
        self.chunks_path = chunks_path
        self.llm = LLMClient(model=model)

    def summarize(self):
        with open(self.graph_path, "rb") as f:
            graph = pickle.load(f)

        with open(self.chunks_path, "r", encoding="utf-8") as f:
            chunks = {c["chunk_id"]: c for c in pickle.loads(pickle.dumps(__import__("json").load(f)))}

        communities = defaultdict(list)

        for node, data in graph.nodes(data=True):
            cid = data.get("community_id")
            for chunk_id in data.get("chunks", []):
                if chunk_id in chunks:
                    communities[cid].append(chunks[chunk_id]["text"])

        summaries = {}

        for cid, texts in communities.items():
            context = "\n".join(texts[:5])  # limit context
            prompt = f"""
Summarize the following content into a concise thematic summary:

{context}

Summary:
"""
            summaries[cid] = self.llm.generate(prompt)

        return summaries
