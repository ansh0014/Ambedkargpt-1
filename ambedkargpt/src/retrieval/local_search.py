# src/retrieval/local_search.py

import json
import pickle
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class LocalGraphRAGSearch:
    """
    Implements Local Graph RAG Search (Equation 4 from SemRAG paper)
    """

    def __init__(
        self,
        graph_path: str,
        chunks_path: str,
        entity_threshold: float = 0.3,
        chunk_threshold: float = 0.25,
        top_k: int = 5,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.entity_threshold = entity_threshold
        self.chunk_threshold = chunk_threshold
        self.top_k = top_k

        self.embedder = SentenceTransformer(embedding_model)

        with open(graph_path, "rb") as f:
            self.graph = pickle.load(f)

        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        # Precompute entity embeddings
        self.entity_embeddings = {}
        for node in self.graph.nodes:
            self.entity_embeddings[node] = self.embedder.encode(node)

    def search(self, query: str) -> List[Dict]:
        query_embedding = self.embedder.encode(query)

        # 1. Find relevant entities
        relevant_entities = []
        for entity, emb in self.entity_embeddings.items():
            sim = cosine_similarity(
                [query_embedding], [emb]
            )[0][0]

            if sim >= self.entity_threshold:
                relevant_entities.append(entity)

        # 2. Collect candidate chunks
        candidate_chunks = []

        for entity in relevant_entities:
            chunk_ids = self.graph.nodes[entity].get("chunks", [])

            for cid in chunk_ids:
                chunk = self.chunks[cid]
                chunk_emb = np.array(chunk["embedding"])

                sim = cosine_similarity(
                    [query_embedding], [chunk_emb]
                )[0][0]

                if sim >= self.chunk_threshold:
                    candidate_chunks.append(
                        {
                            "chunk_id": cid,
                            "text": chunk["text"],
                            "page": chunk["page"],
                            "score": sim,
                        }
                    )

        # 3. Rank and return top-k
        candidate_chunks = sorted(
            candidate_chunks,
            key=lambda x: x["score"],
            reverse=True,
        )

        return candidate_chunks[: self.top_k]


# -------------------------------
# CLI entry point (for testing)
# -------------------------------
if __name__ == "__main__":
    searcher = LocalGraphRAGSearch(
        graph_path="data/processed/knowledge_graph.pkl",
        chunks_path="data/processed/chunks.json",
    )

    results = searcher.search(
        "What was Ambedkar's critique of caste?"
    )

    for r in results:
        print(
            f"[Page {r['page']}] Score={r['score']:.2f}\n{r['text']}\n"
        )
