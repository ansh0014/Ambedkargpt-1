

import json
import pickle
from collections import defaultdict
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class GlobalGraphRAGSearch:
    """
    Implements Global Graph RAG Search (Equation 5 from SemRAG paper)
    """

    def __init__(
        self,
        graph_path: str,
        chunks_path: str,
        top_k_communities: int = 3,
        top_k_chunks: int = 5,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.top_k_communities = top_k_communities
        self.top_k_chunks = top_k_chunks

        self.embedder = SentenceTransformer(embedding_model)

        with open(graph_path, "rb") as f:
            self.graph = pickle.load(f)

        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        self.community_texts = self._build_community_texts()
        self.community_embeddings = {
            cid: self.embedder.encode(text)
            for cid, text in self.community_texts.items()
        }

    def _build_community_texts(self) -> Dict[int, str]:
        """
        Aggregate entity names per community as a proxy summary.
        (Lightweight but SemRAG-compliant)
        """
        communities = defaultdict(list)

        for node, data in self.graph.nodes(data=True):
            cid = data.get("community_id")
            if cid is not None:
                communities[cid].append(node)

        return {cid: " ".join(nodes) for cid, nodes in communities.items()}

    def search(self, query: str) -> List[Dict]:
        query_embedding = self.embedder.encode(query)

        # 1. Rank communities
        scores = []
        for cid, emb in self.community_embeddings.items():
            sim = cosine_similarity(
                [query_embedding], [emb]
            )[0][0]
            scores.append((cid, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_communities = [cid for cid, _ in scores[: self.top_k_communities]]

        # 2. Collect chunks from those communities
        candidate_chunks = []

        for node, data in self.graph.nodes(data=True):
            if data.get("community_id") in top_communities:
                for cid in data.get("chunks", []):
                    chunk = self.chunks[cid]
                    chunk_emb = np.array(chunk["embedding"])

                    sim = cosine_similarity(
                        [query_embedding], [chunk_emb]
                    )[0][0]

                    candidate_chunks.append(
                        {
                            "chunk_id": cid,
                            "text": chunk["text"],
                            "page": chunk["page"],
                            "score": sim,
                        }
                    )

        # 3. Rank and return
        candidate_chunks = sorted(
            candidate_chunks,
            key=lambda x: x["score"],
            reverse=True,
        )

        return candidate_chunks[: self.top_k_chunks]




if __name__ == "__main__":
    searcher = GlobalGraphRAGSearch(
        graph_path="data/processed/knowledge_graph.pkl",
        chunks_path="data/processed/chunks.json",
    )

    results = searcher.search(
        "Explain Ambedkar's views on caste system"
    )

    for r in results:
        print(
            f"[Page {r['page']}] Score={r['score']:.2f}\n{r['text']}\n"
        )
