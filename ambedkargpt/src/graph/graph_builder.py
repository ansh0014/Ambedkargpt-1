# src/graph/graph_builder.py

import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import networkx as nx


class KnowledgeGraphBuilder:
    """
    Builds a Knowledge Graph from entities extracted from semantic chunks.
    Nodes = Entities
    Edges = Co-occurrence relationships inside the same chunk
    """

    def __init__(self):
        self.graph = nx.Graph()

    def build_graph(self, chunks: List[Dict]):
        """
        Build graph using entity co-occurrence.
        """
        for chunk in chunks:
            entities = chunk.get("entities", [])
            chunk_id = chunk.get("chunk_id")

            # Add entity nodes
            for ent in entities:
                node_id = ent["text"]

                if not self.graph.has_node(node_id):
                    self.graph.add_node(
                        node_id,
                        label=ent["label"],
                        chunks=set(),
                    )

                self.graph.nodes[node_id]["chunks"].add(chunk_id)

            # Add edges between co-occurring entities
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    e1 = entities[i]["text"]
                    e2 = entities[j]["text"]

                    if self.graph.has_edge(e1, e2):
                        self.graph[e1][e2]["weight"] += 1
                    else:
                        self.graph.add_edge(
                            e1,
                            e2,
                            relationship="CO_OCCURS",
                            weight=1,
                        )

    def save_graph(self, output_path: str):
        """
        Save the graph to disk.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(self.graph, f)

        print(
            f"[OK] Knowledge Graph saved with "
            f"{self.graph.number_of_nodes()} nodes and "
            f"{self.graph.number_of_edges()} edges"
        )


# -------------------------------
# CLI entry point (for testing)
# -------------------------------
if __name__ == "__main__":
    with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    builder = KnowledgeGraphBuilder()
    builder.build_graph(chunks)
    builder.save_graph("data/processed/knowledge_graph.pkl")
