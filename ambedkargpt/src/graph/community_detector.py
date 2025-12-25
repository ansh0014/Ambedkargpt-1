# src/graph/community_detector.py

import pickle
from pathlib import Path

import networkx as nx
import igraph as ig
import leidenalg


class CommunityDetector:
    """
    Community detection using Leiden algorithm.
    """

    def __init__(self, graph_path: str):
        self.graph_path = graph_path
        self.graph = None

    def load_graph(self):
        with open(self.graph_path, "rb") as f:
            self.graph = pickle.load(f)

    def detect_communities(self):
        # Convert NetworkX → iGraph
        nx_graph = self.graph
        ig_graph = ig.Graph.TupleList(
            nx_graph.edges(),
            directed=False
        )

        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.ModularityVertexPartition
        )

        # Assign community IDs back to NetworkX graph
        for idx, community_id in enumerate(partition.membership):
            node = ig_graph.vs[idx]["name"]
            self.graph.nodes[node]["community_id"] = community_id

        print(f"[OK] Detected {len(set(partition.membership))} communities (Leiden)")

    def save_graph(self, output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(self.graph, f)

        print("[OK] Updated Knowledge Graph saved")


if __name__ == "__main__":
    detector = CommunityDetector(
        graph_path="data/processed/knowledge_graph.pkl"
    )
    detector.load_graph()
    detector.detect_communities()
    detector.save_graph("data/processed/knowledge_graph.pkl")
