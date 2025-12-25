# src/pipeline/ambedkargpt.py

import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.retrieval.local_search import LocalGraphRAGSearch
from src.retrieval.global_search import GlobalGraphRAGSearch
from src.llm.answer_generator import AnswerGenerator


def main():
    print("=== AmbedkarGPT (SemRAG) ===\n")

    query = input("Ask a question about Dr. B.R. Ambedkar: ")

    # 1. Local Graph RAG Search (Equation 4)
    local_search = LocalGraphRAGSearch(
        graph_path="data/processed/knowledge_graph.pkl",
        chunks_path="data/processed/chunks.json",
    )
    local_results = local_search.search(query)

    # 2. Global Graph RAG Search (Equation 5)
    global_search = GlobalGraphRAGSearch(
        graph_path="data/processed/knowledge_graph.pkl",
        chunks_path="data/processed/chunks.json",
    )
    global_results = global_search.search(query)

    # 3. Generate Answer using LLM

    answer_generator = AnswerGenerator(model="mistral:latest")
    answer = answer_generator.generate_answer(
        query=query,
        local_results=local_results,
        global_results=global_results,
    )

    print("\n=== Answer ===\n")
    print(answer)


if __name__ == "__main__":
    main()
