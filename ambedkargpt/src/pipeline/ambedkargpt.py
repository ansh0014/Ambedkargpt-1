from src.retrieval.local_search import LocalGraphRAGSearch
from src.retrieval.global_search import GlobalGraphRAGSearch
from src.llm.answer_generator import AnswerGenerator


def main():
    print("=== AmbedkarGPT (SemRAG) ===\n")

    GRAPH_PATH = "data/processed/knowledge_graph.pkl"
    CHUNKS_PATH = "data/processed/chunks.json"

    # Initialize retrieval components
    local_search = LocalGraphRAGSearch(
        graph_path=GRAPH_PATH,
        chunks_path=CHUNKS_PATH
    )

    global_search = GlobalGraphRAGSearch(
        graph_path=GRAPH_PATH,
        chunks_path=CHUNKS_PATH
    )

    # Initialize answer generator (ONLY model)
    answer_generator = AnswerGenerator(model="mistral:latest")

    while True:
        query = input("Ask a question about Dr. B.R. Ambedkar:\n").strip()
        if not query:
            print("Please enter a valid question.\n")
            continue

        # Run retrieval explicitly
        local_results = local_search.search(query)
        global_results = global_search.search(query)

        # Generate answer from retrieved context
        answer = answer_generator.generate_answer(
            query=query,
            local_results=local_results,
            global_results=global_results
        )

        print("\n--- Answer ---")
        print(answer)
        print("--------------\n")

        choice = input("Do you want to ask another question? (y/n): ").strip().lower()
        if choice not in ("y", "yes"):
            print("\nExiting AmbedkarGPT. Goodbye!")
            break


if __name__ == "__main__":
    main()
