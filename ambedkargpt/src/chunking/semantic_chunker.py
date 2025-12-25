# src/chunking/semantic_chunker.py

import json
from pathlib import Path
from typing import List

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import tiktoken

# Download required NLTK data
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize


class SemanticChunker:
    """
    Implements Algorithm 1 from the SemRAG paper:
    Semantic Chunking via LLM Embedding and Cosine Similarity
    """

    def __init__(
        self,
        pdf_path: str,
        output_path: str,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.8,
        buffer_size: int = 1,
        max_tokens: int = 1024,
        overlap_tokens: int = 128,
    ):
        self.pdf_path = pdf_path
        self.output_path = output_path
        self.similarity_threshold = similarity_threshold
        self.buffer_size = buffer_size
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

        self.embedder = SentenceTransformer(model_name)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    # 1. Load PDF
    def load_pdf(self) -> List[str]:
        reader = PdfReader(self.pdf_path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return pages

    # 2. Sentence splitting
    def split_sentences(self, text: str) -> List[str]:
        return [s.strip() for s in sent_tokenize(text) if s.strip()]

    # 3. Buffer merging
    def buffer_merge(self, sentences: List[str]) -> List[str]:
        merged = []
        for i in range(len(sentences)):
            start = max(0, i - self.buffer_size)
            end = min(len(sentences), i + self.buffer_size + 1)
            merged.append(" ".join(sentences[start:end]))
        return merged

    # 4. Token count
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    # 5. Semantic chunking using cosine similarity
    def semantic_chunk(self, sentences: List[str]) -> List[str]:
        buffered = self.buffer_merge(sentences)
        embeddings = self.embedder.encode(buffered)

        chunks = []
        current_chunk = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = cosine_similarity(
                [embeddings[i - 1]], [embeddings[i]]
            )[0][0]

            if sim >= self.similarity_threshold:
                current_chunk.append(sentences[i])
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    # 6. Enforce token limits
    def split_large_chunks(self, chunks: List[str]) -> List[str]:
        final_chunks = []

        for chunk in chunks:
            if self.count_tokens(chunk) <= self.max_tokens:
                final_chunks.append(chunk)
            else:
                words = chunk.split()
                step = self.overlap_tokens // 2

                for i in range(0, len(words), step):
                    sub = " ".join(words[i : i + self.overlap_tokens])
                    final_chunks.append(sub)

        return final_chunks

    # 7. Full pipeline
    def run(self):
        pages = self.load_pdf()
        all_chunks = []
        chunk_id = 0

        for page_no, page_text in enumerate(pages):
            sentences = self.split_sentences(page_text)
            if not sentences:
                continue

            chunks = self.semantic_chunk(sentences)
            chunks = self.split_large_chunks(chunks)

            for chunk in chunks:
                all_chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "page": page_no + 1,
                        "text": chunk,
                        "embedding": self.embedder.encode(chunk).tolist(),
                    }
                )
                chunk_id += 1

        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2)

        print(f"[OK] Generated {len(all_chunks)} semantic chunks")


if __name__ == "__main__":
    chunker = SemanticChunker(
        pdf_path="data/Ambedkar_book.pdf",
        output_path="data/processed/chunks.json",
    )
    chunker.run()
