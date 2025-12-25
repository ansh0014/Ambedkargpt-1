

from typing import List


def buffer_merge(sentences: List[str], buffer_size: int = 1) -> List[str]:
    """
    Merge neighboring sentences to preserve context.
    Used as part of semantic chunking (SemRAG Algorithm 1).
    """
    merged = []
    for i in range(len(sentences)):
        start = max(0, i - buffer_size)
        end = min(len(sentences), i + buffer_size + 1)
        merged.append(" ".join(sentences[start:end]))
    return merged
