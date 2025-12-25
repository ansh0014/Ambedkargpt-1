

from typing import List, Dict


def rank_by_score(items: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Rank retrieved items by similarity score.
    """
    return sorted(
        items,
        key=lambda x: x.get("score", 0),
        reverse=True
    )[:top_k]
