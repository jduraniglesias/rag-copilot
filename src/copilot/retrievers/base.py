from typing import List, Tuple, Protocol

class Retriever(Protocol):
    def search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Return [(chunk_id, score)] sorted by descending score."""
        ...
