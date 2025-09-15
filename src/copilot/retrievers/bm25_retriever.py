from typing import List, Tuple, Optional

from copilot.retrievers.base import Retriever
from copilot.index.bm25 import BM25Index


class BM25Retriever(Retriever):
    """
    Thin adapter around BM25Index to satisfy the Retriever protocol.
    """

    def __init__(self, bm25_index: BM25Index):
        self._bm25 = bm25_index
        self.chunks = bm25_index.chunks

    def search(self, query: str, k: int) -> List[Tuple[int, float]]:
        if not query or k <= 0:
            return []
        return self._bm25.search(query, k)
