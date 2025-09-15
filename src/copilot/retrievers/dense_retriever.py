from typing import List, Tuple
import numpy as np
from .base import Retriever
from copilot.index.dense_index import DenseIndex
import copilot.index.embeddings as E

class DenseRetriever(Retriever):
    def __init__(self, dense: DenseIndex):
        self.dense = dense

    def search(self, query: str, k: int) -> List[Tuple[int, float]]:
        q = E.encode_query(query)  # [1, dim], normalized
        return self.dense.search(q, k)
