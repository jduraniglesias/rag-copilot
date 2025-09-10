from typing import List
import math
import heapq
from copilot.text.tokenize import tokenize

def top_k_scores(results: List[tuple[int, float]], k: int) -> List[tuple[int, float]]:
    return heapq.nlargest(k, results, key=lambda x: x[1])

class BM25Index:
    def __init__(self, index: dict, k1: float = 1.5, b: float = 0.0):
        self.postings = index["postings"]
        self.doc_len = index["doc_len"]
        self.avgdl = index["avgdl"]
        self.N = index["N"]
        self.chunks = index["chunks"]
        
        # k1 is the saturation factor for term frequency (prevents chunks that spam words from dominating)
        # b used to normalize lengths of doc to prevent longer docs from dominating
        self.k1 = float(k1)
        self.b = float(b)

        # precompute doc frequency (num of docs that contain a term)
        # prevents super common words like 'the' from dominating scoring
        self.df = {term: len(post_list) for term, post_list in self.postings.items()}

    # calculates IDF used in BM25
    def _idf(self, term: str) -> float:
        nt = self.df.get(term, 0)
        idf = math.log(((self.N - nt + 0.5) / (nt + 0.5)) + 1.0)
        return idf

    def score(self, query_tokens: List[str], chunk_id: int) -> float:
        sum = 0.0
        for t in query_tokens:
            for cid, tf in self.postings.get(t, []):
                if cid == chunk_id:
                    if tf > 0.0:
                        dl = self.doc_len[chunk_id]
                        t_score = self._idf(t) * ((tf * (self.k1 + 1.0)) / (tf + self.k1*(1.0 - self.b + self.b * dl/self.avgdl)))
                        sum += t_score
        return sum

    # takes in query, scores all candidate chunks with bm25 and returns the top k scoring chunks
    def search(self, query: str, k: int = 10) -> List[tuple[int, float]]:
        query_tokens = tokenize(query)
        candidates: set[int] = set()
        for t in set(query_tokens):
            candidates.update(cid for cid, _ in self.postings.get(t, []))
        
        if not candidates:
            return []

        results: List[tuple[int, float]] = []
        for cid in candidates:
            score = self.score(query_tokens, cid)
            if score > 0.0:
                results.append((cid, score))
        return top_k_scores(results, k)
