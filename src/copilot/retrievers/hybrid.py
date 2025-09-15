from typing import List, Tuple, Dict
from .base import Retriever
from collections import defaultdict
from copilot.retrievers.score_norm import minmax

# Combines BM25 and Dense into one ranking
# Since BM25 is better for exact terms and Dense is better for paraphrases/synonmys,
# having a hybrid scorer is more efficient and provides best of both

# Only looks at ranks
def rrf_merge(bm25: List[Tuple[int,float]], dense: List[Tuple[int,float]], k_const: int = 60) -> List[Tuple[int,float]]:
    # Convert to rank dicts
    r_b = {cid: r for r, (cid, _) in enumerate(bm25, start=1)}
    r_d = {cid: r for r, (cid, _) in enumerate(dense, start=1)}
    cids = set(r_b) | set(r_d)
    fused = {cid: 0.0 for cid in cids}
    for cid in cids:
        if cid in r_b: fused[cid] += 1.0 / (k_const + r_b[cid])
        if cid in r_d: fused[cid] += 1.0 / (k_const + r_d[cid])
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)

# after normalzing, calculates weighted sums of bm25 and dense scores
def weighted_sum(bm25: List[Tuple[int,float]], dense: List[Tuple[int,float]], alpha: float = 0.5) -> List[Tuple[int,float]]:
    bm = {cid: s for cid, s in bm25}
    de = {cid: s for cid, s in dense}
    bm_n = minmax(bm)
    de_n = minmax(de)
    cids = set(bm_n) | set(de_n)
    fused = {cid: alpha * bm_n.get(cid, 0.0) + (1 - alpha) * de_n.get(cid, 0.0) for cid in cids}
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)

class HybridRetriever(Retriever):
    def __init__(self, bm25: Retriever, dense: Retriever, mode: str = "rrf", alpha: float = 0.5):
        self.bm25 = bm25
        self.dense = dense
        self.mode = mode
        self.alpha = alpha

    def search(self, query: str, k: int) -> List[Tuple[int,float]]:
        b = self.bm25.search(query, k)
        d = self.dense.search(query, k)
        if self.mode == "rrf":
            fused = rrf_merge(b, d)
        else:
            fused = weighted_sum(b, d, self.alpha)
        return fused[:k]
