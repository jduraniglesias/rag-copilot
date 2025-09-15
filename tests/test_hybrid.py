# tests/test_hybrid.py
import pytest

from copilot.retrievers.hybrid import rrf_merge, weighted_sum, HybridRetriever
from copilot.retrievers.score_norm import minmax


def test_rrf_merge_simple():
    b = [(1, 7.0), (2, 6.0), (3, 5.0)]
    d = [(3, .95), (2, .93), (4, .90)]
    fused = rrf_merge(b, d, k_const=60)
    # Should prefer items that rank well in BOTH lists: 2 and 3 near top
    top_ids = [cid for cid, _ in fused[:2]]
    assert any(x in top_ids for x in (2, 3))


def test_rrf_merge_disjoint_lists():
    b = [(10, 1.0), (11, 0.9)]
    d = [(20, 0.8), (21, 0.7)]
    fused = rrf_merge(b, d, k_const=60)
    # All items present in union
    ids = {cid for cid, _ in fused}
    assert ids == {10, 11, 20, 21}
    # Higher ranks (position 1) should dominate within each list
    assert fused[0][0] in (10, 20)


def test_minmax_edge_cases():
    assert minmax({}) == {}
    # Constant scores → all zeros after normalization (since range=0 → rng=1e-9)
    out = minmax({1: 5.0, 2: 5.0})
    assert set(out.keys()) == {1, 2}
    assert all(abs(v - 0.0) < 1e-8 for v in out.values())


def test_weighted_sum_minmax_blend():
    b = [(1, 10.0), (2, 9.0)]         # BM25 scale: arbitrary positives
    d = [(2, 0.95), (3, 0.93)]        # Dense scale: [-1,1]ish cosine
    fused = weighted_sum(b, d, alpha=0.6)

    # Union must include all seen ids
    fused_ids = [cid for cid, _ in fused]
    assert set(fused_ids) == {1, 2, 3}

    # The overlap id (2) should rank above a dense-only id (3)
    assert fused_ids.index(2) < fused_ids.index(3)

    # And it should be near the top; depending on alpha it may be #1 or #2
    assert fused_ids[0] in (1, 2)



class _DummyRetriever:
    """Small test double to isolate HybridRetriever behavior."""
    def __init__(self, results, chunks=None):
        # results: a function (query,k) -> list[(cid, score)]
        self._results = results
        # chunks: minimal structure used only by harness (not needed here)
        self.chunks = chunks if chunks is not None else {}

    def search(self, query: str, k: int):
        return self._results(query, k)


def test_hybrid_retriever_rrf_truncates_k():
    # BM25 returns ids 1..5; Dense returns ids 3..7 with different ranks
    bm25 = _DummyRetriever(lambda q, k: [(i, 10.0 - i) for i in range(1, 6)])
    dense = _DummyRetriever(lambda q, k: [(i, 0.99 - 0.01 * i) for i in range(3, 8)])

    hy = HybridRetriever(bm25, dense, mode="rrf")
    out = hy.search("q", k=3)

    assert len(out) == 3
    # Items 3,4,5 are in both lists; expect these to dominate via RRF
    ids = [cid for cid, _ in out]
    assert any(i in ids for i in (3, 4, 5))


def test_hybrid_retriever_minmax_alpha_tilt():
    # Construct scores so that min-max + alpha makes a difference
    # BM25 likes A strongly; Dense likes B strongly
    bm25_scores = [(100, 100.0), (200, 10.0)]
    dense_scores = [(100, 0.2), (200, 0.99)]

    bm25 = _DummyRetriever(lambda q, k: bm25_scores[:k])
    dense = _DummyRetriever(lambda q, k: dense_scores[:k])

    # Alpha high → favor BM25
    hy_biased_bm25 = HybridRetriever(bm25, dense, mode="minmax", alpha=0.9)
    out1 = hy_biased_bm25.search("q", k=2)
    assert out1[0][0] == 100

    # Alpha low → favor Dense
    hy_biased_dense = HybridRetriever(bm25, dense, mode="minmax", alpha=0.1)
    out2 = hy_biased_dense.search("q", k=2)
    assert out2[0][0] == 200
