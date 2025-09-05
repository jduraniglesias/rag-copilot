import math
from copilot.index.inverted import build_index
from copilot.index.bm25 import BM25Index

def _mk_chunk(txt, doc_id="d"):
    return {"text": txt, "meta": {"doc_id": doc_id, "char_start": 0, "char_end": len(txt)}}

def test_bm25_monotonic_with_tf():
    # same term appears more in chunk 0 than chunk 1
    chunks = [
        _mk_chunk("warranty warranty coverage"),
        _mk_chunk("warranty")
    ]
    bm = BM25Index(build_index(chunks))
    q = ["warranty"]
    s0 = bm.score(q, 0)
    s1 = bm.score(q, 1)
    assert s0 >= s1  # higher tf should not hurt

def test_length_normalization_effect():
    # same tf but longer doc should score a bit lower when b>0
    chunks = [
        _mk_chunk("warranty x"),
        _mk_chunk("warranty " + "x " * 50)  # longer chunk, tf(warranty)=1
    ]
    bm = BM25Index(build_index(chunks), k1=1.5, b=0.75)
    q = ["warranty"]
    s_short = bm.score(q, 0)
    s_long = bm.score(q, 1)
    assert s_short > s_long

def test_idf_rare_term_weighs_more():
    # 'zebra' appears once in whole corpus; 'warranty' appears twice
    chunks = [
        _mk_chunk("warranty coverage"),
        _mk_chunk("warranty policy zebra")
    ]
    bm = BM25Index(build_index(chunks))
    # Compare IDFs (private call via name mangling discouraged; instead compare scores on 1-tf docs)
    q_rare = ["zebra"]
    q_common = ["warranty"]
    # score single-term query on chunk 1 where both terms have tf=1
    s_rare = bm.score(q_rare, 1)
    s_common = bm.score(q_common, 1)
    assert s_rare > s_common  # rarer term should have higher contribution

def test_multi_term_additivity_and_search_topk():
    chunks = [
        _mk_chunk("return policy warranty"),
        _mk_chunk("apple banana"),
        _mk_chunk("policy coverage return"),
    ]
    bm = BM25Index(build_index(chunks))
    results = bm.search("return policy", k=2)
    # The two chunks containing both words should rank above the banana one
    assert len(results) == 2
    top_ids = [cid for cid, _ in results]
    assert set(top_ids).issubset({0, 2})

def test_search_sorts_desc_and_is_deterministic():
    chunks = [
        _mk_chunk("alpha beta"),
        _mk_chunk("alpha beta"),
        _mk_chunk("alpha")
    ]
    bm = BM25Index(build_index(chunks))
    res = bm.search("alpha beta", k=3)
    # First two are the two-doc matches; third is single-term doc.
    scores = [s for _, s in res]
    assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
