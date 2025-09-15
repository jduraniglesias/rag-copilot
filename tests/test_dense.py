# tests/test_dense.py
import os
import math
import pytest

from copilot.index.dense_index import DenseIndex
from copilot.index import embeddings as E
from copilot.retrievers.dense_retriever import DenseRetriever

def test_dense_index_roundtrip(tmp_path):
    texts = ["alpha beta", "gamma delta", "refund policy applies within 30 days"]
    ids   = [10, 11, 12]

    dim = E.get_model().get_sentence_embedding_dimension()
    idx_path = tmp_path / "faiss.index"
    meta_path = tmp_path / "meta.json"

    di = DenseIndex(dim, str(idx_path), str(meta_path))
    di.build(texts, ids)

    assert idx_path.exists() and meta_path.exists(), "Dense index should persist to disk"

    di2 = DenseIndex(dim, str(idx_path), str(meta_path)).load()
    out = di2.search(E.encode_query("return window length"), k=2)

    # Should return at most k
    assert len(out) == 2
    # Should find the 'policy ... 30 days' chunk somewhere in top-2
    assert any(cid == 12 for cid, _ in out)


def test_dense_index_topk_and_sorting(tmp_path):
    texts = [
        "this is about returns policy and exchanges",
        "warranty information for defects in materials",
        "pricing and promotions page",
        "exchanges allowed within 30 days of delivery",
        "contact support and help"
    ]
    ids = [0, 1, 2, 3, 4]

    dim = E.get_model().get_sentence_embedding_dimension()
    idx_path = tmp_path / "faiss.index"
    meta_path = tmp_path / "meta.json"

    di = DenseIndex(dim, str(idx_path), str(meta_path))
    di.build(texts, ids)

    q = "can I exchange an item after delivery?"
    res = di.search(E.encode_query(q), k=3)

    # size bound
    assert len(res) == 3

    # sorted by similarity (descending)
    sims = [s for _, s in res]
    assert all(sims[i] >= sims[i+1] for i in range(len(sims)-1)), "Results must be sorted by score"

    # sanity: the 'exchanges ... 30 days' row should likely be among top results
    assert any(cid == 3 for cid, _ in res)


def test_dense_index_unicode_texts(tmp_path):
    texts = [
        "Política de devoluciones: 30 días desde la entrega.",
        "Información de garantía por defectos de fabricación.",
        "Intercambios permitidos durante el período especificado."
    ]
    ids = [101, 102, 103]

    dim = E.get_model().get_sentence_embedding_dimension()
    idx_path = tmp_path / "faiss.index"
    meta_path = tmp_path / "meta.json"

    di = DenseIndex(dim, str(idx_path), str(meta_path))
    di.build(texts, ids)

    q = "¿Se permiten intercambios?"
    res = di.search(E.encode_query(q), k=2)
    assert len(res) == 2
    # Should retrieve something, and at least one id is from our set
    assert any(cid in ids for cid, _ in res)


def test_dense_index_preserves_chunk_id_mapping(tmp_path):
    texts = ["t0", "t1", "t2", "t3"]
    ids   = [7, 9, 13, 42]

    dim = E.get_model().get_sentence_embedding_dimension()
    idx_path = tmp_path / "faiss.index"
    meta_path = tmp_path / "meta.json"

    di = DenseIndex(dim, str(idx_path), str(meta_path))
    di.build(texts, ids)

    di2 = DenseIndex(dim, str(idx_path), str(meta_path)).load()
    res = di2.search(E.encode_query("t2"), k=4)

    # All returned cids must be from original ids
    for cid, _ in res:
        assert cid in ids


def test_dense_retriever_adapter(tmp_path):
    texts = [
        "returns must be initiated within 30 days of delivery",
        "gift cards are final sale and non-refundable",
        "warranty covers defects in workmanship for one year"
    ]
    ids = [0, 1, 2]

    dim = E.get_model().get_sentence_embedding_dimension()
    idx_path = tmp_path / "faiss.index"
    meta_path = tmp_path / "meta.json"

    di = DenseIndex(dim, str(idx_path), str(meta_path))
    di.build(texts, ids)

    ret = DenseRetriever(di)
    res = ret.search("how long after delivery can I return?", k=2)
    assert len(res) == 2
    # likely to include the returns/30 days chunk
    assert any(cid == 0 for cid, _ in res)
