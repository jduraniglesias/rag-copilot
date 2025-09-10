import json
from pathlib import Path

from copilot.text.chunk import chunk_text
from copilot.index.inverted import build_index
from copilot.index.bm25 import BM25Index
from copilot.eval.harness import (
    load_gold_jsonl, relevant_chunk_ids_by_doc, labels_for_results,
    evaluate_retrieval, evaluate_qa_baseline
)
from copilot.eval.rank_metrics import ndcg_at_k

def _mk_chunk(txt, doc_id="d"):
    return {"text": txt, "meta": {"doc_id": doc_id, "char_start": 0, "char_end": len(txt)}}

def test_load_gold_jsonl(tmp_path: Path):
    p = tmp_path / "qa.jsonl"
    rows = [
        {"id":"q1","question":"Q1?","answer":"A1","doc_id":"doc1.txt"},
        {"id":"q2","question":"Q2?","answer":"A2","doc_id":"doc2.txt"},
    ]
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    loaded = load_gold_jsonl(str(p))
    assert len(loaded) == 2 and loaded[0]["id"] == "q1"
    for r in loaded:
        for k in ("id","question","answer","doc_id"):
            assert k in r

def test_relevant_chunk_ids_by_doc_matches_meta():
    chunks = [
        _mk_chunk("warranty coverage", "warranty.txt"),
        _mk_chunk("returns policy", "returns.txt"),
        _mk_chunk("warranty info", "warranty.txt"),
    ]
    index = build_index(chunks)
    ids = set(relevant_chunk_ids_by_doc(index, "warranty.txt"))
    assert ids == {0, 2}
    assert set(relevant_chunk_ids_by_doc(index, "returns.txt")) == {1}

def test_labels_for_results_binary_and_length_k():
    # pretend search results are [(cid, score), ...]
    results = [(5, 0.9), (7, 0.8), (9, 0.1)]
    labels = labels_for_results(results, relevant_ids=[7, 42], k=3)
    assert labels == [0.0, 1.0, 0.0]

def test_evaluate_retrieval_with_bm25():
    # Build a tiny corpus with clear matches
    chunks = [
        _mk_chunk("The warranty period is 30 days.", "warranty.txt"),
        _mk_chunk("Return window requires proof of purchase.", "returns.txt"),
        _mk_chunk("This is unrelated banana text.", "misc.txt"),
    ]
    bm = BM25Index(build_index(chunks))
    gold = [
        {"id":"q1","question":"What is the warranty period?","answer":"30 days","doc_id":"warranty.txt"},
        {"id":"q2","question":"What is needed to return an item?","answer":"Proof of purchase","doc_id":"returns.txt"},
    ]
    metrics = evaluate_retrieval(bm, gold, k=3)
    # We should usually find a relevant chunk in top-3 for these simple queries
    assert 0.0 <= metrics["ndcg@3"] <= 1.0
    assert metrics["recall@3"] >= 0.5
    assert 0.0 <= metrics["hit_rate@1"] <= 1.0

def test_evaluate_qa_baseline_top1_chunk():
    chunks = [
        _mk_chunk("The warranty period is 30 days.", "warranty.txt"),
        _mk_chunk("Proof of purchase is required for returns.", "returns.txt"),
    ]
    bm = BM25Index(build_index(chunks))
    gold = [
        {"id":"q1","question":"What is the warranty period?","answer":"30 days","doc_id":"warranty.txt"},
        {"id":"q2","question":"What is needed to return an item?","answer":"Proof of purchase","doc_id":"returns.txt"},
    ]
    m = evaluate_qa_baseline(bm, gold, k=1)
    # Copying the whole chunk won't exactly match the gold answers, but F1 should be > 0
    assert 0.0 <= m["em"] <= 1.0
    assert m["f1"] > 0.0
