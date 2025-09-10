import json
import math
from typing import List, Dict, Tuple
from copilot.eval.qa_metrics import exact_match, token_f1
from copilot.eval.rank_metrics import ndcg_at_k

REQUIRED_KEYS = ("id", "question", "answer", "doc_id")

def load_gold_jsonl(path: str) -> List[Dict]:
    """Read JSONL and return a list of dicts with keys: id, question, answer, doc_id."""
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i}: invalid JSON ({e.msg})") from e
            missing = [k for k in REQUIRED_KEYS if k not in obj]
            if missing:
                raise ValueError(f"{path}:{i}: missing required field(s): {', '.join(missing)}")
            rows.append(obj)
    return rows

def relevant_chunk_ids_by_doc(index: dict, doc_id: str) -> List[int]:
    """Return all chunk_ids in index['chunks'] whose meta['doc_id'] == doc_id."""
    chunks: List[int] = []
    for chunk_id, ch in enumerate(index["chunks"]):
        meta = ch.get("meta", {})
        if not isinstance(meta, dict):
            meta = {}
        doc = meta.get("doc_id")
        if doc == doc_id:
            chunks.append(chunk_id)

    return chunks

def labels_for_results(results: List[Tuple[int, float]], relevant_ids: List[int], k: int) -> List[float]:
    """Given search results [(chunk_id, score), ...], return length-k relevance labels (1.0 or 0.0)."""
    labels = List[float] = [0.0] * k
    rel_ids = set(relevant_ids)
    for i in range(len(results)):
        if results[i][0] in rel_ids:
            labels[i] = 1.0
    return labels

def rel_chunk_ids(bm25, doc_id):
    return [i for i, ch in enumerate(bm25.chunks) if ch["meta"]["doc_id"] == doc_id] 

def evaluate_retrieval(bm25, gold_items: List[Dict], k: int = 5) -> Dict[str, float]:
    """
    For each gold item:
      - run bm25.search(question, k)
      - build labels with labels_for_results(...)
      - compute NDCG@k and Recall@k (recall = 1 if any label==1 else 0)
    Return averages: {'ndcg@k': float, 'recall@k': float, 'hit_rate@1': float}
    """
    avgs: Dict[str, float] = {}
    recall_total = []
    ndcg_total = []
    hit_rate_total = []

    for gold_item in gold_items:
        results = bm25.search(gold_item["question"], k)
        rel_ids = rel_chunk_ids(bm25, gold_item["doc_id"])
        labels = labels_for_results(results, rel_ids, k)
        
        hit_rate_total.append(1.0 if labels and labels[0] > 0.0 else 0.0)

        recall_total.append(1.0 if any(labels) else 0.0)
        
        ndcg_total.append(ndcg_at_k(labels, k))
    
    avgs[f"ndcg@{k}"] = (sum(ndcg_total) / max(1, len(ndcg_total)))
    avgs[f"recall@{k}"] = (sum(recall_total) / max(1, len(recall_total)))
    avgs["hit_rate@1"] = (sum(hit_rate_total) / max(1, len(hit_rate_total)))
    
    return avgs


def evaluate_qa_baseline(bm25, gold_items: List[Dict], k: int = 1) -> Dict[str, float]:
    """
    Very crude QA baseline:
      - take top-1 chunk text as the 'prediction'
      - compute EM and token F1 vs gold answer
    Return averages: {'em': float, 'f1': float}
    """
    em_total = []
    f1_total = []
    for gold_item in gold_items:
        results = bm25.search(gold_item["question"], k)
        pred = ""
        if len(results) > 0:
            top_cid, top_score = results[0]
            top_chunk = bm25.chunks[top_cid]
            pred = top_chunk["text"]
        else:
            pred = ""
        em_total.append(exact_match(pred, gold_item["answer"]))
        f1_total.append(token_f1(pred, gold_item["answer"]))
    
    return {"em": sum(em_total) / len(gold_items), "f1": sum(f1_total) / len(gold_items)}
    # TO:DO review code of this eval_qa function