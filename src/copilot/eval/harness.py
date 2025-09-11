import json
import math
from typing import List, Dict, Tuple, Callable
from copilot.eval.qa_metrics import exact_match, token_f1
from copilot.eval.rank_metrics import ndcg_at_k, precision_at_k, mrr_at_k
from collections import Counter
from copilot.text.tokenize import tokenize

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
    labels: List[float] = [0.0] * k
    rel_ids = set(relevant_ids)
    for i in range(len(results)):
        if results[i][0] in rel_ids:
            labels[i] = 1.0
    return labels

def rel_chunk_ids_by_span_tokens(
    bm25, doc_id: str, answer: str, fallback_to_doc: bool = True
):
    """
    Span-level (token-based): chunk is relevant if it contains at least the
    token multiset of the gold answer (case-insensitive, punctuation-robust).
    """
    gold_toks = tokenize(answer or "")
    if not gold_toks:
        # Empty/invalid gold → optionally fall back immediately
        return [i for i, ch in enumerate(bm25.chunks) if ch["meta"]["doc_id"] == doc_id] if fallback_to_doc else []

    gold_cnt = Counter(gold_toks)
    rel = []
    for i, ch in enumerate(bm25.chunks):
        if ch["meta"]["doc_id"] != doc_id:
            continue
        ct = Counter(tokenize(ch["text"]))
        if all(ct[t] >= c for t, c in gold_cnt.items()):
            rel.append(i)

    if not rel and fallback_to_doc:
        rel = [i for i, ch in enumerate(bm25.chunks) if ch["meta"]["doc_id"] == doc_id]
    return rel

# evaluates how well the retriever ranks the correct chunks
def evaluate_retrieval(bm25, gold_items: List[Dict], k: int = 5) -> Dict[str, float]:
    """
    For each gold item:
      - runs bm25.search(question, k)
      - builds labels with labels_for_results(...)
      - computes NDCG@k and Recall@k (recall = 1 if any label==1 else 0)
    """
    avgs: Dict[str, float] = {}
    recall_total = []
    ndcg_total = []
    hit_rate_total = []
    precision_total = []
    mrr_total = []

    for gold_item in gold_items:
        results = bm25.search(gold_item["question"], k)
        rel_ids = rel_chunk_ids_by_span_tokens(bm25, gold_item["doc_id"], gold_item["answer"])
        labels = labels_for_results(results, rel_ids, k)
        
        hit_rate_total.append(1.0 if labels and labels[0] > 0.0 else 0.0)

        recall_total.append(1.0 if any(labels) else 0.0)
        
        ndcg_total.append(ndcg_at_k(labels, k))

        precision_total.append(precision_at_k(labels, k))

        mrr_total.append(mrr_at_k(labels, k))
    
    avgs[f"ndcg@{k}"] = (sum(ndcg_total) / max(1, len(ndcg_total)))
    avgs[f"recall@{k}"] = (sum(recall_total) / max(1, len(recall_total)))
    avgs["hit_rate@1"] = (sum(hit_rate_total) / max(1, len(hit_rate_total)))
    avgs[f"precision@{k}"] = (sum(precision_total) / max(1, len(precision_total)))
    avgs[f"mrr@{k}"] = (sum(mrr_total) / max(1, len(mrr_total)))

    return avgs

# gives a baseline, if top chunk returned, how good is this answer
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
            top_cid = results[0][0]
            top_chunk = bm25.chunks[top_cid]
            pred = top_chunk["text"]
        else:
            pred = ""
        em_total.append(exact_match(pred, gold_item["answer"]))
        f1_total.append(token_f1(pred, gold_item["answer"]))
    
    return {"em": sum(em_total) / len(gold_items), "f1": sum(f1_total) / len(gold_items)}
    # TO:DO review code of this eval_qa function

def evaluate_qa_with_answerer(
    bm25,
    gold_items: List[Dict],
    k_retrieval: int = 5,
    k_ctx: int = 3,
    answer_fn: Callable[[str, List[str], List[Dict]], Tuple[str, float]] = None,
) -> Dict[str, float]:
    """
    1) Retrieve top-k_retrieval chunks.
    2) Build a small context of top k_ctx chunk texts.
    3) Call your answerer to produce a short string.
    4) Compute EM/F1 vs gold answer. Return averages.
    """
    assert answer_fn is not None, "answer_fn required"
    ems, f1s = [], []
    for item in gold_items:
        q = item["question"]
        results = bm25.search(q, k=k_retrieval)
        if not results:
            ems.append(0.0); f1s.append(0.0); continue
        # Context: top k_ctx passages (+ optional meta)
        ctx_ids = [cid for cid, _ in results[:k_ctx]]
        passages = [bm25.chunks[cid]["text"] for cid in ctx_ids]
        metas = [bm25.chunks[cid]["meta"] for cid in ctx_ids]

        pred, _conf = answer_fn(q, passages, metas)
        gold = item["answer"]
        ems.append(exact_match(pred, gold))
        f1s.append(token_f1(pred, gold))
    return {
        "em": sum(ems)/len(ems) if ems else 0.0,
        "f1": sum(f1s)/len(f1s) if f1s else 0.0,
    }