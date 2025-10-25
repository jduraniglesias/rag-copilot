import json
import time
import math
import random
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

def split_gold(
    gold_items: list[dict],
    dev_ratio: float = 0.7,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Deterministically shuffle then split gold into (dev, test).
    If dev_ratio>=1 or <=0, returns (gold_items, []) or ([], gold_items).
    """
    items = list(gold_items)
    rng = random.Random(seed)
    rng.shuffle(items)
    n_dev = int(len(items) * dev_ratio)
    return items[:n_dev], items[n_dev:]

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

def _percentile(values, pct: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if pct <= 0: return xs[0]
    if pct >= 100: return xs[-1]
    k = (pct / 100.0) * (len(xs) - 1)
    lo = math.floor(k); hi = math.ceil(k)
    if lo == hi:
        return float(xs[int(k)])
    frac = k - lo
    return float(xs[lo] + (xs[hi] - xs[lo]) * frac)

def evaluate_suite(
    retriever,                   # any object with .search(query, k) and .chunks
    gold_items: List[Dict],
    k_ndcg: int = 10,            # for NDCG
    k_recall: int = 3,           # for Recall@3 and Hit@1
    k_ctx: int = 3,              # how many passages to send to answerer
    answer_fn: Callable[[str, List[str], List[Dict]], Tuple[str, float]] | None = None,
    context_gate_regex: str = r"\b(return|refund|exchange)\b",   # gate noisy passages
    rerank_top: int = 0,
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    rerank_max_length: int = 256,
    rerank_batch_size: int = 16,
) -> Dict[str, float]:
    """
    Runs retrieval once per gold, computes:
      - NDCG@k_ndcg, Recall@k_recall, Hit@1
      - Optional EM/F1 via answer_fn on a re-ranked + gated top-k_ctx context
      - Latencies:
         * search_p50_ms / search_p95_ms (retriever.search only)
         * full_p50_ms / full_p95_ms (retrieval + context + answerer, if answer_fn given)

    Context gating:
      - A passage is eligible if it matches `context_gate_regex` OR passes a token-overlap gate with the question.
      - If no passages match, we fall back to the top-k_ctx unfiltered results.

    Each passage meta gets:
      - retrieval_score (float) and retrieval_rank (1-based), so the answerer can weight sentences.

    Reranking:
      - If rerank_top > 0, we apply a cross-encoder to the top-N retrieved items *for context only*.
        Retrieval metrics are computed on the original ranking.
    """
    import time, re

    OVERLAP_THRESHOLD = 0.15  # token-overlap gate (fraction of q tokens present in passage)

    ndcgs, recalls, hits = [], [], []
    supported = []
    ems, f1s = [], []
    search_ms_all, full_ms_all = [], []

    # We may need more than k_max if reranking the top-N for context
    k_max = max(k_ndcg, k_recall, k_ctx)
    base_k = max(k_max, rerank_top or 0)

    gate = re.compile(context_gate_regex, re.I) if context_gate_regex else None

    for item in gold_items:
        q = item["question"]
        doc_id = item["doc_id"]
        gold_ans = item["answer"]

        # ---- time retrieval only
        t0 = time.perf_counter()
        results = retriever.search(q, k=base_k)
        t1 = time.perf_counter()
        search_ms_all.append((t1 - t0) * 1000.0)

        # span-level relevant chunks for this doc/answer (uses retriever.chunks)
        rel_ids = rel_chunk_ids_by_span_tokens(retriever, doc_id, gold_ans)

        # metrics at different cutoffs (without re-searching)
        labels_ndcg = labels_for_results(results[:k_ndcg], rel_ids, k_ndcg)
        labels_rec  = labels_for_results(results[:k_recall], rel_ids, k_recall)
        labels_hit  = labels_for_results(results[:1],      rel_ids, 1)

        ndcgs.append(ndcg_at_k(labels_ndcg, k_ndcg))
        recalls.append(1.0 if any(labels_rec) else 0.0)
        hits.append(1.0 if (labels_hit and labels_hit[0] > 0) else 0.0)

        # Optional: end-to-end (retrieval already done) → build context + answerer
        if answer_fn is not None:
            if results:
                ranked_for_ctx = results

                # ---- (B) Context-only cross-encoder rerank (preserves retrieval metrics above)
                if rerank_top and rerank_top > 0:
                    topN = results[:min(rerank_top, len(results))]
                    passagesN = [(cid, retriever.chunks[cid]["text"]) for cid, _ in topN]
                    try:
                        from copilot.rerank.cross_encoder import rerank as ce_rerank
                        ce_sorted = ce_rerank(
                            q, passagesN, top_m=len(topN),
                            model_name=rerank_model,
                            max_length=rerank_max_length,
                            batch_size=rerank_batch_size,
                        )
                        ce_ids = [cid for cid, _ in ce_sorted]
                        ce_idset = set(ce_ids)
                        # keep CE order first, then append the rest (still in original order)
                        rest = [pair for pair in results if pair[0] not in ce_idset]
                        ranked_for_ctx = ce_sorted + rest
                    except Exception:
                        # Fail-safe: if CE fails, fall back to original ranking
                        ranked_for_ctx = results

                # ---- context gating by query-term coverage + regex
                q_toks = set(tokenize(q))
                filtered = []
                for cid, s in ranked_for_ctx:
                    text = retriever.chunks[cid]["text"]
                    # token-overlap gate
                    ptoks = set(tokenize(text))
                    overlap_ok = (len(q_toks & ptoks) / max(1, len(q_toks))) >= OVERLAP_THRESHOLD
                    # regex gate
                    regex_ok = bool(gate.search(text)) if gate else False
                    if overlap_ok or regex_ok:
                        filtered.append((cid, s))

                # Take filtered if we got any, otherwise fall back to ranked_for_ctx
                ctx_pairs = (filtered or ranked_for_ctx)[:k_ctx]
                ctx_ids = [cid for cid, _ in ctx_pairs]

                # build context for the answerer and inject retrieval_rank/score
                passages: List[str] = []
                metas: List[Dict] = []
                for rank_idx, (cid, s) in enumerate(ctx_pairs, start=1):
                    passages.append(retriever.chunks[cid]["text"])
                    m = dict(retriever.chunks[cid]["meta"])
                    m["retrieval_score"] = float(s)
                    m["retrieval_rank"] = rank_idx
                    metas.append(m)

                # call your answerer
                t2 = time.perf_counter()
                pred, _conf = answer_fn(q, passages, metas)
                t3 = time.perf_counter()
                full_ms_all.append((t3 - t0) * 1000.0)

                ems.append(exact_match(pred, gold_ans))
                f1s.append(token_f1(pred, gold_ans))
                p = (pred or "").strip().lower()
                supported.append(1.0 if p and any(p in psg.lower() for psg in passages) else 0.0)
            else:
                # no results → still track e2e latency for fairness
                full_ms_all.append((t1 - t0) * 1000.0)
                ems.append(0.0)
                f1s.append(0.0)

    out = {
        f"ndcg@{k_ndcg}": sum(ndcgs) / max(1, len(ndcgs)),
        f"recall@{k_recall}": sum(recalls) / max(1, len(recalls)),
        "hit_rate@1": sum(hits) / max(1, len(hits)),
        "search_p50_ms": _percentile(search_ms_all, 50.0),
        "search_p95_ms": _percentile(search_ms_all, 95.0),
    }
    if answer_fn is not None:
        out.update({
            "em": sum(ems) / max(1, len(ems)),
            "f1": sum(f1s) / max(1, len(f1s)),
            "supported_rate": sum(supported) / max(1, len(supported)),  # NEW
            "full_p50_ms": _percentile(full_ms_all, 50.0),
            "full_p95_ms": _percentile(full_ms_all, 95.0),
        })
    return out

