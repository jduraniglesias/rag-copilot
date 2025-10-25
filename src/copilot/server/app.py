from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Reuse your existing plumbing
from copilot.cli import ensure_bm25_index, _build_retriever
from copilot.qa.answering import answer as answer_short
from copilot.eval.harness import load_gold_jsonl, evaluate_suite

app = FastAPI(title="RAG Copilot API", version="0.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---- Schemas
class IndexRequest(BaseModel):
    documents: str = "data/documents"
    size: int = 600
    overlap: int = 120
    dense: bool = True  # also build dense/faiss

class AskRequest(BaseModel):
    query: str
    k: int = 10
    k_ctx: int = 3
    documents: str = "data/documents"
    retriever: str = Field("hybrid", pattern="^(bm25|dense|hybrid)$")
    hybrid_mode: str = Field("minmax", pattern="^(rrf|minmax)$")
    alpha: float = 0.8
    rerank_top: int = 0  # 0 = off

class Citation(BaseModel):
    doc_id: str
    char_start: int
    char_end: int
    chunk_id: int
    rank: int

class AskResponse(BaseModel):
    answer: str
    confidence: float
    citations: List[Citation]

class EvaluateRequest(BaseModel):
    documents: str = "data/documents"
    gold: str = "data/qa_gold.jsonl"
    retriever: str = Field("hybrid", pattern="^(bm25|dense|hybrid)$")
    hybrid_mode: str = "minmax"
    alpha: float = 0.8
    k: int = 10
    k_ctx: int = 3
    rerank_top: int = 0
    # you can expose more knobs later

# ---- Routes
@app.post("/index")
def index(req: IndexRequest):
    bm25 = ensure_bm25_index(req.documents, size=req.size, overlap=req.overlap)
    if req.dense:
        # reuse CLI helper to (re)build dense
        from copilot.cli import _ensure_dense_index_from_bm25
        _ensure_dense_index_from_bm25(bm25, force_rebuild=True)
    return {"num_chunks": len(bm25.chunks)}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    bm25 = ensure_bm25_index(req.documents)
    retriever_like, _label = _build_retriever(req.retriever, bm25, req.hybrid_mode, req.alpha)

    base_k = max(req.k, req.rerank_top or 0)
    results = retriever_like.search(req.query, k=base_k)

    ranked = results
    if req.rerank_top and results:
        topN = results[:req.rerank_top]
        passages = [(cid, bm25.chunks[cid]["text"]) for cid, _ in topN]
        from copilot.rerank.cross_encoder import rerank as ce_rerank
        ranked = ce_rerank(req.query, passages, top_m=len(topN)) + results[req.rerank_top:]

    # Build context for the answerer (top k_ctx)
    ctx = ranked[:req.k_ctx]
    ctx_ids = [cid for cid, _ in ctx]
    passages = [bm25.chunks[cid]["text"] for cid in ctx_ids]
    metas = [bm25.chunks[cid]["meta"] for cid in ctx_ids]

    # Get the short answer
    pred, conf = answer_short(req.query, passages, metas)

    pred_l = (pred or "").strip().lower()
    if pred_l:
        def _contains(cid: int) -> bool:
            return pred_l in bm25.chunks[cid]["text"].lower()
        # sort puts True first (False=1, True=0 via the negation trick)
        ctx = sorted(ctx, key=lambda pair: (not _contains(pair[0]),))

    cits = []
    for rnk, (cid, _) in enumerate(ctx, start=1):
        m = bm25.chunks[cid]["meta"]
        cits.append(Citation(
            doc_id=m["doc_id"],
            char_start=m["char_start"],
            char_end=m["char_end"],
            chunk_id=cid,
            rank=rnk
        ))

    return AskResponse(answer=pred, confidence=float(conf or 0.0), citations=cits)


@app.post("/evaluate")
def evaluate(req: EvaluateRequest):
    bm25 = ensure_bm25_index(req.documents)
    gold_path = Path(req.gold)
    if not gold_path.exists():
        raise HTTPException(404, f"Gold not found: {gold_path}")

    r_like, label = _build_retriever(req.retriever, bm25, req.hybrid_mode, req.alpha)
    suite = evaluate_suite(
        r_like,
        load_gold_jsonl(str(gold_path)),
        k_ndcg=max(10, req.k),
        k_recall=3,
        k_ctx=req.k_ctx,
        answer_fn=answer_short,
        rerank_top=req.rerank_top,
    )
    suite["retriever"] = label

    from copilot.eval.runlog import log_run
    log_run({"cfg": req.dict(), "metrics": suite})

    return suite


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
