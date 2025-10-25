from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import uvicorn
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Reuse your existing plumbing
from copilot.index.persist import save_bm25
from copilot.cli import ensure_bm25_index, _build_retriever
from copilot.qa.answering import answer as answer_short
from copilot.eval.harness import load_gold_jsonl, evaluate_suite
from copilot.eval.runlog import log_run

import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from copilot.sheets.tools import sum_range as tool_sum, pivot_table as tool_pivot
from copilot.sheets.router import parse as parse_intent

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
    rerank_top: int = 0
    high_accuracy: bool = False
    auto_rerank_threshold: float = 0.5
    auto_rerank_top: int = 20

class ActRequest(BaseModel):
    instruction: str = Field(..., description="e.g., 'sum sales where region=west'")
    data: List[Dict[str, Any]] = Field(..., description="List of JSON rows")
    col: Optional[str] = None
    where: Optional[Dict[str, Any]] = None
    index: Optional[str] = None
    values: Optional[str] = None
    aggfunc: Optional[str] = "sum"

class Citation(BaseModel):
    doc_id: str
    char_start: int
    char_end: int
    chunk_id: int
    rank: int
    snippet: str | None = None
    page: int | None = None
    title: str | None = None
    answer_span_start: int | None = None
    answer_span_end: int | None = None
    chunk_span_start: int | None = None
    chunk_span_end: int | None = None
    snippet_span_start: int | None = None
    snippet_span_end: int | None = None

class AskResponse(BaseModel):
    answer: str
    confidence: float
    citations: List[Citation]

class ActResponse(BaseModel):
    kind: str
    explanation: str
    result_number: Optional[float] = None
    result_rows: Optional[List[Dict[str, Any]]] = None

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
        from copilot.cli import _ensure_dense_index_from_bm25
        _ensure_dense_index_from_bm25(bm25, force_rebuild=True)
    # persist BM25 for fast restarts
    save_bm25(bm25, "data/index/bm25.json.gz")
    return {"num_chunks": len(bm25.chunks)}

@app.post("/act", response_model=ActResponse)
def act(req: ActRequest):
    if not req.data:
        raise HTTPException(400, "data must be a non-empty list of rows")
    df = pd.DataFrame(req.data)

    intent = parse_intent(req.instruction)

    if intent.kind == "sum":
        col = req.col or next((c for c in df.columns if c.lower() in ("amount","sales","total","value")), None)
        if not col:
            raise HTTPException(400, "sum: specify 'col' (column to sum)")
        val = tool_sum(df, col=col, where=req.where)
        expl = f"Summed column '{col}'" + (f" with filter {req.where}" if req.where else "")
        return ActResponse(kind="sum", explanation=expl, result_number=val)

    if intent.kind == "pivot":
        index = req.index or next((c for c in df.columns if c.lower() in ("region","category","dept","type")), None)
        values = req.values or next((c for c in df.columns if c.lower() in ("amount","sales","total","value","count")), None)
        if not index or not values:
            raise HTTPException(400, "pivot: specify 'index' and 'values'")
        tbl = tool_pivot(df, index=index, values=values, aggfunc=req.aggfunc or "sum")
        expl = f"Pivoted by '{index}' with values '{values}' using '{req.aggfunc or 'sum'}'."
        return ActResponse(kind="pivot", explanation=expl, result_rows=tbl.to_dict(orient="records"))

    # fallback: treat as QA over docs
    # you could optionally call /ask internally here
    raise HTTPException(501, "Unsupported instruction; try 'sum' or 'pivot'")


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    import re  # local import to keep this block self-contained

    # --- helpers -------------------------------------------------------------
    def _build_ctx_from_results(bm25, results, k_ctx):
        ctx = results[:k_ctx]
        ctx_ids = [cid for cid, _ in ctx]
        passages = [bm25.chunks[cid]["text"] for cid in ctx_ids]
        metas = [bm25.chunks[cid]["meta"] for cid in ctx_ids]
        return ctx, passages, metas

    def _apply_ce_rerank(query: str, bm25, results, top_n: int):
        """Cross-encoder rerank only the top_n for context; keep the rest in original order."""
        topN = results[:min(top_n, len(results))]
        if not topN:
            return results
        from copilot.rerank.cross_encoder import rerank as ce_rerank
        passagesN = [(cid, bm25.chunks[cid]["text"]) for cid, _ in topN]
        ce_sorted = ce_rerank(query, passagesN, top_m=len(topN))
        ce_ids = [cid for cid, _ in ce_sorted]
        ce_idset = set(ce_ids)
        rest = [pair for pair in results if pair[0] not in ce_idset]
        return ce_sorted + rest

    def _make_snippet_and_offsets(txt: str, span: str, window: int = 60):
        """
        Returns (snippet, snippet_span_start, snippet_span_end, chunk_span_start, chunk_span_end)
        If span not found, returns a windowed snippet and all indices as None.
        """
        if not txt:
            return "", None, None, None, None
        s = txt
        if not span:
            snip = (s[: 2 * window] + "…") if len(s) > 2 * window else s
            return snip, None, None, None, None

        i = s.lower().find(span.lower())
        if i < 0:
            snip = (s[: 2 * window] + "…") if len(s) > 2 * window else s
            return snip, None, None, None, None

        # Compute snippet window around the span
        start = max(0, i - window)
        end = min(len(s), i + len(span) + window)
        prefix_ellipsis = start > 0
        suffix_ellipsis = end < len(s)
        snippet = ("…" if prefix_ellipsis else "") + s[start:end] + ("…" if suffix_ellipsis else "")

        # Offsets inside the snippet
        # If we added a leading ellipsis, the span shifts by 1 char in the snippet
        lead = 1 if prefix_ellipsis else 0
        snippet_span_start = (i - start) + lead
        snippet_span_end = snippet_span_start + len(span)

        # Offsets inside the chunk text
        chunk_span_start = i
        chunk_span_end = i + len(span)

        return snippet, snippet_span_start, snippet_span_end, chunk_span_start, chunk_span_end


    # --- retrieval -----------------------------------------------------------
    bm25 = ensure_bm25_index(req.documents)
    retriever_like, _label = _build_retriever(req.retriever, bm25, req.hybrid_mode, req.alpha)

    # Fetch enough for any (manual or auto) rerank
    base_k = max(
        req.k,
        req.rerank_top or 0,
        (req.auto_rerank_top if getattr(req, "high_accuracy", False) else 0),
    )
    results = retriever_like.search(req.query, k=base_k)

    # Optional manual CE rerank (context only)
    ranked = results
    if req.rerank_top and results:
        try:
            ranked = _apply_ce_rerank(req.query, bm25, results, req.rerank_top)
        except Exception:
            ranked = results  # fail-safe

    # First pass: build context → answer
    ctx, passages, metas = _build_ctx_from_results(bm25, ranked, req.k_ctx)
    pred, conf = answer_short(req.query, passages, metas)

    # Auto high-accuracy mode: if low confidence and no manual rerank, try CE and re-answer
    used_auto_rerank = False
    if (
        getattr(req, "high_accuracy", False)
        and not req.rerank_top
        and (conf is None or float(conf) < float(getattr(req, "auto_rerank_threshold", 0.5)))
        and results
    ):
        try:
            ranked2 = _apply_ce_rerank(req.query, bm25, results, getattr(req, "auto_rerank_top", 20))
            ctx, passages, metas = _build_ctx_from_results(bm25, ranked2, req.k_ctx)
            pred, conf = answer_short(req.query, passages, metas)
            used_auto_rerank = True
        except Exception:
            pass  # keep first-pass answer

    # Reorder ctx so 1) the passage containing the span comes first, 2) anchors break ties
    pred_l = (pred or "").strip().lower()
    KEY = re.compile(r"\b(return|refund|exchange)\b", re.I)

    def _features(cid: int) -> tuple[bool, bool]:
        txt = bm25.chunks[cid]["text"]
        has_span = bool(pred_l and pred_l in txt.lower())
        has_anchor = bool(KEY.search(txt))
        # sort by (has_span desc, has_anchor desc) via negation trick
        return (not has_span, not has_anchor)

    ctx = sorted(ctx, key=lambda pair: _features(pair[0]))

    # Build citations with snippet (+ page/title if present in meta)
    cits = []
    for rnk, (cid, _) in enumerate(ctx, start=1):
        m = bm25.chunks[cid]["meta"]
        txt = bm25.chunks[cid]["text"]

        snippet, sn_s, sn_e, ch_s, ch_e = _make_snippet_and_offsets(txt, pred_l, window=60)

        # Absolute (document) offsets: chunk is [char_start, char_end) in the doc
        abs_s = abs_e = None
        if ch_s is not None:
            abs_s = int(m["char_start"]) + ch_s
            abs_e = int(m["char_start"]) + ch_e

        cits.append(Citation(
            doc_id=m["doc_id"],
            char_start=m["char_start"],
            char_end=m["char_end"],
            chunk_id=cid,
            rank=rnk,
            snippet=snippet,
            page=m.get("page"),
            title=m.get("title"),
            answer_span_start=abs_s,
            answer_span_end=abs_e,
            chunk_span_start=ch_s,
            chunk_span_end=ch_e,
            snippet_span_start=sn_s,
            snippet_span_end=sn_e,
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

    cfg = req.model_dump()
    log_run(config=cfg, metrics=suite)

    return suite


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
