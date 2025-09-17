import argparse
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Text & indexing
from copilot.text.chunk import chunk_text
from copilot.index.inverted import build_index
from copilot.index.bm25 import BM25Index

# QA
from copilot.qa.answering import answer as answer_short

# Evaluation harness & metrics
from copilot.eval.harness import (
    load_gold_jsonl,
    evaluate_retrieval,
    evaluate_qa_baseline,
    evaluate_qa_with_answerer,
)

# NEW: dense + hybrid imports
from copilot.index.dense_index import DenseIndex
from copilot.index import embeddings as E
from copilot.retrievers.bm25_retriever import BM25Retriever
from copilot.retrievers.dense_retriever import DenseRetriever
from copilot.retrievers.hybrid import HybridRetriever

# -----------------------
# Constants
# -----------------------
FAISS_INDEX_PATH = Path("data/index/faiss.index")
FAISS_META_PATH = Path("data/index/dense_meta.json")

# -----------------------
# In-memory "context"
# -----------------------
_ctx = {
    "bm25": None,         # BM25Index
    "docs_dir": None,     # Path used to index
    "num_chunks": 0,
    "dense": None,        # DenseIndex
    "dense_ready": False, # bool
}

# -----------------------
# Utilities
# -----------------------
def load_documents(dir_path: Path, size: int = 600, overlap: int = 120) -> List[Dict]:
    """
    Load .txt files from a folder and return a list of chunks.
    (chunk = slice of a larger doc; overlap = how much each chunk shares with the next)
    """
    chunks: List[Dict] = []
    txt_files = sorted(dir_path.glob("*.txt"))
    for p in txt_files:
        text = p.read_text(encoding="utf-8", errors="ignore")
        chunks.extend(chunk_text(text, doc_id=p.name, size=size, overlap=overlap))
    return chunks

def ensure_bm25_index(documents: str, size: int = 600, overlap: int = 120) -> BM25Index:
    """
    Ensure we have a BM25 index in memory; (re)build if the documents path changed.
    """
    global _ctx
    docs_dir = Path(documents)
    if _ctx["bm25"] is not None and _ctx["docs_dir"] == docs_dir:
        return _ctx["bm25"]

    if not docs_dir.exists() or not docs_dir.is_dir():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

    chunks = load_documents(docs_dir, size=size, overlap=overlap)
    index = build_index(chunks)
    bm25 = BM25Index(index)

    _ctx["bm25"] = bm25
    _ctx["docs_dir"] = docs_dir
    _ctx["num_chunks"] = len(chunks)
    # changing docs invalidates any prior dense cache
    _ctx["dense"] = None
    _ctx["dense_ready"] = False
    return bm25

def _chunks_as_ids_texts(chunks):
    """
    Return (chunk_ids, chunk_texts) from bm25.chunks whether it's a list or dict.
    """
    if isinstance(chunks, dict):
        ids = sorted(chunks.keys())
        texts = [chunks[cid]["text"] for cid in ids]
    else:
        # assume sequence/list
        ids = list(range(len(chunks)))
        texts = [ch["text"] for ch in chunks]
    return ids, texts

def _ensure_dense_index_from_bm25(bm25: BM25Index, force_rebuild: bool = False) -> DenseIndex:
    """
    Ensure a DenseIndex exists on disk and in memory. Builds from bm25.chunks if missing or forced.
    """
    global _ctx
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    need_build = force_rebuild or (not FAISS_INDEX_PATH.exists() or not FAISS_META_PATH.exists())
    if need_build:
        chunk_ids, chunk_texts = _chunks_as_ids_texts(bm25.chunks)

        # sanity
        if not chunk_texts:
            raise RuntimeError("No chunks available to build dense index.")
        if any("text" not in bm25.chunks[cid] for cid in (chunk_ids if isinstance(bm25.chunks, dict) else range(len(bm25.chunks)))):
            raise RuntimeError("Each chunk must have a 'text' field to build dense index.")

        dim = E.get_model().get_sentence_embedding_dimension()
        dense = DenseIndex(dim, str(FAISS_INDEX_PATH), str(FAISS_META_PATH))
        dense.build(chunk_texts, chunk_ids)
        _ctx["dense"] = dense
        _ctx["dense_ready"] = True
        return dense

    # load cached
    if _ctx["dense"] is None or not _ctx["dense_ready"]:
        dim = E.get_model().get_sentence_embedding_dimension()
        dense = DenseIndex(dim, str(FAISS_INDEX_PATH), str(FAISS_META_PATH)).load()
        _ctx["dense"] = dense
        _ctx["dense_ready"] = True
        return dense

    return _ctx["dense"]

def _print_search_results(results: List[Tuple[int, float]], bm25: BM25Index, limit: int) -> None:
    """
    Pretty-print top search results with basic 'citations'
    (doc_id + character offsets = where in the original text the chunk came from).
    """
    for rank, (cid, score) in enumerate(results[:limit], start=1):
        ch = bm25.chunks[cid]
        meta = ch["meta"]
        snippet = ch["text"].replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:200] + "…"
        print(
            f"[{rank}] score={score:.4f} "
            f"doc={meta['doc_id']} "
            f"chars=({meta['char_start']},{meta['char_end']})"
        )
        print(f"    {snippet}\n")

def _print_metrics_table(title: str, rows: Dict[str, float]) -> None:
    """
    Print a simple key/value metrics table.
    """
    print(f"\n== {title} ==")
    key_w = max(len(k) for k in rows.keys()) if rows else 0
    for k in sorted(rows.keys()):
        v = rows[k]
        # values may be non-floats (e.g., latency strings); handle gracefully
        try:
            print(f"{k.ljust(key_w)} : {float(v):.4f}")
        except Exception:
            print(f"{k.ljust(key_w)} : {v}")

# -----------------------
# Retriever wiring
# -----------------------
def _build_retriever(
    mode: str,
    bm25: BM25Index,
    hybrid_mode: str = "rrf",
    alpha: float = 0.6,
) -> Tuple[object, str]:
    """
    Return (retriever_like, label). The returned object has .search(query, k).
    For non-BM25 modes, we wrap the retriever so evaluation can still access bm25.chunks.
    """
    bm25_ret = BM25Retriever(bm25)
    label = mode.upper()

    if mode == "bm25":
        return bm25_ret, "BM25"

    # Ensure dense index exists/loaded
    dense = _ensure_dense_index_from_bm25(bm25, force_rebuild=False)
    dense_ret = DenseRetriever(dense)

    if mode == "dense":
        # Wrap to expose bm25 chunks for harness (which expects .chunks)
        return _BM25Compatible(bm25, dense_ret), "Dense"

    if mode == "hybrid":
        hy_ret = HybridRetriever(bm25_ret, dense_ret, mode=hybrid_mode, alpha=alpha)
        return _BM25Compatible(bm25, hy_ret), f"Hybrid-{hybrid_mode.upper()}"

    raise ValueError(f"Unknown retriever mode: {mode}")

class _BM25Compatible:
    """
    Minimal adapter so evaluate_* functions that expect an object with:
      - .search(query, k)
      - .chunks (to map chunk_id -> text/meta for span checks)
    keep working for Dense/Hybrid.
    """
    def __init__(self, bm25: BM25Index, retriever):
        self._ret = retriever
        self.chunks = bm25.chunks

    def search(self, query: str, k: int) -> List[Tuple[int, float]]:
        return self._ret.search(query, k)

# -----------------------
# Subcommand handlers
# -----------------------
def cmd_index(args: argparse.Namespace) -> None:
    bm25 = ensure_bm25_index(args.documents, size=args.size, overlap=args.overlap)
    print(f"Indexed {_ctx['num_chunks']} chunks from: {Path(args.documents).resolve()}")

    if args.dense:
        _ensure_dense_index_from_bm25(bm25, force_rebuild=True)
        print(f"Dense index built at: {FAISS_INDEX_PATH} (meta: {FAISS_META_PATH})")

def cmd_ask(args: argparse.Namespace) -> None:
    bm25 = ensure_bm25_index(args.documents, size=args.size, overlap=args.overlap)
    retriever_like, label = _build_retriever(
        mode=args.retriever,
        bm25=bm25,
        hybrid_mode=args.hybrid_mode,
        alpha=args.alpha,
    )
    base_k = max(args.k, args.rerank_top or 0)
    results = retriever_like.search(args.query, k=base_k)

    # Optional cross-encoder rerank for top-N
    if args.rerank_top and results:
        topN = results[:args.rerank_top]
        # Use BM25 chunk store for text/metadata
        passages = [(cid, bm25.chunks[cid]["text"]) for cid, _ in topN]
        from copilot.rerank.cross_encoder import rerank as ce_rerank
        results = ce_rerank(args.query, passages, top_m=args.k)

    if not results:
        print("No results.")
        return
    print(f"[Retriever: {label}]")
    _print_search_results(results, bm25, limit=args.k)

def cmd_evaluate(args: argparse.Namespace) -> None:
    """
    Side-by-side comparison table for retrieval + QA (+ latency).
    """
    from copilot.eval.harness import load_gold_jsonl, evaluate_suite

    bm25 = ensure_bm25_index(args.documents, size=args.size, overlap=args.overlap)

    gold_path = Path(args.gold)
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold file not found: {gold_path}")
    gold_items = load_gold_jsonl(str(gold_path))

    # Which retrievers to run
    modes = [args.retriever] if args.retriever != "all" else ["bm25", "dense", "hybrid"]

    # Collect rows
    rows = []
    for mode in modes:
        r_like, label = _build_retriever(
            mode=mode,
            bm25=bm25,
            hybrid_mode=args.hybrid_mode,
            alpha=args.alpha,
        )
        suite = evaluate_suite(
            r_like,
            gold_items,
            k_ndcg=max(10, args.k),
            k_recall=3,
            k_ctx=args.k_ctx,
            answer_fn=answer_short if args.qa == "answerer" else None,
            rerank_top=getattr(args, "rerank_top", 0),
        )

        # Decide which latency to display
        p95_ms = suite.get("full_p95_ms") if args.qa == "answerer" else suite.get("search_p95_ms")

        rows.append({
            "Retriever": label,
            "NDCG@10":   suite.get("ndcg@10", 0.0),
            "Recall@3":  suite.get("recall@3", 0.0),
            "Hit@1":     suite.get("hit_rate@1", 0.0),
            "EM":        suite.get("em", float("nan")) if args.qa == "answerer" else float("nan"),
            "F1":        suite.get("f1", float("nan")) if args.qa == "answerer" else float("nan"),
            "p95_ms":    p95_ms if p95_ms is not None else float("nan"),
        })

    # Pretty-print a compact table
    headers = ["Retriever", "NDCG@10", "Recall@3", "Hit@1", "EM", "F1", "p95_ms"]
    col_w = {h: max(len(h), max(len(f"{row[h]:.4f}") if isinstance(row[h], float) else len(str(row[h])) for row in rows)) for h in headers}
    # header
    line = "  ".join(h.ljust(col_w[h]) for h in headers)
    print("\n" + line)
    print("-" * len(line))
    # rows
    for row in rows:
        def fmt(v):
            if isinstance(v, float):
                if math.isnan(v):
                    return "--"
                return f"{v:.4f}"
            return str(v)
        print("  ".join(fmt(row[h]).ljust(col_w[h]) for h in headers))

def _parse_grid(arg: str, cast):
    vals = [v.strip() for v in arg.split(",") if v.strip()]
    return [cast(v) for v in vals]

def cmd_tune(args: argparse.Namespace) -> None:
    """
    Grid-search k_ctx (and alpha for hybrid) on a DEV split; report best config and TEST scores.
    """
    from copilot.eval.harness import load_gold_jsonl, evaluate_suite, split_gold

    bm25 = ensure_bm25_index(args.documents, size=args.size, overlap=args.overlap)

    gold_path = Path(args.gold)
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold file not found: {gold_path}")
    gold_items = load_gold_jsonl(str(gold_path))
    dev_items, test_items = split_gold(gold_items, dev_ratio=args.dev_ratio, seed=args.seed)
    if not dev_items or not test_items:
        # fallback: tune and report on the same set if too small
        dev_items, test_items = gold_items, gold_items

    # grids
    kctx_grid = _parse_grid(args.k_ctx_grid, int)
    alpha_grid = _parse_grid(args.alpha_grid, float) if args.retriever == "hybrid" else [None]

    # build base retriever objects once per loop (we pass alpha through _build_retriever)
    def run_suite(mode, alpha, items):
        r_like, label = _build_retriever(
            mode=mode,
            bm25=bm25,
            hybrid_mode=args.hybrid_mode,
            alpha=(alpha if alpha is not None else 0.0),
        )

        # If using cross-encoder rerank during tuning, wrap evaluate_suite to apply it.
        # Simple inline variant: rerank only the context ordering, not retrieval metrics
        # (keeps NDCG stable; EM/F1 reflect re-ordered context).
        def suite_with_optional_rerank(items):
            # replicate evaluate_suite but inject rerank before building passages
            # Here we call evaluate_suite directly (no rerank path),
            # because the "official" approach is to integrate CE inside evaluate_suite.
            # If you've already added CE inside evaluate_suite, this is not needed.
            return evaluate_suite(
                r_like,
                items,
                k_ndcg=max(10, args.k),
                k_recall=3,
                k_ctx=max(kctx_grid),  # we'll override context size per combo when interpreting results
                answer_fn=answer_short,  # we care about EM/F1
            )

        return evaluate_suite(
            r_like,
            items,
            k_ndcg=max(10, args.k),
            k_recall=3,
            k_ctx=max(kctx_grid),  # use max; answerer will still get top k_ctx passages later
            answer_fn=answer_short,
        )

    # Tune on DEV
    trials = []
    best = None  # (F1, EM, config, suite)
    for alpha in alpha_grid:
        for kctx in kctx_grid:
            r_like, label = _build_retriever(
                mode=args.retriever,
                bm25=bm25,
                hybrid_mode=args.hybrid_mode,
                alpha=(alpha if alpha is not None else 0.0),
            )
            # Evaluate with chosen k_ctx
            suite = evaluate_suite(
                r_like,
                dev_items,
                k_ndcg=max(10, args.k),
                k_recall=3,
                k_ctx=kctx,
                answer_fn=answer_short,
            )
            trials.append((alpha, kctx, suite))
            key = (suite.get("f1", 0.0), suite.get("em", 0.0), suite.get("ndcg@10", 0.0))
            if best is None or key > (best[0], best[1], best[2]):
                best = (suite["f1"], suite["em"], suite.get("ndcg@10", 0.0), alpha, kctx, suite)

    # Print DEV grid
    print("\nDEV grid (higher is better)")
    print("alpha   k_ctx   F1       EM       NDCG@10")
    print("------------------------------------------")
    for alpha, kctx, suite in trials:
        a = "--" if alpha is None else f"{alpha:.2f}"
        print(f"{a:<7} {kctx:<6} {suite.get('f1',0):.4f}  {suite.get('em',0):.4f}  {suite.get('ndcg@10',0):.4f}")

    # Best config
    best_f1, best_em, best_ndcg, best_alpha, best_kctx, best_suite = best
    print("\nBest on DEV:")
    a = "--" if best_alpha is None else f"{best_alpha:.2f}"
    print(f"retriever={args.retriever} mode={args.hybrid_mode} alpha={a} k_ctx={best_kctx}")
    print(f"F1={best_f1:.4f} EM={best_em:.4f} NDCG@10={best_ndcg:.4f}")

    # Evaluate on TEST with best config
    r_like, _ = _build_retriever(
        mode=args.retriever,
        bm25=bm25,
        hybrid_mode=args.hybrid_mode,
        alpha=(best_alpha if best_alpha is not None else 0.0),
    )
    test_suite = evaluate_suite(
        r_like,
        test_items,
        k_ndcg=max(10, args.k),
        k_recall=3,
        k_ctx=best_kctx,
        answer_fn=answer_short,
    )

    print("\nTEST results (best config)")
    for k in ["ndcg@10", "recall@3", "hit_rate@1", "em", "f1", "full_p95_ms"]:
        if k in test_suite:
            print(f"{k}: {test_suite[k]:.4f}")

# -----------------------
# Main / argparse wiring
# -----------------------
def main():
    parser = argparse.ArgumentParser(
        prog="copilot",
        description="RAG Copilot CLI (BM25 baseline + Dense/Hybrid; evaluate against a gold set).",
    )
    sub = parser.add_subparsers(dest="cmd")

    # Shared args helper
    def add_common_args(p):
        p.add_argument("--documents", type=str, default="data/documents",
                       help="Folder with .txt files to index (default: data/documents)")
        p.add_argument("--size", type=int, default=600,
                       help="Chunk size in characters (default: 600)")
        p.add_argument("--overlap", type=int, default=120,
                       help="Chunk overlap in characters (default: 120)")

    # index
    p_idx = sub.add_parser("index", help="Index .txt files in a folder")
    add_common_args(p_idx)
    p_idx.add_argument("--dense", action="store_true",
                       help="Also (re)build the FAISS dense index")
    p_idx.set_defaults(func=cmd_index)

    # ask
    p_ask = sub.add_parser("ask", help="Run a search over the indexed chunks")
    add_common_args(p_ask)
    p_ask.add_argument("query", type=str, help="Your search query")
    p_ask.add_argument("--k", type=int, default=5, help="Top-k results to return (default: 5)")
    p_ask.add_argument("--retriever", choices=["bm25", "dense", "hybrid"], default="hybrid",
                       help="Which retriever to use (default: bm25)")
    p_ask.add_argument("--hybrid-mode", choices=["rrf", "minmax"], default="minmax",
                       help="Hybrid fusion strategy (default: rrf)")
    p_ask.add_argument("--alpha", type=float, default=0.8,
                       help="Weighted-sum blend for hybrid minmax (ignored for rrf)")
    p_ask.add_argument("--rerank-top", type=int, default=0,
                   help="If >0, cross-encoder reranks the top-N candidates before printing")
    p_ask.set_defaults(func=cmd_ask)

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate retrieval and QA metrics on a gold set")
    add_common_args(p_eval)
    p_eval.add_argument("--gold", type=str, default="data/qa_gold.jsonl",
                        help="Path to gold JSONL file (default: data/qa_gold.jsonl)")
    p_eval.add_argument("--k", type=int, default=5, help="Top-k for retrieval metrics (default: 5)")
    p_eval.add_argument(
        "--qa",
        type=str,
        choices=["baseline", "answerer"],
        default="baseline",
        help="QA scoring mode: 'baseline' uses top-1 chunk text; 'answerer' uses your extractor."
    )
    p_eval.add_argument(
        "--k-ctx",
        type=int,
        default=4,
        help="How many top chunks to pass to the answerer context."
    )
    p_eval.add_argument("--retriever", choices=["bm25", "dense", "hybrid", "all"], default="bm25",
                        help="Choose a retriever to evaluate, or 'all' to compare")
    p_eval.add_argument("--hybrid-mode", choices=["rrf", "minmax"], default="rrf",
                        help="Hybrid fusion strategy when retriever=hybrid or all")
    p_eval.add_argument("--alpha", type=float, default=0.8,
                        help="Weighted-sum blend for hybrid minmax (ignored for rrf)")
    p_eval.add_argument("--rerank-top", type=int, default=0,
                    help="If >0, cross-encoder reranks the top-N before building the answerer context")
    p_eval.set_defaults(func=cmd_evaluate)

    p_tune = sub.add_parser("tune", help="Grid-search retriever/ctx knobs on a dev split, then report on test")
    add_common_args(p_tune)
    p_tune.add_argument("--gold", type=str, default="data/qa_gold.jsonl",
                        help="Path to gold JSONL file (default: data/qa_gold.jsonl)")
    p_tune.add_argument("--retriever", choices=["bm25", "dense", "hybrid"], default="hybrid",
                        help="Retriever to tune (default: hybrid)")
    p_tune.add_argument("--hybrid-mode", choices=["rrf", "minmax"], default="minmax",
                        help="Hybrid fusion strategy if retriever=hybrid (default: minmax)")
    p_tune.add_argument("--alpha-grid", type=str, default="0.6,0.7,0.8,0.9",
                        help="Comma-separated alphas for hybrid minmax (ignored otherwise)")
    p_tune.add_argument("--k-ctx-grid", type=str, default="2,3,4",
                        help="Comma-separated k_ctx values to try")
    p_tune.add_argument("--k", type=int, default=10,
                        help="Top-k to retrieve before context/rerank (default: 10)")
    p_tune.add_argument("--dev-ratio", type=float, default=0.7,
                        help="Portion of gold used for dev (default: 0.7)")
    p_tune.add_argument("--seed", type=int, default=42, help="Split seed (default: 42)")
    # optional: enable reranker during tuning too
    p_tune.add_argument("--rerank-top", type=int, default=0,
                        help="If >0, cross-encoder rerank top-N before building context")
    p_tune.set_defaults(func=cmd_tune)

    args = parser.parse_args()
    if not getattr(args, "func", None):
        parser.print_help()
    else:
        args.func(args)

if __name__ == "__main__":
    main()
