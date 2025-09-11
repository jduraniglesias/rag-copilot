import argparse
from pathlib import Path
from typing import List, Dict, Tuple

# Text & indexing
from copilot.text.chunk import chunk_text
from copilot.index.inverted import build_index
from copilot.index.bm25 import BM25Index

# Evaluation harness & metrics
from copilot.eval.harness import (
    load_gold_jsonl,
    evaluate_retrieval,
    evaluate_qa_baseline,
)

# ---------
# In-memory "context" (simple module-level cache).
# In a real service you'd persist indexes to disk; for now, keep it simple.
# ---------
_ctx = {
    "bm25": None,     # BM25Index
    "docs_dir": None, # Path used to index
    "num_chunks": 0,
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
    return bm25

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
        print(f"{k.ljust(key_w)} : {v:.4f}")

# -----------------------
# Subcommand handlers
# -----------------------
def cmd_index(args: argparse.Namespace) -> None:
    bm25 = ensure_bm25_index(args.documents, size=args.size, overlap=args.overlap)
    print(f"Indexed {_ctx['num_chunks']} chunks from: {Path(args.documents).resolve()}")

def cmd_ask(args: argparse.Namespace) -> None:
    bm25 = ensure_bm25_index(args.documents, size=args.size, overlap=args.overlap)
    results = bm25.search(args.query, k=args.k)
    if not results:
        print("No results.")
        return
    _print_search_results(results, bm25, limit=args.k)

def cmd_evaluate(args: argparse.Namespace) -> None:
    """
    Run retrieval + crude QA baseline evaluation on your gold set.
    (NDCG@k / Recall@k / HitRate@1, plus EM/F1 for top-1 chunk-as-answer.)
    """
    bm25 = ensure_bm25_index(args.documents, size=args.size, overlap=args.overlap)
    gold_path = Path(args.gold)
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold file not found: {gold_path}")

    gold_items = load_gold_jsonl(str(gold_path))

    # Retrieval metrics (ranking quality)
    ret = evaluate_retrieval(bm25, gold_items, k=args.k)
    _print_metrics_table(f"Retrieval (BM25) @k={args.k}", ret)

    # Crude QA baseline: take top-1 chunk text as the answer
    qa = evaluate_qa_baseline(bm25, gold_items, k=1)
    _print_metrics_table("QA baseline (top-1 chunk text)", qa)

# -----------------------
# Main / argparse wiring
# -----------------------
def main():
    parser = argparse.ArgumentParser(
        prog="copilot",
        description="RAG Copilot CLI (BM25 baseline; evaluate against a gold set).",
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
    p_idx.set_defaults(func=cmd_index)

    # ask
    p_ask = sub.add_parser("ask", help="Run a BM25 search over the indexed chunks")
    add_common_args(p_ask)
    p_ask.add_argument("query", type=str, help="Your search query")
    p_ask.add_argument("--k", type=int, default=5, help="Top-k results to return (default: 5)")
    p_ask.set_defaults(func=cmd_ask)

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate retrieval and QA metrics on a gold set")
    add_common_args(p_eval)
    p_eval.add_argument("--gold", type=str, default="data/qa_gold.jsonl",
                        help="Path to gold JSONL file (default: data/qa_gold.jsonl)")
    p_eval.add_argument("--k", type=int, default=5, help="Top-k for retrieval metrics (default: 5)")
    p_eval.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    if not getattr(args, "func", None):
        parser.print_help()
    else:
        args.func(args)

if __name__ == "__main__":
    main()
