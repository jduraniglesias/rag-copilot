import json, gzip
from typing import Any, Dict
from copilot.index.bm25 import BM25Index

def save_bm25(bm25: BM25Index, path: str = "data/index/bm25.json.gz"):
    obj = {
        "postings": bm25.postings,
        "doc_len": bm25.doc_len,
        "avgdl": bm25.avgdl,
        "N": bm25.N,
        "chunks": bm25.chunks,
        "k1": bm25.k1,
        "b": bm25.b,
    }
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f)

def load_bm25(path: str = "data/index/bm25.json.gz") -> BM25Index:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        obj = json.load(f)
    return BM25Index(obj, k1=obj.get("k1", 1.5), b=obj.get("b", 0.75))
