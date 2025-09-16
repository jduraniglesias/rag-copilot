from typing import List, Tuple
from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

# reranker that reads query and passage and reorders the top candidates
# based using a tiny Transformer
@lru_cache(maxsize=1)
def _load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    mdl.eval()
    return tok, mdl

@torch.inference_mode()
def rerank(query: str, passages: List[Tuple[int, str]], top_m: int) -> List[Tuple[int, float]]:
    tok, mdl = _load_model()
    pairs = [(query, p) for _, p in passages]
    batch = tok.batch_encode_plus(pairs, padding=True, truncation=True, max_length=256, return_tensors="pt")
    logits = mdl(**batch).logits.squeeze(-1)  # [B]
    scores = logits.tolist() if isinstance(logits, torch.Tensor) else logits
    scored = [(passages[i][0], float(scores[i])) for i in range(len(passages))]
    return sorted(scored, key=lambda x: x[1], reverse=True)[:top_m]
