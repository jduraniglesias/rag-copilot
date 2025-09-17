from __future__ import annotations
from typing import List, Tuple, Optional
from functools import lru_cache

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def _get_device(explicit: Optional[str] = None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@lru_cache(maxsize=2)
def _load_model(model_name: str = DEFAULT_MODEL):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    mdl.eval()
    return tok, mdl

@torch.inference_mode()
def rerank(
    query: str,
    passages: List[Tuple[int, str]],   # [(chunk_id, text), ...]
    top_m: int,
    model_name: str = DEFAULT_MODEL,
    device: Optional[str] = None,
    max_length: int = 256,
    batch_size: int = 16,
) -> List[Tuple[int, float]]:
    """
    Returns top_m [(chunk_id, ce_score)] sorted by CE score desc.
    """
    if not passages:
        return []
    tok, mdl = _load_model(model_name)
    dev = _get_device(device)
    mdl.to(dev)

    # Batch over (query, passage)
    all_scores: List[float] = []
    for i in range(0, len(passages), batch_size):
        batch_pairs = [(query, passages[j][1]) for j in range(i, min(i+batch_size, len(passages)))]
        enc = tok.batch_encode_plus(
            batch_pairs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(dev) for k, v in enc.items()}
        logits = mdl(**enc).logits.squeeze(-1)  # [B]
        if logits.ndim == 0:  # single example edge-case
            logits = logits.unsqueeze(0)
        all_scores.extend(logits.detach().cpu().tolist())

    scored = [(passages[i][0], float(all_scores[i])) for i in range(len(passages))]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_m]

# reranker that reads query and passage and reorders the top candidates
# based using a tiny Transformer