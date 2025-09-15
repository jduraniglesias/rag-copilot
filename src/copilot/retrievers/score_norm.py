from typing import Dict

# normalizes scoring for BM25 and Dense scores

def minmax(scores: Dict[int, float]) -> Dict[int, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1e-9
    return {cid: (s - lo) / rng for cid, s in scores.items()}
