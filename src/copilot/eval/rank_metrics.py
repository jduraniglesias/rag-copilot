import math
from typing import List

def dcg_at_k(labels: List[float], k: int) -> float:
    s = 0.0
    for i, rel in enumerate(labels[:k]):
        s += rel / math.log2(i+2)
    return s

def ndcg_at_k(labels: List[float], k: int) -> float:
    actual = dcg_at_k(labels, k)
    ideal = dcg_at_k(sorted(labels[:k], reverse=True), k)
    if ideal == 0.0:
        return 0.0
    else:
        return actual / ideal
    
def precision_at_k(labels: List[float], k: int) -> float:
    if k <= 0:
        return 0.0
    return float(sum(labels[:k])) / k

def mrr_at_k(labels: List[float], k: int) -> float:
    for i, rel in enumerate(labels[:k]):
        if rel > 0:
            return 1.0 / ((i+1))
    return 0.0