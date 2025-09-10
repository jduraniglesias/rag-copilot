from typing import List
from copilot.text.tokenize import tokenize
from collections import Counter

# lowercase and use tokenize to remove punctuation
def normalize_answer(s: str) -> str:
    return " ".join(tokenize(s))

def exact_match(pred, gold) -> int:
    if normalize_answer(pred) == normalize_answer(gold):
        return 1
    return 0

def token_f1(pred, gold) -> float:
    pred_tokens = tokenize(pred)
    gold_tokens = tokenize(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    
    # keeps min of each token from each counter
    overlap = sum((pred_counter & gold_counter).values())
    precision = overlap / sum(pred_counter.values())
    recall = overlap / sum(gold_counter.values())
    return 2 * precision * recall / (precision + recall)

