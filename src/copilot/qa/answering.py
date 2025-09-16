# src/copilot/qa/answering.py
from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import re

from copilot.text.tokenize import tokenize
from copilot.eval.qa_metrics import token_f1

# ---- Inline definitions (plain English) -------------------------------------
# sentence selector = pick the best sentence among passages
# span extraction   = extract a short substring (e.g., "30 days") from that sentence
# canonicalization  = normalize equivalent phrases to a single form

# ---- Domain phrases we care about (verbatim matches) ------------------------
# (These are short answers you want to return as-is, before heuristics.)
_DOMAIN_PHRASES = [
    "proof of purchase",
    "delivery date",
    "non-transferable",
    "non transferable",  # will canonicalize -> "non-transferable"
    "gift cards are final sale",
    "customer pays return shipping",               # will canonicalize -> "customer is responsible for return shipping"
    "customer is responsible for return shipping",
    "defects in materials and workmanship",
    "exchanges within 30 days",
]

# ---- Canonicalization rules (map variants -> one expected form) -------------
_CANON_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bnon[-\s]?transferable\b", re.IGNORECASE), "non-transferable"),
    (re.compile(r"\bcustomer\s+(?:pays|pays\s+for|is\s+responsible\s+for)\s+return\s+shipping\b", re.IGNORECASE),
     "customer is responsible for return shipping"),
]

def _canonicalize(ans: str) -> str:
    s = ans.strip()
    for pat, repl in _CANON_RULES:
        if pat.search(s):
            return repl
    return s

# ---- Regexes for numeric/short answers -------------------------------------
# durations like "30 days", "5-10 business days", "12 months"
_RE_DURATION = re.compile(
    r"\b\d{1,3}(?:-\d{1,3})?\s+(?:business\s+)?(?:day|days|month|months|year|years)\b",
    re.IGNORECASE,
)
# percents like "15%"
_RE_PERCENT = re.compile(r"\b\d{1,3}(?:\.\d+)?\s*%\b", re.IGNORECASE)
# "15% restocking fee"
_RE_RESTOCK_FEE = re.compile(r"\b\d{1,3}(?:\.\d+)?\s*%\s+restocking\s+fee\b", re.IGNORECASE)

_STOPWORDS = {
    "the","a","an","and","or","of","to","in","for","on","with","by","is","are","be",
    "this","that","it","as","at","from","within","under","without","your","you","our",
    "we","will","may","must","can","not","do","does","did","how","what","when","who"
}

# ---- Sentence splitting -----------------------------------------------------
def _split_sentences(text: str) -> List[str]:
    # naive sentence splitter good enough for policy text
    parts = re.split(r"(?<=[\.\?\!])\s+", text.strip())
    return [p.strip() for p in parts if p and len(p.strip()) > 1]

def _content_tokens(text: str) -> List[str]:
    return [t for t in tokenize(text) if t not in _STOPWORDS]

# ---- Scoring helpers --------------------------------------------------------
def _has_domain_phrase(s_lc: str) -> Optional[str]:
    for ph in _DOMAIN_PHRASES:
        if ph in s_lc:
            return ph
    return None

def _find_regex_span(sentence: str) -> Optional[str]:
    # more specific first
    m = _RE_RESTOCK_FEE.search(sentence)
    if m:
        return m.group(0).strip()
    m = _RE_DURATION.search(sentence)
    if m:
        return m.group(0).strip()
    m = _RE_PERCENT.search(sentence)
    if m:
        return m.group(0).strip()
    return None

def _sentence_score(question: str, sentence: str) -> float:
    """Score = token-F1 overlap + bonuses when we detect strong answer cues."""
    base = token_f1(sentence, question)
    s_lc = sentence.lower()
    bonus = 0.0
    if _has_domain_phrase(s_lc):
        bonus += 0.50  # strong bias toward sentences with known phrases
    if _find_regex_span(sentence):
        bonus += 0.40  # and numeric/fee patterns
    return base + bonus

def _pick_best_sentence(question: str, passages: List[str]) -> str:
    best, best_score = "", -1.0
    for p in passages:
        for s in _split_sentences(p):
            sc = _sentence_score(question, s)
            if sc > best_score:
                best, best_score = s, sc
    return best

# ---- Span extraction --------------------------------------------------------
def _phrase_first(sentence_lc: str) -> Optional[str]:
    ph = _has_domain_phrase(sentence_lc)
    if ph:
        return ph
    return None

def _shortest_keyword_window(sentence: str, question: str) -> Optional[str]:
    # Try to pick a concise substring that covers 2–3 content words from the question
    s_lc = sentence.lower()
    q_keywords = []
    seen = set()
    for t in _content_tokens(question):
        if t not in seen:
            q_keywords.append(t)
            seen.add(t)

    hits: List[Tuple[int, int]] = []
    for kw in q_keywords:
        pos = s_lc.find(kw)
        if pos != -1:
            hits.append((pos, pos + len(kw)))
    if not hits:
        return None

    # window over up to first 3 hits
    hits.sort(key=lambda x: x[0])
    hits = hits[:3]
    start = min(h[0] for h in hits)
    end = max(h[1] for h in hits)
    span = sentence[max(0, start):min(len(sentence), end)].strip(" ,.;:!?")
    return span if span else None

def _extract_span(sentence: str, question: str) -> Tuple[str, float]:
    if not sentence.strip():
        return "", 0.0
    s = sentence.strip()
    s_lc = s.lower()

    # 1) exact domain phrases (then canonicalize)
    ph = _phrase_first(s_lc)
    if ph:
        return _canonicalize(ph), 0.90

    # 2) regex-based numeric spans (durations/percent/fee)
    rgx = _find_regex_span(s)
    if rgx:
        return _canonicalize(rgx), 0.85

    # 3) fallback: shortest keyword window
    win = _shortest_keyword_window(s, question)
    if win:
        # slightly lower confidence; still canonicalize in case window matches a rule
        return _canonicalize(win), min(0.80, 0.50 + 0.30 * token_f1(win, question))

    # 4) last resort: trimmed sentence (shortened)
    short = s
    if len(short) > 140:
        short = short[:140].rstrip() + "…"
    return _canonicalize(short), min(0.60, 0.30 + 0.30 * token_f1(short, question))

# ---- Public API -------------------------------------------------------------
def answer(
    question: str,
    passages: List[str],
    passages_meta: Optional[List[Dict]] = None,
) -> Tuple[str, Optional[float]]:
    """
    Produce a short answer string (and a rough confidence) from top retrieved passages.

    Improvements over the basic version:
      - We weight sentences by the retrieval strength of their parent passage.
        * retrieval_rank (1 = strongest)  -> w_rank = 1 / rank
        * retrieval_score (per-query)     -> min-max normalized to [0,1]
        * passage_weight = 0.5*w_rank + 0.5*w_score (w_score=0 if missing)
      - Sentence score = (token-F1(question, sentence) + small pattern bonuses) * passage_weight
      - Then we extract a concise span from the best sentence and normalize it.
    """
    if not passages:
        return "", 0.0

    # --- compute per-passage weights from meta (rank + score) ---
    def _minmax(xs: List[float]) -> List[float]:
        if not xs:
            return []
        lo = min(xs); hi = max(xs)
        rng = (hi - lo) or 1e-9
        return [(x - lo) / rng for x in xs]

    if passages_meta and len(passages_meta) == len(passages):
        ranks  = [m.get("retrieval_rank") for m in passages_meta]
        scores = [m.get("retrieval_score") for m in passages_meta]

        w_rank = [1.0 / float(r) if isinstance(r, (int, float)) and r > 0 else 1.0 for r in ranks]
        # normalize scores if present; else zeros
        score_vals = [float(s) for s in scores if isinstance(s, (int, float))]
        if score_vals:
            norm_map = {i: v for i, v in enumerate(_minmax([float(s) if isinstance(s, (int, float)) else 0.0
                                                            for s in scores]))}
            w_score = [norm_map[i] for i in range(len(passages))]
        else:
            w_score = [0.0] * len(passages)

        passage_weights = [0.5 * wr + 0.5 * ws for wr, ws in zip(w_rank, w_score)]
    else:
        passage_weights = [1.0] * len(passages)

    # --- split passages into sentences and carry the passage weight to each sentence ---
    splitter = re.compile(r'(?<=[.!?])\s+')
    sentences: List[str] = []
    sent_weights: List[float] = []
    for p, w in zip(passages, passage_weights):
        for s in splitter.split(p.strip()):
            s = s.strip()
            if s:
                sentences.append(s)
                sent_weights.append(w)

    if not sentences:
        # fallback to entire first passage
        best_sentence = passages[0]
    else:
        # --- score sentences: token-F1 + pattern bonuses, then weight by passage strength ---
        from copilot.eval.qa_metrics import token_f1  # lazy import to avoid cycles

        best_idx, best_score = 0, -1.0
        for i, s in enumerate(sentences):
            base = token_f1(s, question)

            # light domain bonuses to help policy Q&A
            bonus = 0.0
            if re.search(r'\b\d{1,3}\s*(?:business\s+days|days|weeks|months)\b', s, re.I):
                bonus += 0.12
            if re.search(r'\b\d{1,2}%\b', s):
                bonus += 0.08
            if re.search(r'\$\s*\d', s):
                bonus += 0.06
            if re.search(r'\b(exchang\w+|return\w+|refund\w+)\b', s, re.I):
                bonus += 0.05
            if re.search(r'\bwithin\b', s, re.I):
                bonus += 0.03
            if re.search(r'\b(delivery|pickup)\b', s, re.I):
                bonus += 0.02

            score = (base + bonus) * sent_weights[i]
            if score > best_score:
                best_score, best_idx = score, i

        best_sentence = sentences[best_idx]

    # --- extract concise span from best sentence (uses your existing extractor) ---
    pred, conf = _extract_span(best_sentence, question)

    # --- normalize whitespace + lowercase, strip trailing punctuation ---
    pred = " ".join(pred.strip().split()).lower().rstrip(".,;:!?")

    # lightly modulate confidence by the chosen sentence's passage weight (if available)
    try:
        w = sent_weights[best_idx] if sentences else passage_weights[0]
        w = max(0.0, min(1.0, float(w)))
        conf = (float(conf) * (0.7 + 0.3 * w)) if conf is not None else w
    except Exception:
        pass

    def _is_supported(p: str, ctx: list[str]) -> bool:
        p = p.strip().lower()
        if not p:
            return False
        return any(p in c.lower() for c in ctx)

    if not _is_supported(pred, passages):
        # penalize confidence hard (or abstain)
        conf = (conf * 0.2) if conf is not None else 0.2

    return pred, conf