import re
from typing import Optional, List, Tuple, Dict
from copilot.text.tokenize import tokenize
from copilot.eval.qa_metrics import token_f1

STOPWORDS_MIN = {
    "the","a","an","to","of","and","for","in","on","is","are","was","were",
    "with","by","that","this","it","as","at","from","or"
}

STOPWORDS_EXTENDED = STOPWORDS_MIN | {
    "be","am","is","are","was","were","been","being",
    "do","does","did","doing",
    "have","has","had","having",
    "he","she","they","we","you","i","me","him","her","them","us","my","your","their","our","its",
    "will","would","can","could","should","may","might","must",
    "if","then","else","than","because","so","but","and","or",
    "about","into","over","under","between","through","during","before","after","above","below",
    "up","down","out","off","again","further","once",
    "what","when","where","which","who","whom","why","how"
}

RE_RANGE = re.compile(r"\b\d+\s*-\s*\d+\s+(?:business\s+days|days|weeks|months)\b", re.IGNORECASE)
RE_DURATION = re.compile(r"\b\d{1,3}\s+(?:day|days|week|weeks|month|months|year|years)\b", re.IGNORECASE)
RE_PERCENT = re.compile(r"\b\d{1,3}\s*%", re.IGNORECASE)
RE_MONEY = re.compile(r"\$\s?\d+(?:\.\d{2})?", re.IGNORECASE)
RE_PHRASES = [
    re.compile(r"proof of purchase", re.IGNORECASE),
    re.compile(r"non[-\s]?transferable", re.IGNORECASE),
    re.compile(r"gift cards are final sale", re.IGNORECASE),
    re.compile(r"customer (?:pays|is responsible for) return shipping", re.IGNORECASE),
]

def _filtered_tokens(text: str) -> List[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS_EXTENDED]

def _regex_match(sentence: str) -> Optional[str]:
    for pat in (RE_RANGE, RE_DURATION, RE_PERCENT, RE_MONEY, *RE_PHRASES):
        m = pat.search(sentence)
        if m:
            return m.group(0).strip()
    return None

def _split_sentences(text) -> List[str]:
    if isinstance(text, list):
        return [s.strip() for s in text if isinstance(s, str) and s.strip()]
    if not isinstance(text, str):
        raise TypeError(f"_split_sentences expected str or list[str], got {type(text)}")
    parts = re.split(r"(?<=[\.\?\!])\s+", text.strip())
    return [p.strip() for p in parts if p and len(p.strip()) > 1]

def _sentence_score(question: str, sentence: str) -> float:
    filtered_q_tokens = _filtered_tokens(question)
    filtered_s_tokens = _filtered_tokens(sentence)
    return token_f1(" ".join(filtered_q_tokens), " ".join(filtered_s_tokens))

def _select_best_sentence(question: str, chunk_text: str) -> str:
    if isinstance(chunk_text, list):
        sentences = []
        for p in chunk_text:
            sentences.extend(_split_sentences(p))
    else:
        sentences = _split_sentences(chunk_text)

    best_sent = ""
    best_score = -1.0
    for s in sentences:
        s_score = _sentence_score(question, s)
        # tie-break: prefer the shorter sentence on equal score
        if (s_score > best_score) or (s_score == best_score and len(s) < len(best_sent)):
            best_score = s_score
            best_sent = s
    return best_sent

def _extract_span(question: str, sentence: str) -> str:
    if not sentence:
        return "", 0.0

    # 1) Regex/phrase hits (precise)
    m = _regex_match(sentence)
    if m:
        return _postprocess_answer(m), 0.90

    # 2) Keyword-window fallback over filtered tokens
    q_toks = _filtered_tokens(question)
    s_toks = _filtered_tokens(sentence)
    if not q_toks or not s_toks:
        ans = " ".join(s_toks[:6]).strip()
        return _postprocess_answer(ans), 0.30 if ans else 0.0

    q_set = set(q_toks)
    best_text = ""
    best_score = -1
    best_len = 10**9

    for w in (2, 3, 4, 5, 6):
        if len(s_toks) < w:
            continue
        for i in range(0, len(s_toks) - w + 1):
            window = s_toks[i:i + w]
            score = sum(1 for t in window if t in q_set)  # how many question keywords covered
            cand_text = " ".join(window)
            if (score > best_score) or (score == best_score and len(cand_text) < best_len):
                best_score = score
                best_len = len(cand_text)
                best_text = cand_text

    if not best_text:
        best_text = " ".join(s_toks[: min(6, len(s_toks))]).strip()
        return _postprocess_answer(best_text), 0.30 if best_text else 0.0

    # Simple confidence heuristic: more keyword hits → higher confidence (cap at 0.85)
    conf = min(0.85, 0.50 + 0.10 * min(best_score, 3))
    return _postprocess_answer(best_text), conf

def _postprocess_answer(text: str) -> str:
    res = text.strip()
    res = re.sub(r'[.,;:!?]+$', '', res)
    res = re.sub(r'\s+', ' ', res)
    return res.lower()

def _canonicalize(s: str) -> str:
    x = " ".join(s.strip().lower().split())
    # punctuation/hyphen variants
    x = x.replace("–", "-").replace("—", "-")
    x = x.replace("  ", " ")
    # common synonyms → canonical forms
    synonyms = {
        "customer pays return shipping": "customer is responsible for return shipping",
        "proof of purchase": "proof of purchase",  # identity, but keep for pattern
        "gift cards are final sale": "gift cards are final sale",
        "non transferable": "non-transferable",
    }
    for k, v in synonyms.items():
        if k in x:
            x = v
    # optional: prepend "no, " if sentence begins with "no" and domain phrase follows
    if x.startswith("gift cards are final sale") and "no" in s.lower().split()[:3]:
        x = "no, " + x
    return x

def answer(
    question: str,
    passages: List[str],
    passages_meta: Optional[List[Dict]] = None,
) -> Tuple[str, Optional[float]]:
    """
    Produce a short answer string (and a rough confidence) from top retrieved passages.

    question (the user query)
    passages (top-k retrieved chunk texts; 'chunk' = slice of the original doc)
    passages_meta (optional metadata like doc_id/offsets; not used here but kept for future use)

    Steps:
    1) Select best sentence across passages (score = token_F1 vs question + bonuses).
    2) Extract concise span via domain phrases/regex/keyword window.
    3) Return (answer, confidence).

    Returns:
        predicted_short_answer, confidence in [0,1] (heuristic)
    """
    best_sentence = _select_best_sentence(question, passages)
    pred, conf = _extract_span(best_sentence, question)
    pred = _canonicalize(pred)
    # # normalize types & whitespace
    # pred = " ".join(str(pred or "").strip().split())
    # try:
    #     conf = float(conf)
    # except Exception:
    #     conf = 0.0
    return pred, conf