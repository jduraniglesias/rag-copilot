from copilot.qa.answering import (
    split_sentences, sentence_score, select_best_sentence,
    extract_span, predict_answer_from_chunk
)

# Minimal tokenizer assumptions from your earlier code:
# lowercase, punctuation -> space, split on whitespace

def test_split_sentences_basic():
    txt = "The warranty period is 30 days. Packages without an RMA may be refused! When does it start? On delivery."
    sents = split_sentences(txt)
    assert len(sents) >= 3
    assert any("30 days" in s for s in sents)
    assert any("refused" in s for s in sents)

def test_sentence_score_overlap_prefers_relevant_sentence():
    q = "What is the warranty period?"
    s1 = "The warranty period is 30 days."
    s2 = "Packages without an RMA may be refused."
    assert sentence_score(q, s1) > sentence_score(q, s2)

def test_extract_span_regex_duration():
    q = "What is the warranty period?"
    s = "The warranty period is 30 days."
    ans = extract_span(q, s)
    assert ans.lower() == "30 days"

def test_extract_span_regex_phrase():
    q = "What is required to process a return?"
    s = "To process a return, you must provide proof of purchase."
    ans = extract_span(q, s)
    assert "proof of purchase" in ans.lower()

def test_extract_span_percent():
    q = "Do opened items incur a fee?"
    s = "Opened items may be accepted at our discretion and are subject to a 15% restocking fee."
    ans = extract_span(q, s)
    print(f"PRINTED: {ans}")
    assert "15%" in ans

def test_predict_answer_from_chunk_end_to_end():
    q1 = "What is the warranty period?"
    chunk1 = "Limited Warranty. The standard warranty period is 30 days. Accidental damage is not covered."
    ans1 = predict_answer_from_chunk(q1, chunk1).lower()
    assert "30 days" in ans1

    q2 = "What is required to process a return?"
    chunk2 = "Return Window. You may return items within 30 days of delivery. To process a return, you must provide proof of purchase."
    ans2 = predict_answer_from_chunk(q2, chunk2).lower()
    assert "proof of purchase" in ans2
