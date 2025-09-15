import pytest
from copilot.qa.answering import answer

# Helper to keep tests tidy
def run_answer(q, passages, k_ctx=None):
    # k_ctx is ignored here (handled by the harness); answer() just takes passages list.
    pred, conf = answer(q, passages, None)
    return pred, conf


def test_duration_extraction_30_days():
    # Regex: RE_DURATION should match "30 days"
    passage = "The standard warranty period is 30 days. The warranty starts on the delivery date."
    pred, conf = run_answer("What is the warranty period?", [passage])
    assert pred == "30 days"          # short span, normalized/lowercased
    assert conf >= 0.85               # regex/phrase path should yield high confidence


def test_business_days_range_extraction():
    # Regex: RE_RANGE should match "5-10 business days"
    passage = "After your item is inspected, approved refunds are issued within 5-10 business days."
    pred, conf = run_answer("How long do refunds take after inspection?", [passage])
    assert pred == "5-10 business days"
    assert conf >= 0.85


def test_percent_fee_phrase_extraction():
    # Regex: RESTOCK_FEE via RE_PERCENT + context: "15% restocking fee"
    passage = "Opened items may be accepted at our discretion and are subject to a 15% restocking fee."
    pred, conf = run_answer("Do opened items incur a fee?", [passage])
    assert pred == "15% restocking fee"
    assert conf >= 0.85


def test_proof_of_purchase_phrase_extraction():
    # Phrase: "proof of purchase"
    passage = "To process a return, you must provide proof of purchase such as a receipt or order number."
    pred, conf = run_answer("What is required to process a return?", [passage])
    assert pred == "proof of purchase"
    assert conf >= 0.85


def test_shipping_canonicalization():
    # Canonicalization: "customer pays return shipping" => "customer is responsible for return shipping"
    passage = "Unless otherwise required by law, customer pays return shipping."
    pred, conf = run_answer("Who pays shipping for warranty returns?", [passage])
    assert pred == "customer is responsible for return shipping"
    assert 0.0 <= conf <= 1.0


def test_non_transferable_canonicalization():
    # Canonicalization: "non transferable" => "non-transferable"
    passage = "The warranty is non transferable and applies only to the original purchaser."
    pred, conf = run_answer("Is the warranty transferable?", [passage])
    assert pred == "non-transferable"
    assert 0.0 <= conf <= 1.0


def test_multi_passage_best_sentence_selection():
    # Ensure the answerer can pick the correct sentence from among multiple passages.
    p1 = "This paragraph is unrelated to the question and talks about apples and bananas."
    p2 = "Proof of purchase is required to return a product."
    pred, conf = run_answer("What do I need to show to return a product?", [p1, p2])
    assert pred == "proof of purchase"
    assert conf >= 0.85


def test_lowercasing_and_punctuation_stripping():
    # Postprocess: lower-case + strip trailing punctuation
    passage = "Refunds are issued within 30 days."
    pred, conf = run_answer("When will I receive my refund?", [passage])
    assert pred.endswith("days")
    assert pred == pred.lower()
    assert pred[-1].isalpha() or pred[-1].isdigit()


def test_fallback_returns_reasonable_span_when_no_regex_or_phrase_hits():
    # No regex hits, no listed domain phrase: answerer should fall back to a short window or short sentence.
    passage = "This warranty covers defects in materials and workmanship under normal use."
    pred, conf = run_answer("What does the warranty cover?", [passage])
    # We don't enforce the exact fallback text (implementation-dependent), but it should be non-empty and short-ish.
    assert isinstance(pred, str) and len(pred) > 0
    assert len(pred.split()) <= 10
    assert 0.0 <= conf <= 1.0
