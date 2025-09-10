# tests/test_metrics.py
from copilot.eval.qa_metrics import exact_match, token_f1

def test_em_basic():
    assert exact_match("30 days", "30 days") == 1
    assert exact_match("The warranty is 30 days.", "30 days") == 0  # not exact

def test_f1_partial_credit():
    assert token_f1("The warranty is 30 days.", "30 days") > 0.0
    assert token_f1("", "") == 1.0
    assert token_f1("", "abc") == 0.0
