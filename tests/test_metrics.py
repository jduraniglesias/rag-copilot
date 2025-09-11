# tests/test_metrics.py
from copilot.eval.qa_metrics import exact_match, token_f1
from copilot.eval.rank_metrics import precision_at_k, mrr_at_k

def test_em_basic():
    assert exact_match("30 days", "30 days") == 1
    assert exact_match("The warranty is 30 days.", "30 days") == 0  # not exact

def test_f1_partial_credit():
    assert token_f1("The warranty is 30 days.", "30 days") > 0.0
    assert token_f1("", "") == 1.0
    assert token_f1("", "abc") == 0.0

def test_precision_at_k_basic():
    labels = [1.0, 0.0, 1.0]
    assert abs(precision_at_k(labels, 3) - (2/3)) < 1e-9

def test_precision_at_k_handles_short_list_and_zero_k():
    labels = [1.0]
    assert precision_at_k(labels, 3) == (1/3)  # pad implicit zeros
    assert precision_at_k(labels, 0) == 0.0

def test_mrr_at_k_first_relevant_positions():
    assert mrr_at_k([1.0, 0.0, 0.0], 3) == 1.0       # first position
    assert abs(mrr_at_k([0.0, 1.0, 0.0], 3) - 0.5) < 1e-9  # second position
    assert abs(mrr_at_k([0.0, 0.0, 1.0], 3) - (1/3)) < 1e-9
    assert mrr_at_k([0.0, 0.0, 0.0], 3) == 0.0
    assert mrr_at_k([], 3) == 0.0