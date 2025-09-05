import math
import pytest
from copilot.text.chunk import chunk_text

def _expected_chunk_count(n_chars: int, size: int, overlap: int) -> int:
    # If text fits in one chunk
    if n_chars <= size:
        return 1
    step = size - overlap
    # first chunk covers `size`; remaining length is n - size
    remaining = max(0, n_chars - size)
    return 1 + math.ceil(remaining / step)

def test_short_text_single_chunk():
    text = "hello world"
    chunks = chunk_text(text, doc_id="doc", size=50, overlap=10)
    assert len(chunks) == 1
    assert chunks[0]["text"] == text
    assert chunks[0]["meta"]["doc_id"] == "doc"
    assert chunks[0]["meta"]["char_start"] == 0
    assert chunks[0]["meta"]["char_end"] == len(text)

def test_exact_size_one_chunk():
    text = "a" * 600
    chunks = chunk_text(text, doc_id="doc", size=600, overlap=120)
    assert len(chunks) == 1
    c = chunks[0]
    assert c["meta"]["char_start"] == 0 and c["meta"]["char_end"] == 600
    assert len(c["text"]) == 600

def test_overlap_and_bounds():
    n = 1000
    size, overlap = 300, 100
    text = "x" * n
    chunks = chunk_text(text, doc_id="doc", size=size, overlap=overlap)
    # 1) count matches formula
    assert len(chunks) == _expected_chunk_count(n, size, overlap)
    # 2) boundaries make sense
    assert chunks[0]["meta"]["char_start"] == 0
    for i in range(len(chunks)):
        c = chunks[i]
        start, end = c["meta"]["char_start"], c["meta"]["char_end"]
        assert 0 <= start < end <= n
        # size is exact except possibly the last chunk
        if i < len(chunks) - 1:
            assert end - start == size
        else:
            assert 1 <= end - start <= size
        # sliding window: next_start = this_end - overlap
        if i < len(chunks) - 1:
            next_start = chunks[i + 1]["meta"]["char_start"]
            assert next_start == end - overlap
    # 3) last chunk ends at n
    assert chunks[-1]["meta"]["char_end"] == n

def test_invalid_params_raise():
    text = "abc"
    with pytest.raises(AssertionError):
        chunk_text(text, doc_id="d", size=0, overlap=0)  # size must be > 0
    with pytest.raises(AssertionError):
        chunk_text(text, doc_id="d", size=100, overlap=100)  # overlap < size
    with pytest.raises(AssertionError):
        chunk_text(text, doc_id="d", size=100, overlap=150)  # overlap < size
