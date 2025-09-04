import pytest
from copilot.text.tokenize import tokenize, normalize

@pytest.mark.parametrize("text,expected", [
    ("Hello, world!", ["hello", "world"]),
    ("  Hello   \n  world\t", ["hello", "world"]),
    ("Warranty: 2-year (parts & labor).", ["warranty", "2", "year", "parts", "labor"]),
    ("co-operate can't re-enter", ["co", "operate", "can", "t", "re", "enter"]),
    ("Información útil — versión 2.0!", ["información", "útil", "versión", "2", "0"]),
    ("", []),
    ("   ", []),
])
def test_tokenize_various(text, expected):
    assert tokenize(text) == expected

def test_normalize_idempotent():
    s = "Hello—World!!  Co-operate, please…"
    once = normalize(s)
    twice = normalize(once)
    assert once == twice

def test_tokenize_is_lowercased():
    assert tokenize("MiXeD CaSe") == ["mixed", "case"]

def test_only_alnum_and_spaces_remain_after_normalize():
    s = "Price: $19.99 & tax% included."
    norm = normalize(s)
    # After normalize, there should be no punctuation like $, %, :, .
    for ch in norm:
        assert ch.isalnum() or ch.isspace()

def test_multiple_spaces_collapse_on_split():
    s = "one,,,   two\t\tthree\n\nfour"
    # commas become spaces; split should drop empties
    assert tokenize(s) == ["one", "two", "three", "four"]
