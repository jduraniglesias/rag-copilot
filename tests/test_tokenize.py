import pytest
from copilot.text.tokenize import tokenize, normalize

# Cases that don't depend on accents
BASIC_CASES = [
    ("Hello, world!", ["hello", "world"]),
    ("  Hello   \n  world\t", ["hello", "world"]),
    ("Warranty: 2-year (parts & labor).", ["warranty", "2", "year", "parts", "labor"]),
    ("co-operate can't re-enter", ["co", "operate", "can", "t", "re", "enter"]),
    ("", []),
    ("   ", []),
]

def test_tokenize_various_preserve_accents_default():
    # Default behavior should preserve accents (removeAccents=False)
    for text, expected in BASIC_CASES:
        assert tokenize(text) == expected

def test_tokenize_spanish_preserve_vs_strip():
    # This case checks both behaviors on an accented string
    text = "Información útil — versión 2.0!"
    expected_keep = ["información", "útil", "versión", "2", "0"]
    expected_strip = ["informacion", "util", "version", "2", "0"]

    # Default path: keep accents
    assert tokenize(text) == expected_keep
    # Explicit strip: remove accents
    assert normalize(text, removeAccents=True).split() == expected_strip

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
