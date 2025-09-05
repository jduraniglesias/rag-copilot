from copilot.index.inverted import build_index

def _mk_chunk(txt, doc_id="d"):
    return {"text": txt, "meta": {"doc_id": doc_id, "char_start": 0, "char_end": len(txt)}}

def test_build_index_shapes_and_stats():
    chunks = [
        _mk_chunk("warranty coverage coverage"),
        _mk_chunk("returns policy"),
        _mk_chunk("warranty policy applies")
    ]
    idx = build_index(chunks)
    assert idx["N"] == 3
    assert len(idx["doc_len"]) == 3
    assert idx["avgdl"] > 0
    assert "postings" in idx and isinstance(idx["postings"], dict)

def test_postings_have_correct_tf_and_unique_entries():
    chunks = [
        _mk_chunk("a a a b"),
        _mk_chunk("b c"),
        _mk_chunk("a c c")
    ]
    idx = build_index(chunks)
    postings = idx["postings"]
    # term 'a' appears in chunks 0 (tf=3) and 2 (tf=1)
    a_list = sorted(postings["a"])
    assert a_list == [(0, 3), (2, 1)]
    # term 'b' appears in chunks 0 (tf=1) and 1 (tf=1)
    b_list = sorted(postings["b"])
    assert b_list == [(0, 1), (1, 1)]
    # term not present shouldn't exist
    assert "zzz" not in postings

def test_doc_len_matches_tokenization():
    chunks = [
        _mk_chunk("Hello, world!"),
        _mk_chunk("warranty coverage")
    ]
    idx = build_index(chunks)
    # Using our simple tokenizer: ["hello","world"] length 2, ["warranty","coverage"] length 2
    assert idx["doc_len"] == [2, 2]
