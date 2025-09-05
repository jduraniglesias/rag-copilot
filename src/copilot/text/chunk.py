from typing import List, Dict

def chunk_text(
    text: str,
    doc_id: str,
    size: int = 600,
    overlap: int = 120,
) -> List[Dict]:
    """
    Split `text` into overlapping character-based chunks.

    Args:
        text: the original (unmodified) document text. Keep it raw so char offsets stay accurate.
        doc_id: identifier for the source (filename).
        size: max characters per chunk (window length).
        overlap: amount characters should overlap to prevent cutting off facts.

    Returns:
        List of dicts (hash) with:
          - "text": the chunk text
          - "meta": {"doc_id", "char_start", "char_end"}
    """
    assert size > 0, "size must be > 0"
    assert 0 <= overlap < size, "overlap must be >=0 and < size"

    chunks: List[Dict] = []
    n = len(text)
    if n == 0:
        return [{
            "text": "",
            "meta": {"doc_id": doc_id, "char_start": 0, "char_end": 0}
        }]

    start = 0
    while start < n:
        end = min(start + size, n)
        chunk = text[start:end]
        chunks.append({
            "text": chunk,
            "meta": {
                "doc_id": doc_id,
                "char_start": start,
                "char_end": end
            }
        })
        if end == n:
            break
        start = end - overlap  # slide window back by overlap
    return chunks

def chunk_file(path: str, size: int = 600, overlap: int = 120) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    return chunk_text(txt, doc_id=path, size=size, overlap=overlap)
