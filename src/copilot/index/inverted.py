from typing import List, Dict, Tuple
from copilot.text.tokenize import tokenize

# building inverted index for bm25 to use
# inverted index maps tokens (words) to a specific chunk where it's at.

def build_index(chunks: List[Dict]) -> Dict:
    postings: Dict[str, List[Tuple[int, int]]] = {}
    num_chunks = len(chunks)
    doc_lengths: List[int] = [0] * num_chunks
    for chunk_id, ch in enumerate(chunks):
        token_freq: Dict[str, int] = {}
        text = ch.get("text", "")
        meta = ch.get("meta", {})
        if not isinstance(meta, dict):
            meta = {}
        doc = meta.get("doc_id")
        start = meta.get("char_start")
        end = meta.get("char_end")

        tokens = tokenize(text)
        for token in tokens:
            token_freq[token] = token_freq.get(token, 0) + 1
        
        doc_lengths[chunk_id] = len(tokens)

        for token, tf in token_freq.items():
            postings.setdefault(token, []).append((chunk_id, tf))
    
    totaldl = 0
    for length in doc_lengths:
        totaldl += length
    avgdl = totaldl / num_chunks
    built_index: Dict = { 
        "postings": postings,
        "doc_len": doc_lengths,
        "avgdl": avgdl,
        "N": num_chunks,
        "chunks": chunks
    }
    return built_index

# Essentially im looping thru the chunks list,
# for each chunk tokenize the words,
# then loop thru the words and build the postings list
