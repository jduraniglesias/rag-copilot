from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from functools import lru_cache

MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

# Uses model to capture phrases/synonyms of words with vectors
# Embeddings used to turn text to vectors
# MiniLM used to learn paired texts so related pairs have high cosine similarity
# Better since BM25 is lexical and measures exact words/phrases

# Loads model once and caches it
@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = 512
    return model

def encode_texts(texts: List[str]) -> np.ndarray:
    # Return L2-normalized embeddings for cosine via inner-product
    model = get_model()
    vecs = model.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
    return vecs

def encode_query(q: str) -> np.ndarray:
    return encode_texts([q])
