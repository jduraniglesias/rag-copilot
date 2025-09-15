import json, os
import numpy as np
import faiss

# Essentially just creating a database of chunk vectors
# Takes text from chunk and encodes it to L2 normalized vector

# Uses FAISS to search the vectors efficiently 
# for the exact/approximate similarity

class DenseIndex:
    def __init__(self, dim: int, index_path: str, meta_path: str):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = None
        self.chunk_ids = None  # np.ndarray[int], maps row -> chunk_id

    def build(self, chunk_texts, chunk_ids):
        from copilot.index import embeddings as E
        vecs = E.encode_texts(chunk_texts)  # [n, dim], already L2-normalized
        self.index = faiss.IndexFlatIP(self.dim)  # IP acts as cosine if normalized
        self.index.add(vecs)
        self.chunk_ids = np.asarray(chunk_ids, dtype=np.int64)
        self._save()

    def _save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w") as f:
            json.dump({"chunk_ids": self.chunk_ids.tolist(), "dim": self.dim}, f)

    def load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path) as f:
            meta = json.load(f)
        self.chunk_ids = np.asarray(meta["chunk_ids"], dtype=np.int64)
        self.dim = meta["dim"]
        return self

    def search(self, qvec, k):
        # qvec: shape [1, dim], already L2-normalized
        D, I = self.index.search(qvec, k)
        # D: similarities in [-1,1] (cosine). I: row indices
        rows = I[0]
        sims = D[0]
        out = []
        for r, s in zip(rows, sims):
            if r == -1: 
                continue
            out.append((int(self.chunk_ids[r]), float(s)))
        return out
