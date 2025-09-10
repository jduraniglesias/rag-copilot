# rag-copilot
rag copilot ai project

# BM25 vs TF-IDF

TF-IDF:
TF measures how many times a term appears in a document
IDF measures how unique a term is in all the documents

The issue with TF-IDF is that there is no penalty or measure to counter longer documents scoring higher than shorter documents that better match the query. 

This is where BM25 comes in with the goal to improve TF so it has diminishing returns the more terms added and also adds penalty for longer documents

# BM25

Formula:
IDF(t) = ln( ((N - n_t + 0.5) / (n_t + 0.5)) + 1 )

BM25: Σ over terms t in q [
  IDF(t) * (tf(t,d) * (k1 + 1)) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
]

Formula Breakdown:
IDF is the same original scoring for the uniqueness of the query in all the documents

TF(t, d) in both numerator and denominator used to solve diminishing returns issue

theta = |d| / avgdl, is used to penalize longer documents where |d| is the length of the document so if longer than the avg document length, the larger theta gets which penalizes overall bm25 score