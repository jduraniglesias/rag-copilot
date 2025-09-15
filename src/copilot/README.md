0) Run before to enable CLI:
$env:PYTHONPATH = (Resolve-Path .\src)

1) Index your docs (build BM25 over data/documents)
python -m copilot.cli index --documents data/documents --size 600 --overlap 120

2) Ask an ad-hoc question (see top-k chunks with citations)
python -m copilot.cli ask "What is the warranty period?" --k 5 --documents data/documents

3) Evaluate (baseline = whole top chunk as the “answer”)
python -m copilot.cli evaluate --gold data/qa_gold.jsonl --k 3 --qa baseline --documents data/documents

4) Evaluate using your short-answer extractor
python -m copilot.cli evaluate --gold data/qa_gold.jsonl --k 3 --qa answerer