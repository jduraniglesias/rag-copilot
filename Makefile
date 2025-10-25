.PHONY: api test eval
api:
\tuvicorn copilot.server.app:app --reload

test:
\tpytest -q

eval:
\tpython -m copilot.cli evaluate --retriever all --qa answerer --k 10 --k-ctx 3
