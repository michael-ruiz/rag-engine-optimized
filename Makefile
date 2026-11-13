PYTHON ?= python
PORT ?= 8080# not 8000: that is the default port of a local `vllm serve` (vllm-metal)
MODEL ?= Qwen/Qwen3.5-0.8B
FILTER ?= layers.15.
PROMPT ?= 3x + 5 = 20. Solve for x.
DOCS ?= data/sample_corpus

.PHONY: install-mac install-cuda test inspect ingest serve

install-mac:
	$(PYTHON) -m pip install -r requirements-mac.txt
	$(PYTHON) -m pip install --no-deps -e .

install-cuda:
	$(PYTHON) -m pip install -r requirements-cuda.txt
	$(PYTHON) -m pip install --no-deps -e .

test:
	$(PYTHON) -m pytest -q

inspect:
	$(PYTHON) scripts/inspect_weights.py --model "$(MODEL)" --filter "$(FILTER)" --prompt "$(PROMPT)"

# stop `make serve` first: embedded Qdrant allows one process per folder
ingest:
	$(PYTHON) scripts/ingest_docs.py $(DOCS)

# PATCH_RAG_MODEL / PATCH_RAG_BACKEND env vars pick the checkpoint and backend
serve:
	$(PYTHON) -m uvicorn src.serving.app:app --host 127.0.0.1 --port $(PORT)
