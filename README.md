# patch-rag

Low-latency enterprise RAG engine built around **Qwen3.5-0.8B** (modular enough for Llama-3 / Qwen-4B) that
suppresses generation loops and hallucinations with surgical weight-space edits instead of retraining.

## Layout
| Path | Purpose |
|---|---|
| `src/model/inspector.py` | `ModelInspector`: parameter scan (shape/dtype/MB), forward hooks → activation cache |
| `src/model/patcher.py` | `WeightPatcher`: rank-1 projection removal `W' = W - α·u uᵀ W`, column nulling, checkpoint export |
| `src/rag/ingestor.py` | `DocumentIngestor`: .md/.txt/.pdf/.py → section-aware recursive chunks with metadata |
| `src/rag/retriever.py` | `HybridRetriever`: embedded Qdrant dense + BM25, RRF fusion, cross-encoder rerank |
| `src/rag/pipeline.py` | `RAGPipeline`: grounded prompt, `<think>` split, citation validation, latencies |
| `src/serving/backends.py` | `VLLMBackend` (native CUDA or OpenAI-compatible HTTP) and `TransformersMPSBackend` |
| `src/serving/app.py` | FastAPI `/health`, `/generate`, `/v1/ingest`, `/v1/rag/query` (base vs patched A/B) |
| `src/utils/device.py` | `DeviceManager`: detects `cuda` → `mps` → `cpu`, picks dtype (bf16 on GPU, fp32 on CPU) |
| `scripts/` | CLI entry points: inspect, patch, ingest |
| `config/default_config.yaml` | Defaults for model, device, hook targets, RAG, serving |
| `data/sample_corpus/` | Small fictional corpus (Helio Labs) for smoke tests |
| `Makefile` | `install-mac`, `install-cuda`, `test`, `inspect`, `ingest`, `serve` |

## Install

### macOS (Apple Silicon, local dev)
```bash
make install-mac        # requirements-mac.txt: base + mlx, mlx-lm
make test
```
Optional native Metal vLLM server (`vllm serve`):
```bash
brew tap vllm-project/vllm-metal && brew install vllm-project/vllm-metal/vllm-metal
vllm serve Qwen/Qwen3.5-0.8B                          # OpenAI-compatible API on 127.0.0.1:8000
PATCH_RAG_BACKEND=vllm make serve                     # patch-rag API on :8080, proxies to it
```
Intel Macs: there are no PyPI torch wheels past 2.2 and no MLX wheels, so install torch/transformers
from conda-forge, then `pip install --no-deps -e .`. MLX lines in `requirements-mac.txt` are skipped automatically.

### Linux (NVIDIA CUDA, deploy/benchmark)
```bash
make install-cuda       # requirements-cuda.txt: base + vllm, autoawq
make test
```

## Serving backends
`serving.backend` in the config (or env `PATCH_RAG_BACKEND`) selects the strategy:

| Value | Behaviour |
|---|---|
| `auto` (default) | native vLLM on CUDA when installed, otherwise transformers on mps/cpu |
| `vllm` | in-process vLLM on Linux CUDA; elsewhere HTTP to `serving.vllm_url` (`http://127.0.0.1:8000/v1`) |
| `transformers` | in-process HF transformers on the detected device; serves patched `.safetensors` directly |

`auto` never picks the HTTP server: that server may be running a different (unpatched) model.
`PATCH_RAG_MODEL` selects the checkpoint (model id or patched directory).

## RAG pipeline
```
files ─► DocumentIngestor ─► chunks {text, source_file, chunk_index, section_header}
                                   │  bge-small-en-v1.5 (DeviceManager device)
                                   ▼
                    Qdrant (./data/qdrant_local, or rag.qdrant_url) + in-memory BM25
query ─► dense top-30 ┐
         BM25 top-30  ┴─► RRF (k=60) ─► top-20 ─► cross-encoder rerank ─► top-4
                                                                      │
       grounded system prompt + [source_file:chunk_index] tagged context ┘
                                   ▼
            backend.chat() (transformers / vLLM) ─► split <think> ─► validate citations
```
```bash
make ingest                                  # indexes data/sample_corpus (DOCS=... to change)
PATCH_RAG_PATCHED_MODEL=patched/ make serve
curl -s localhost:8080/v1/rag/query -H 'content-type: application/json' \
  -d '{"question": "How many PTO days roll over?", "use_patched_model": false}'
curl -s localhost:8080/v1/ingest -H 'content-type: application/json' -d '{"paths": ["data/sample_corpus"]}'
```
`/v1/rag/query` returns `answer`, `thinking_trace`, `citations`, `invalid_citations` (cited but never
retrieved, a hallucination signal), `contexts`, `retrieval_latency_ms`, `generation_latency_ms`, `backend`,
`model_variant`. `use_patched_model: true` lazily loads `serving.patched_model_path`
(env `PATCH_RAG_PATCHED_MODEL`) on first use. `enable_thinking: true` turns on Qwen3.5 reasoning (off by default
in its chat template); give it a generous `max_new_tokens` or the answer is cut off inside the trace.

- Embedded Qdrant locks its folder: run `make ingest` while the server is stopped, or use `POST /v1/ingest`.
- `/v1/ingest` only reads paths under `rag.ingest_root`.
- The BM25 index is held in memory and rebuilt from Qdrant at startup and after each ingest (fine for local corpora).

## Weight-patching workflow
```bash
# 1. inspect layer 15 and capture its MLP output
python scripts/inspect_weights.py --filter layers.15. --targets model.layers.15.mlp \
    --prompt "3x + 5 = 20. Solve for x." --save-activations acts.pt
# 2. project the mean activation direction out of that layer's down_proj
python scripts/apply_patch.py --param model.layers.15.mlp.down_proj.weight --rank1 \
    --direction acts.pt --direction-key model.layers.15.mlp --alpha 0.5 --out patched/
# 3. serve the patched checkpoint
PATCH_RAG_MODEL=patched/ make serve
```

## Notes
- Weights use the `nn.Linear` convention `W: [out, in]`; rank-1 directions live in the **output** space.
- Qwen3.5-0.8B ties `lm_head` to `embed_tokens`; the patcher refuses edits to tied weights unless `--allow-tied`.
- Qwen3.5 is hybrid: layers 3, 7, 11, 15, 19, 23 use `self_attn` (`o_proj`); the rest use `linear_attn` (`out_proj`).
- Qwen3.5 checkpoints are vision-language; loading via `AutoModelForCausalLM` and saving produces a
  **text-only** checkpoint (vision tower dropped).
- vLLM and AutoAWQ ship Linux CUDA wheels only; they are skipped on macOS via environment markers.
- `WeightPatcher` loads in bf16 on every device (even CPU) so patched checkpoints keep the base model's format.
- On MPS, if an op is unsupported, set `PYTORCH_ENABLE_MPS_FALLBACK=1` to run it on CPU.
