"""FastAPI service: raw generation, document ingestion and grounded RAG queries.

Run: PATCH_RAG_MODEL=<base model> PATCH_RAG_PATCHED_MODEL=<patched checkpoint dir> uvicorn src.serving.app:app
"""
from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src import load_config
from src.rag.ingestor import DocumentIngestor
from src.rag.pipeline import RAGAnswer, RAGPipeline
from src.rag.retriever import HybridRetriever
from src.serving.backends import DEFAULT_VLLM_URL, Backend, select_backend

logger = logging.getLogger(__name__)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class GenerateResponse(BaseModel):
    text: str
    backend: str


class IngestRequest(BaseModel):
    paths: list[str] = Field(min_length=1, description="Files or directories, relative to rag.ingest_root")


class IngestResponse(BaseModel):
    files: int
    chunks: int
    total_chunks: int


class RAGQueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_n: int | None = Field(default=None, ge=1, le=50)
    max_new_tokens: int = Field(default=512, ge=1, le=8192)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    enable_thinking: bool = False
    use_patched_model: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    # uvicorn only configures its own loggers; without this, src.* key=value logs are dropped.
    logging.basicConfig(level=os.environ.get("PATCH_RAG_LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(os.environ.get("PATCH_RAG_CONFIG"))
    serving, rag = cfg["serving"], cfg["rag"]
    device = cfg["model"].get("device", "auto")
    kind = os.environ.get("PATCH_RAG_BACKEND") or serving.get("backend", "auto")
    url = serving.get("vllm_url", DEFAULT_VLLM_URL)
    patched_path = os.environ.get("PATCH_RAG_PATCHED_MODEL") or serving.get("patched_model_path")

    app.state.backend = select_backend(os.environ.get("PATCH_RAG_MODEL") or serving["model_path"], kind, device, url)
    app.state.patched_factory = (lambda: select_backend(patched_path, kind, device, url)) if patched_path else None
    app.state.patched_backend = None
    app.state.patched_lock = threading.Lock()
    retriever = HybridRetriever.from_config(rag, device)
    app.state.pipeline = RAGPipeline(
        retriever, app.state.backend,
        {k: rag[k] for k in ("top_n", "candidates", "rerank_top_k", "rrf_k") if k in rag},
    )
    app.state.ingestor = DocumentIngestor(rag.get("chunk_size", 800), rag.get("chunk_overlap", 120))
    app.state.ingest_root = Path(rag.get("ingest_root", ".")).resolve()
    yield
    retriever.close()


app = FastAPI(title="patch-rag", lifespan=lifespan)


def _backend(request: Request, use_patched: bool) -> Backend:
    state = request.app.state
    if not use_patched:
        return state.backend
    if state.patched_factory is None:
        raise HTTPException(400, "No patched model configured (serving.patched_model_path or PATCH_RAG_PATCHED_MODEL).")
    with state.patched_lock:  # load once, lazily, so A/B testing costs memory only when used
        if state.patched_backend is None:
            try:
                state.patched_backend = state.patched_factory()
            except Exception as e:
                raise HTTPException(500, f"Could not load patched model: {e}") from e
    return state.patched_backend


@app.get("/health")
def health(request: Request) -> dict:
    state = request.app.state
    patched = state.patched_backend.name if state.patched_backend else ("configured" if state.patched_factory else None)
    return {"status": "ok", "backend": state.backend.name, "patched_backend": patched, "chunks": len(state.pipeline.retriever)}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest, request: Request) -> GenerateResponse:
    backend = request.app.state.backend
    try:
        text = backend.generate(req.prompt, req.max_new_tokens, req.temperature)
    except Exception as e:  # surface backend failures as 500 with a message, not a stack trace
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}") from e
    return GenerateResponse(text=text, backend=backend.name)


@app.post("/v1/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest, request: Request) -> IngestResponse:
    state = request.app.state
    root: Path = state.ingest_root
    paths = []
    for p in req.paths:
        resolved = (root / p).resolve()
        if not resolved.is_relative_to(root):  # trust boundary: never read outside the ingest root
            raise HTTPException(400, f"Path '{p}' is outside the ingest root.")
        if not resolved.exists():
            raise HTTPException(404, f"Path '{p}' not found under the ingest root.")
        paths.append(resolved)
    chunks = state.ingestor.ingest_paths(paths)
    retriever = state.pipeline.retriever
    retriever.ingest(chunks)
    return IngestResponse(files=len({c.source_file for c in chunks}), chunks=len(chunks), total_chunks=len(retriever))


@app.post("/v1/rag/query", response_model=RAGAnswer)
def rag_query(req: RAGQueryRequest, request: Request) -> RAGAnswer:
    backend = _backend(request, req.use_patched_model)
    try:
        answer = request.app.state.pipeline.query(
            req.question, backend=backend, top_n=req.top_n, max_new_tokens=req.max_new_tokens,
            temperature=req.temperature, enable_thinking=req.enable_thinking,
        )
    except Exception as e:
        logger.exception("rag_query failed")
        raise HTTPException(status_code=500, detail=f"RAG query failed: {e}") from e
    answer.model_variant = "patched" if req.use_patched_model else "base"
    return answer
