"""Grounded RAG orchestration: retrieve -> strict grounded prompt -> generate -> parse <think> and citations."""
from __future__ import annotations

import logging
import re
import time

from pydantic import BaseModel

from src.rag.retriever import HybridRetriever, RetrievedChunk
from src.serving.backends import Backend

logger = logging.getLogger(__name__)

NO_CONTEXT_ANSWER = "I don't know: no relevant documents were found in the knowledge base."

SYSTEM_PROMPT = """You are a precise enterprise assistant. Answer ONLY using the numbered context passages below.

Rules:
1. Every factual sentence must end with a citation of the passage it came from, written exactly as that passage's tag in square brackets, in the form [<source_file>:<chunk_index>]. Use several tags if needed.
2. Never use outside knowledge. Never invent facts, numbers, names, or citations.
3. If the context does not contain the answer, reply exactly: "I don't know based on the provided documents."
4. Be concise. Do not repeat yourself."""

_CITATION = re.compile(r"\[([^\[\]\n]+?):(\d+)\]")


class Citation(BaseModel):
    source_file: str
    chunk_index: int


class ContextChunk(BaseModel):
    source_file: str
    chunk_index: int
    section_header: str
    text: str
    rrf_score: float
    rerank_score: float | None = None


class RAGAnswer(BaseModel):
    answer: str
    thinking_trace: str
    citations: list[Citation]
    invalid_citations: list[Citation]  # cited but never retrieved: a hallucination signal
    contexts: list[ContextChunk]
    retrieval_latency_ms: float
    generation_latency_ms: float
    backend: str
    model_variant: str = "base"  # "base" | "patched", for A/B comparisons


def split_thinking(text: str, thinking_opened: bool = False) -> tuple[str, str]:
    """Return (thinking_trace, answer).

    Handles a full <think>...</think> block, a closing tag only (Qwen3.5's template puts the opening
    <think> in the prompt), and an unclosed <think> (generation cut off mid-thought: no answer).
    `thinking_opened`: the prompt already opened <think>, so untagged output is an unfinished trace.
    """
    if thinking_opened and "<think>" not in text and "</think>" not in text:
        return text.strip(), ""
    if "</think>" in text:
        thinking, answer = text.split("</think>", 1)
        return thinking.replace("<think>", "", 1).strip(), answer.replace("</think>", "").strip()
    if "<think>" in text:
        before, thinking = text.split("<think>", 1)
        return thinking.strip(), before.strip()
    return "", text.strip()


def extract_citations(answer: str) -> list[Citation]:
    seen: dict[tuple[str, int], Citation] = {}
    for src, idx in _CITATION.findall(answer):
        seen.setdefault((src.strip(), int(idx)), Citation(source_file=src.strip(), chunk_index=int(idx)))
    return list(seen.values())


def format_context(results: list[RetrievedChunk]) -> str:
    blocks = []
    for r in results:
        header = f" ({r.chunk.section_header})" if r.chunk.section_header else ""
        blocks.append(f"[{r.chunk.citation}]{header}\n{r.chunk.text}")
    return "\n\n---\n\n".join(blocks)


class RAGPipeline:
    def __init__(self, retriever: HybridRetriever, backend: Backend, search_kwargs: dict | None = None) -> None:
        self.retriever = retriever
        self.backend = backend
        self.search_kwargs = search_kwargs or {}

    def query(
        self,
        question: str,
        backend: Backend | None = None,
        top_n: int | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        enable_thinking: bool = False,
    ) -> RAGAnswer:
        backend = backend or self.backend
        kwargs = {**self.search_kwargs, **({"top_n": top_n} if top_n else {})}

        t0 = time.perf_counter()
        results = self.retriever.search(question, **kwargs)
        retrieval_ms = (time.perf_counter() - t0) * 1000
        contexts = [
            ContextChunk(**r.chunk.to_payload(), rrf_score=r.rrf_score, rerank_score=r.rerank_score) for r in results
        ]

        if not results:  # nothing to ground on: refuse instead of letting the model guess
            logger.info("rag_query retrieved=0 retrieval_ms=%.1f", retrieval_ms)
            return RAGAnswer(
                answer=NO_CONTEXT_ANSWER, thinking_trace="", citations=[], invalid_citations=[], contexts=[],
                retrieval_latency_ms=retrieval_ms, generation_latency_ms=0.0, backend=backend.name,
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Context:\n\n{format_context(results)}\n\nQuestion: {question}\n\n"
                "Answer using only the context above and cite each fact with its passage tag."  # small models follow the last instruction best
            )},
        ]
        t1 = time.perf_counter()
        raw = backend.chat(messages, max_new_tokens, temperature, enable_thinking)
        generation_ms = (time.perf_counter() - t1) * 1000

        thinking, answer = split_thinking(raw, thinking_opened=enable_thinking)
        retrieved = {(r.chunk.source_file, r.chunk.chunk_index) for r in results}
        cited = extract_citations(answer)
        valid = [c for c in cited if (c.source_file, c.chunk_index) in retrieved]
        invalid = [c for c in cited if (c.source_file, c.chunk_index) not in retrieved]
        logger.info(
            "rag_query retrieved=%d citations=%d invalid=%d retrieval_ms=%.1f generation_ms=%.1f backend=%s",
            len(results), len(valid), len(invalid), retrieval_ms, generation_ms, backend.name,
        )
        return RAGAnswer(
            answer=answer, thinking_trace=thinking, citations=valid, invalid_citations=invalid, contexts=contexts,
            retrieval_latency_ms=retrieval_ms, generation_latency_ms=generation_ms, backend=backend.name,
        )
