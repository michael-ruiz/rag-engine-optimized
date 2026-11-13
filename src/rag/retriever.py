"""Local-first hybrid retrieval: Qdrant dense search + BM25 lexical search, fused with RRF, then
cross-encoder reranking. Runs with embedded Qdrant storage (no Docker) or a remote Qdrant URL.
"""
from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, FilterSelector, MatchAny, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

from src.rag.ingestor import Chunk

logger = logging.getLogger(__name__)

EmbedFn = Callable[[list[str]], list[list[float]]]
RerankFn = Callable[[str, list[str]], list[float]]
_TOKEN = re.compile(r"\w+")
_ID_NAMESPACE = uuid.UUID("6f1c6a52-5d0b-4c39-9a57-0d2a8f1e7b11")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank), rank starting at 1."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def sentence_transformer_embedder(model_name: str, device: str = "auto") -> EmbedFn:
    """Dense embedder on DeviceManager's device; vectors are L2-normalised for cosine search."""
    from sentence_transformers import SentenceTransformer

    from src.utils.device import DeviceManager

    model = SentenceTransformer(model_name, device=DeviceManager.detect(device).device)
    return lambda texts: model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).tolist()


def cross_encoder_reranker(model_name: str, device: str = "auto") -> RerankFn:
    from sentence_transformers import CrossEncoder

    from src.utils.device import DeviceManager

    model = CrossEncoder(model_name, device=DeviceManager.detect(device).device)
    return lambda query, texts: model.predict([(query, t) for t in texts]).tolist()


@dataclass
class RetrievedChunk:
    chunk: Chunk
    rrf_score: float
    rerank_score: float | None = None


class HybridRetriever:
    """Dense (Qdrant) + BM25 retrieval fused with RRF, optionally reranked by a cross-encoder.

    Pass `url` for a remote Qdrant server, otherwise data persists under `path`.
    Embedded mode locks `path`: only one process (or client) may open it at a time.
    """

    def __init__(
        self,
        embed_fn: EmbedFn,
        rerank_fn: RerankFn | None = None,
        collection: str = "patch_rag",
        path: str | None = "./data/qdrant_local",
        url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.embed_fn = embed_fn
        self.rerank_fn = rerank_fn
        self.collection = collection
        self.client = QdrantClient(url=url, api_key=api_key) if url else QdrantClient(path=path)
        self._load_corpus()

    @classmethod
    def from_config(cls, rag: dict, device: str = "auto") -> HybridRetriever:
        """Build from the `rag:` config section with the real sentence-transformers models."""
        import os

        reranker = rag.get("reranker_model")
        return cls(
            embed_fn=sentence_transformer_embedder(rag["embedding_model"], device),
            rerank_fn=cross_encoder_reranker(reranker, device) if reranker else None,
            collection=rag.get("collection", "patch_rag"),
            path=rag.get("qdrant_path", "./data/qdrant_local"),
            url=rag.get("qdrant_url"),
            api_key=os.environ.get("QDRANT_API_KEY"),
        )

    # shortcut: BM25 index lives in memory and is rebuilt from Qdrant on startup and after each ingest;
    # fine for local/dev corpora, upgrade to Qdrant sparse vectors (fastembed Bm25) for large ones.
    def _load_corpus(self) -> None:
        chunks: dict[str, Chunk] = {}
        if self.client.collection_exists(self.collection):
            offset = None
            while True:
                points, offset = self.client.scroll(self.collection, limit=1000, offset=offset, with_payload=True)
                chunks.update({str(p.id): Chunk(**p.payload) for p in points})
                if offset is None:
                    break
        ids = list(chunks)
        bm25 = BM25Okapi([tokenize(chunks[i].text) for i in ids]) if ids else None
        self._index = (chunks, ids, bm25)  # one assignment, so concurrent searches never see a half-built index
        logger.info("loaded collection=%s chunks=%d", self.collection, len(ids))

    def __len__(self) -> int:
        return len(self._index[1])

    @staticmethod
    def point_id(chunk: Chunk) -> str:
        return str(uuid.uuid5(_ID_NAMESPACE, chunk.citation))

    def ingest(self, chunks: list[Chunk], batch_size: int = 64) -> int:
        """Index chunks. Re-ingesting a source file replaces all of its previous chunks."""
        if not chunks:
            return 0
        vectors = [v for i in range(0, len(chunks), batch_size)
                   for v in self.embed_fn([c.text for c in chunks[i:i + batch_size]])]
        dim = len(vectors[0])
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(self.collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
        else:
            existing = self.client.get_collection(self.collection).config.params.vectors.size
            if existing != dim:
                raise ValueError(
                    f"Collection '{self.collection}' stores {existing}-d vectors but the embedder makes {dim}-d; "
                    "use a new collection name when changing embedding models."
                )
        sources = sorted({c.source_file for c in chunks})
        self.client.delete(
            self.collection,
            points_selector=FilterSelector(filter=Filter(must=[FieldCondition(key="source_file", match=MatchAny(any=sources))])),
        )
        self.client.upsert(
            self.collection,
            points=[PointStruct(id=self.point_id(c), vector=v, payload=c.to_payload()) for c, v in zip(chunks, vectors, strict=True)],
        )
        self._load_corpus()
        logger.info("ingested chunks=%d sources=%d", len(chunks), len(sources))
        return len(chunks)

    def _dense(self, query: str, limit: int) -> list[str]:
        hits = self.client.query_points(self.collection, query=self.embed_fn([query])[0], limit=limit).points
        return [str(h.id) for h in hits]

    @staticmethod
    def _lexical(index: tuple, query: str, limit: int) -> list[str]:
        _, ids, bm25 = index
        scores = bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limit]
        return [ids[i] for i in order if scores[i] > 0]

    def search(self, query: str, top_n: int = 4, candidates: int = 30, rerank_top_k: int = 20, rrf_k: int = 60) -> list[RetrievedChunk]:
        """Dense top-`candidates` + BM25 top-`candidates` -> RRF -> top `rerank_top_k` -> rerank -> top `top_n`."""
        index = self._index
        chunks, ids, _ = index
        if not ids or not query.strip():
            return []
        fused = rrf_fuse([self._dense(query, candidates), self._lexical(index, query, candidates)], k=rrf_k)
        top = sorted(fused, key=fused.get, reverse=True)[:rerank_top_k]
        results = [RetrievedChunk(chunks[i], fused[i]) for i in top if i in chunks]
        if self.rerank_fn and results:
            for r, s in zip(results, self.rerank_fn(query, [r.chunk.text for r in results]), strict=True):
                r.rerank_score = float(s)
            results.sort(key=lambda r: r.rerank_score, reverse=True)
        return results[:top_n]

    def close(self) -> None:
        self.client.close()
