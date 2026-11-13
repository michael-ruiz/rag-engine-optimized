"""Chunk documents (.md, .txt, .pdf, .py) and index them into the hybrid retriever.

Example:
  python scripts/ingest_docs.py data/sample_corpus
Embedded Qdrant locks its folder: stop a running server first, or use POST /v1/ingest instead.
"""
from __future__ import annotations

import argparse
import logging

from src import load_config
from src.rag.ingestor import DocumentIngestor
from src.rag.retriever import HybridRetriever


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", help="Files or directories to ingest")
    p.add_argument("--config", help="YAML config (default: config/default_config.yaml)")
    p.add_argument("--collection", help="Override rag.collection")
    p.add_argument("--chunk-size", type=int)
    p.add_argument("--chunk-overlap", type=int)
    p.add_argument("--query", help="After ingesting, run one retrieval and print the top chunks")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    rag = {**cfg["rag"], **({"collection": args.collection} if args.collection else {})}
    ingestor = DocumentIngestor(args.chunk_size or rag["chunk_size"], args.chunk_overlap or rag["chunk_overlap"])
    try:
        chunks = ingestor.ingest_paths(args.paths)
    except FileNotFoundError as e:
        raise SystemExit(f"error: path not found: {e}") from None
    if not chunks:
        raise SystemExit("error: no supported files (.md, .txt, .pdf, .py) found")

    retriever = HybridRetriever.from_config(rag, cfg["model"].get("device", "auto"))
    try:
        retriever.ingest(chunks)
        print(f"indexed {len(chunks)} chunks from {len({c.source_file for c in chunks})} files "
              f"-> collection '{rag['collection']}' ({len(retriever)} total)")
        if args.query:
            for r in retriever.search(args.query, top_n=rag["top_n"]):
                print(f"  [{r.chunk.citation}] rerank={r.rerank_score} rrf={r.rrf_score:.4f} {r.chunk.text[:80]!r}")
    finally:
        retriever.close()


if __name__ == "__main__":
    main()
