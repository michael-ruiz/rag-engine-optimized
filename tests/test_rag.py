"""Phase 2 tests: ingestion, hybrid RRF retrieval, <think> parsing, grounded pipeline and RAG endpoints.

All offline: hashing embedder, keyword reranker, fake LLM backend, Qdrant in a tmp directory.
"""
import hashlib
import math
import types

import pytest
from fastapi.testclient import TestClient

from src.rag import ingestor as ingestor_mod
from src.rag.ingestor import Chunk, DocumentIngestor, split_text
from src.rag.pipeline import NO_CONTEXT_ANSWER, SYSTEM_PROMPT, RAGPipeline, extract_citations, split_thinking
from src.rag.retriever import HybridRetriever, rrf_fuse, tokenize

# ---------------------------------------------------------------- fakes


def hash_embed(texts, dim=64):
    """Deterministic bag-of-words hashing embedder, L2-normalised."""
    out = []
    for t in texts:
        v = [0.0] * dim
        for tok in tokenize(t):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


def keyword_rerank(query, texts):
    q = set(tokenize(query))
    return [len(q & set(tokenize(t))) / (len(tokenize(t)) or 1) for t in texts]


class FakeBackend:
    def __init__(self, reply="", name="fake"):
        self.reply, self.name, self.calls = reply, name, []

    def chat(self, messages, max_new_tokens, temperature, enable_thinking=False):
        self.calls.append(messages)
        return self.reply

    def generate(self, prompt, max_new_tokens, temperature):
        return self.reply


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "docs"
    (root / "sub").mkdir(parents=True)
    (root / "handbook.md").write_text(
        "# Handbook\nIntro text.\n\n## PTO\nEmployees get 21 days of paid time off per year.\n\n"
        "```bash\n# not a heading\necho hi\n```\n\n## Expenses\nMeals are reimbursed up to $65 per day.\n"
    )
    (root / "faq.txt").write_text("Passwords expire every 180 days. The VPN is called Tunnel.")
    (root / "sub" / "limits.py").write_text(
        'MAX = 42\n\n\ndef is_safe(t):\n    """Shut down above 42 Nm."""\n    return t <= MAX\n\n\n'
        "class ThermalMonitor:\n    COOLDOWN_S = 120\n"
    )
    (root / "image.png").write_bytes(b"\x89PNG")
    (root / ".hidden").mkdir()
    (root / ".hidden" / "secret.md").write_text("# Secret\nzebra")
    return root


@pytest.fixture
def retriever(tmp_path):
    r = HybridRetriever(hash_embed, keyword_rerank, collection="t", path=str(tmp_path / "qdrant"))
    yield r
    r.close()


# ---------------------------------------------------------------- ingestion


def test_split_text_respects_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = split_text(text, chunk_size=120, chunk_overlap=30)
    assert len(chunks) > 1 and all(len(c) <= 120 for c in chunks)
    for a, b in zip(chunks, chunks[1:]):
        assert b.split()[0] in a.split()  # each chunk starts inside the previous one's tail
    assert set(text.split()) == {w for c in chunks for w in c.split()}  # nothing dropped


def test_split_text_hard_cuts_unbroken_text():
    assert all(len(c) <= 50 for c in split_text("x" * 500, 50, 10))


def test_invalid_chunk_params():
    with pytest.raises(ValueError):
        DocumentIngestor(chunk_size=100, chunk_overlap=100)


def test_markdown_sections_and_metadata(corpus):
    chunks = DocumentIngestor(chunk_size=200, chunk_overlap=20).chunk_file(corpus / "handbook.md")
    assert [c.section_header for c in chunks] == ["Handbook", "PTO", "Expenses"]  # fenced "# not a heading" ignored
    assert [c.chunk_index for c in chunks] == [0, 1, 2]
    assert chunks[1].source_file.endswith("docs/handbook.md") and "21 days" in chunks[1].text
    assert "## PTO" in chunks[1].text


def test_python_sections(corpus):
    headers = [c.section_header for c in DocumentIngestor().chunk_file(corpus / "sub" / "limits.py")]
    assert headers == ["", "def is_safe", "class ThermalMonitor"]


def test_pdf_pages_become_sections(monkeypatch, tmp_path):
    fake_pages = [types.SimpleNamespace(extract_text=lambda: "First page."),
                  types.SimpleNamespace(extract_text=lambda: None),
                  types.SimpleNamespace(extract_text=lambda: "Third page.")]
    monkeypatch.setitem(__import__("sys").modules, "pypdf", types.SimpleNamespace(PdfReader=lambda p: types.SimpleNamespace(pages=fake_pages)))
    (tmp_path / "doc.pdf").write_bytes(b"%PDF")
    chunks = DocumentIngestor().chunk_file(tmp_path / "doc.pdf")
    assert [(c.section_header, c.text) for c in chunks] == [("page 1", "First page."), ("page 3", "Third page.")]


def test_ingest_paths_walks_dirs_and_skips_hidden_and_unsupported(corpus):
    files = {c.source_file.split("docs/")[1] for c in DocumentIngestor().ingest_paths([corpus])}
    assert files == {"handbook.md", "faq.txt", "sub/limits.py"}
    with pytest.raises(FileNotFoundError):
        DocumentIngestor().ingest_paths([corpus / "missing"])
    assert ingestor_mod.SUPPORTED_SUFFIXES == {".md", ".txt", ".pdf", ".py"}


# ---------------------------------------------------------------- retrieval


def test_rrf_fuse_math():
    scores = rrf_fuse([["a", "b", "c"], ["c", "a"]], k=60)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert scores["b"] == pytest.approx(1 / 62)
    assert sorted(scores, key=scores.get, reverse=True) == ["a", "c", "b"]


def test_bm25_rescues_keyword_match_when_dense_is_blind(tmp_path):
    chunks = [Chunk(f"generic filler text number {i}", "f.txt", i, "") for i in range(5)]
    chunks.append(Chunk("the xylophone inventory is in room 114", "f.txt", 5, ""))
    r = HybridRetriever(lambda texts: [[1.0, 0.0]] * len(texts), None, collection="t", path=str(tmp_path / "q"))
    try:
        r.ingest(chunks)
        hits = r.search("xylophone", top_n=3)
        assert hits[0].chunk.chunk_index == 5
        assert hits[0].rrf_score > hits[1].rrf_score  # dense + lexical vs dense only
    finally:
        r.close()


def test_hybrid_search_with_rerank(retriever, corpus):
    retriever.ingest(DocumentIngestor(200, 20).ingest_paths([corpus]))
    hits = retriever.search("how many days of paid time off", top_n=2)
    assert len(hits) == 2
    assert "21 days" in hits[0].chunk.text and hits[0].rerank_score is not None
    assert hits[0].rerank_score >= hits[1].rerank_score
    assert retriever.search("   ") == []


def test_reingest_replaces_chunks_and_persists(tmp_path, corpus):
    path = str(tmp_path / "qdrant")
    ing = DocumentIngestor(200, 20)
    r = HybridRetriever(hash_embed, collection="t", path=path)
    r.ingest(ing.ingest_paths([corpus]))
    n = len(r)
    (corpus / "handbook.md").write_text("# Handbook\nOnly one section now.")
    r.ingest(ing.chunk_file(corpus / "handbook.md"))
    assert len(r) == n - 2  # 3 handbook chunks replaced by 1; stale ones gone
    r.close()

    reopened = HybridRetriever(hash_embed, collection="t", path=path)  # BM25 rebuilt from disk
    try:
        assert len(reopened) == n - 2
        assert "Tunnel" in reopened.search("VPN name Tunnel", top_n=1)[0].chunk.text
        with pytest.raises(ValueError, match="new collection"):
            reopened.embed_fn = lambda texts: [[1.0, 0.0]] * len(texts)
            reopened.ingest([Chunk("x", "y.txt", 0, "")])
    finally:
        reopened.close()


def test_empty_collection_returns_nothing(retriever):
    assert len(retriever) == 0 and retriever.search("anything") == []


# ---------------------------------------------------------------- generation parsing


@pytest.mark.parametrize(
    "raw, thinking, answer",
    [
        ("<think>\nreason\n</think>\n\nThe answer.", "reason", "The answer."),
        ("reason only closed\n</think>\nThe answer.", "reason only closed", "The answer."),  # template opened <think>
        ("<think>cut off mid thought", "cut off mid thought", ""),
        ("Plain answer.", "", "Plain answer."),
        ("<think></think>Answer.", "", "Answer."),
    ],
)
def test_split_thinking(raw, thinking, answer):
    assert split_thinking(raw) == (thinking, answer)


def test_split_thinking_when_prompt_opened_think_and_output_was_cut():
    assert split_thinking("Thinking Process: step 1...", thinking_opened=True) == ("Thinking Process: step 1...", "")
    assert split_thinking("trace</think>Answer.", thinking_opened=True) == ("trace", "Answer.")


def test_extract_citations_dedupes():
    cites = extract_citations("A [docs/a.md:0]. B [docs/a.md:0][b.txt:12]. Not one [x] or [y:z].")
    assert [(c.source_file, c.chunk_index) for c in cites] == [("docs/a.md", 0), ("b.txt", 12)]


# ---------------------------------------------------------------- pipeline


def _loaded_pipeline(retriever, corpus, reply):
    retriever.ingest(DocumentIngestor(200, 20).ingest_paths([corpus]))
    backend = FakeBackend(reply)
    return RAGPipeline(retriever, backend, {"top_n": 2}), backend


def test_pipeline_grounds_parses_and_validates_citations(retriever, corpus):
    pipe, backend = _loaded_pipeline(retriever, corpus, "")
    pto = next(r.chunk for r in retriever.search("paid time off days", top_n=1))
    backend.reply = f"<think>look at PTO</think>Employees get 21 days [{pto.citation}]. Also [made_up.md:9]."

    res = pipe.query("How many days of paid time off?", enable_thinking=True)
    assert res.answer.startswith("Employees get 21 days") and res.thinking_trace == "look at PTO"
    assert [(c.source_file, c.chunk_index) for c in res.citations] == [(pto.source_file, pto.chunk_index)]
    assert [(c.source_file, c.chunk_index) for c in res.invalid_citations] == [("made_up.md", 9)]
    assert len(res.contexts) == 2 and res.retrieval_latency_ms >= 0 and res.generation_latency_ms >= 0
    system, user = backend.calls[0]
    assert system == {"role": "system", "content": SYSTEM_PROMPT}
    assert f"[{pto.citation}]" in user["content"] and "Question: How many days" in user["content"]


def test_pipeline_refuses_without_context(retriever):
    backend = FakeBackend("should not be called")
    res = RAGPipeline(retriever, backend).query("anything?")
    assert res.answer == NO_CONTEXT_ANSWER and res.contexts == [] and backend.calls == []


# ---------------------------------------------------------------- endpoints


@pytest.fixture
def client(retriever, corpus):
    """App with fake components injected; lifespan is skipped so no real models load."""
    import threading

    from src.serving.app import app

    pto = "Employees get 21 days."
    base, patched = FakeBackend(pto, "base"), FakeBackend(pto + " (patched)", "patched")
    factory_calls = []
    s = app.state
    s.backend, s.patched_backend, s.patched_lock = base, None, threading.Lock()
    s.patched_factory = lambda: factory_calls.append(1) or patched
    s.pipeline = RAGPipeline(retriever, base, {"top_n": 2})
    s.ingestor = DocumentIngestor(200, 20)
    s.ingest_root = corpus.resolve()
    c = TestClient(app)
    c.factory_calls = factory_calls
    return c


def test_ingest_endpoint_and_path_guard(client):
    r = client.post("/v1/ingest", json={"paths": ["."]})
    assert r.status_code == 200 and r.json() == {"files": 3, "chunks": r.json()["chunks"], "total_chunks": r.json()["chunks"]}
    assert client.post("/v1/ingest", json={"paths": ["../"]}).status_code == 400
    assert client.post("/v1/ingest", json={"paths": ["/etc"]}).status_code == 400
    assert client.post("/v1/ingest", json={"paths": ["nope.md"]}).status_code == 404
    assert client.post("/v1/ingest", json={"paths": []}).status_code == 422


def test_rag_query_endpoint_ab(client):
    client.post("/v1/ingest", json={"paths": ["."]})
    base = client.post("/v1/rag/query", json={"question": "paid time off?"})
    assert base.status_code == 200
    body = base.json()
    assert body["backend"] == "base" and body["model_variant"] == "base" and body["answer"] == "Employees get 21 days." and len(body["contexts"]) == 2
    assert {"thinking_trace", "citations", "invalid_citations", "retrieval_latency_ms", "generation_latency_ms"} <= body.keys()

    for _ in range(2):
        patched = client.post("/v1/rag/query", json={"question": "paid time off?", "use_patched_model": True})
        assert patched.json()["backend"] == "patched" and patched.json()["model_variant"] == "patched"
    assert client.factory_calls == [1]  # patched model loaded lazily, exactly once
    assert client.get("/health").json()["patched_backend"] == "patched"
    assert client.post("/v1/rag/query", json={"question": ""}).status_code == 422


def test_rag_query_without_patched_model_configured(client):
    client.app.state.patched_factory = None
    r = client.post("/v1/rag/query", json={"question": "x", "use_patched_model": True})
    assert r.status_code == 400 and "No patched model" in r.json()["detail"]
