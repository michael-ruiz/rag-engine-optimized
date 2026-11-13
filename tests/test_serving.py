import io
import json

import pytest
import torch

from src.serving import backends
from src.serving.backends import TransformersMPSBackend, VLLMBackend, select_backend


@pytest.fixture
def no_model_load(monkeypatch):
    """Stop TransformersMPSBackend from loading real weights."""
    monkeypatch.setattr(TransformersMPSBackend, "__init__", lambda self, *a, **k: None)


def test_auto_without_cuda_uses_transformers(monkeypatch, no_model_load):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert isinstance(select_backend("m"), TransformersMPSBackend)


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        select_backend("m", backend="tensorrt")


def test_vllm_http_completion(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    sent = {}

    def fake_urlopen(req, timeout):
        sent.update(url=req.full_url, body=json.loads(req.data))
        return io.BytesIO(json.dumps({"choices": [{"text": " Paris"}]}).encode())

    monkeypatch.setattr(backends.urllib.request, "urlopen", fake_urlopen)
    b = select_backend("Qwen/Qwen3.5-0.8B", backend="vllm", vllm_url="http://127.0.0.1:8000/v1/")
    assert isinstance(b, VLLMBackend) and b.name == "vllm-http"
    assert b.generate("Capital of France?", 4, 0.0) == " Paris"
    assert sent["url"] == "http://127.0.0.1:8000/v1/completions"
    assert sent["body"] == {"model": "Qwen/Qwen3.5-0.8B", "prompt": "Capital of France?", "max_tokens": 4, "temperature": 0.0}


def test_vllm_http_unreachable_is_clear_error(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    b = VLLMBackend("m", base_url="http://127.0.0.1:9/v1")
    with pytest.raises(RuntimeError, match="unreachable"):
        b.generate("hi", 1, 0.0)
