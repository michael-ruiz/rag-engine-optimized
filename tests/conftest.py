import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM


@pytest.fixture
def tiny_model():
    """Tiny randomly initialised Llama (offline) with tied embeddings, like Qwen3.5-0.8B."""
    torch.manual_seed(0)
    cfg = LlamaConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=100, tie_word_embeddings=True,
    )
    return LlamaForCausalLM(cfg).to(torch.bfloat16).eval()
