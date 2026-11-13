import pytest
import torch

from src.model.inspector import ModelInspector


def test_scan_totals_match_parameters(tiny_model):
    rows = ModelInspector(tiny_model).scan()
    assert sum(r.bytes for r in rows) == sum(p.numel() * p.element_size() for p in tiny_model.parameters())
    assert all(r.dtype == torch.bfloat16 for r in rows)
    assert all("layers.0." in r.name for r in ModelInspector(tiny_model).scan("layers.0."))


def test_hooks_capture_full_hidden_states(tiny_model):
    ids = torch.randint(0, 100, (1, 7))
    with ModelInspector(tiny_model) as insp:
        insp.register_hooks(["model.layers.0.mlp", "model.layers.1.self_attn"])
        insp.forward(ids)
        assert insp.cache["model.layers.0.mlp"].shape == (1, 7, 32)
        assert insp.cache["model.layers.1.self_attn"].shape == (1, 7, 32)  # tuple output unwrapped
    assert not insp._handles
    assert not tiny_model.model.layers[0].mlp._forward_hooks


def test_unknown_target_raises_without_partial_hooks(tiny_model):
    insp = ModelInspector(tiny_model)
    with pytest.raises(KeyError, match="Close matches"):
        insp.register_hooks(["model.layers.0.mlp", "model.layers.0.mlpp"])
    assert not insp._handles
