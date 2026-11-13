import pytest
import torch
from transformers import LlamaForCausalLM

from src.model.patcher import WeightPatcher

DOWN = "model.layers.0.mlp.down_proj.weight"


def _param(model, name):
    return dict(model.named_parameters(remove_duplicate=False))[name]


def test_rank1_removes_direction_in_place(tiny_model):
    w = _param(tiny_model, DOWN)
    shape, dtype, ptr = w.shape, w.dtype, w.data_ptr()
    u = torch.randn(w.shape[0])
    before = (u / u.norm()) @ w.float()
    WeightPatcher(tiny_model).rank1_project_out(DOWN, u, alpha=1.0)
    after = (u / u.norm()) @ w.float()
    assert (w.shape, w.dtype, w.data_ptr()) == (shape, dtype, ptr)
    assert after.norm() < 0.05 * before.norm()  # bf16 rounding leaves a small residual


def test_null_columns_only_touches_given_columns(tiny_model):
    w = _param(tiny_model, DOWN)
    orig = w.clone()
    WeightPatcher(tiny_model).null_columns(DOWN, [1, 5])
    assert torch.all(w[:, [1, 5]] == 0)
    keep = [c for c in range(w.shape[1]) if c not in (1, 5)]
    assert torch.equal(w[:, keep], orig[:, keep])


def test_validation_errors(tiny_model):
    patcher = WeightPatcher(tiny_model)
    with pytest.raises(ValueError, match="output dims"):
        patcher.rank1_project_out(DOWN, torch.randn(7))
    with pytest.raises(ValueError, match="non-zero"):
        patcher.rank1_project_out(DOWN, torch.zeros(32))
    with pytest.raises(IndexError):
        patcher.null_columns(DOWN, [64])
    with pytest.raises(KeyError):
        patcher.null_columns("nope.weight", [0])
    with pytest.raises(ValueError, match="2-D"):
        patcher.null_columns("model.norm.weight", [0])
    assert patcher.log == []


def test_tied_weight_guard(tiny_model):
    patcher = WeightPatcher(tiny_model)
    with pytest.raises(ValueError, match="shares storage"):
        patcher.null_columns("lm_head.weight", [0])
    patcher.null_columns("lm_head.weight", [0], allow_tied=True)
    assert torch.all(tiny_model.model.embed_tokens.weight[:, 0] == 0)


def test_save_roundtrip(tiny_model, tmp_path):
    patcher = WeightPatcher(tiny_model)
    patcher.null_columns(DOWN, [3])
    out = patcher.save(tmp_path / "patched")
    assert (out / "patch_log.json").exists()
    reloaded = LlamaForCausalLM.from_pretrained(out, dtype=torch.bfloat16)
    assert torch.equal(_param(reloaded, DOWN), _param(tiny_model, DOWN))
