"""Surgical in-place weight edits (rank-1 projection removal, column nulling) and checkpoint export."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.model.inspector import load_model

logger = logging.getLogger(__name__)


class WeightPatcher:
    """Apply validated in-place edits to a model's 2-D weight matrices.

    Weights follow the nn.Linear convention W: [out_features, in_features].
    """

    def __init__(
        self,
        model_or_path: nn.Module | str | Path,
        tokenizer=None,
        device: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        # dtype stays bf16 (not DeviceManager's CPU fp32) so the saved checkpoint keeps the base model's format.
        if isinstance(model_or_path, (str, Path)):
            model_or_path, tokenizer = load_model(str(model_or_path), device, dtype)
        self.model = model_or_path
        self.tokenizer = tokenizer
        self.log: list[dict[str, Any]] = []

    def _get(self, param_name: str, allow_tied: bool) -> nn.Parameter:
        params = dict(self.model.named_parameters(remove_duplicate=False))
        if param_name not in params:
            raise KeyError(f"Parameter '{param_name}' not found.")
        w = params[param_name]
        if w.ndim != 2:
            raise ValueError(f"'{param_name}' has shape {tuple(w.shape)}; only 2-D weights can be patched.")
        aliases = [n for n, p in params.items() if n != param_name and p.data_ptr() == w.data_ptr()]
        if aliases and not allow_tied:
            raise ValueError(
                f"'{param_name}' shares storage with {aliases}; editing it edits those too. "
                "Pass allow_tied=True if that is intended."
            )
        return w

    @staticmethod
    def _check_unchanged(w: torch.Tensor, shape, dtype, device, ptr) -> None:
        if (w.shape, w.dtype, w.device, w.data_ptr()) != (shape, dtype, device, ptr):
            raise RuntimeError("Patch changed tensor shape/dtype/device/storage; this is a bug.")

    @torch.no_grad()
    def rank1_project_out(
        self, param_name: str, u: torch.Tensor, alpha: float = 1.0, allow_tied: bool = False
    ) -> None:
        """W' = W - alpha * u (u^T W), removing direction u from W's output space.

        u is normalized internally and must have length out_features.
        """
        w = self._get(param_name, allow_tied)
        u = u.detach().flatten().to(device=w.device, dtype=torch.float32)
        if u.numel() != w.shape[0]:
            raise ValueError(f"Direction has {u.numel()} dims; '{param_name}' has {w.shape[0]} output dims.")
        norm = u.norm()
        if not torch.isfinite(norm) or norm == 0:
            raise ValueError("Direction vector must be finite and non-zero.")
        u = u / norm

        before = (w.shape, w.dtype, w.device, w.data_ptr())
        w32 = w.float()
        new = w32 - alpha * torch.outer(u, u @ w32)
        if not torch.isfinite(new).all():
            raise ValueError("Patch produced non-finite values; weight left unchanged.")
        w.copy_(new.to(w.dtype))
        self._check_unchanged(w, *before)
        self.log.append({"op": "rank1_project_out", "param": param_name, "alpha": alpha})
        logger.info("rank-1 projection on %s (alpha=%s)", param_name, alpha)

    @torch.no_grad()
    def null_columns(self, param_name: str, cols: list[int], allow_tied: bool = False) -> None:
        """Zero the given input-feature columns W[:, cols] (e.g. silence MLP neurons in down_proj)."""
        w = self._get(param_name, allow_tied)
        bad = [c for c in cols if not 0 <= c < w.shape[1]]
        if bad:
            raise IndexError(f"Columns {bad} out of range for '{param_name}' with {w.shape[1]} columns.")
        before = (w.shape, w.dtype, w.device, w.data_ptr())
        w[:, cols] = 0
        self._check_unchanged(w, *before)
        self.log.append({"op": "null_columns", "param": param_name, "cols": list(cols)})
        logger.info("nulled %d columns of %s", len(cols), param_name)

    def save(self, out_dir: str | Path) -> Path:
        """Write the patched checkpoint, tokenizer (if any) and patch_log.json to `out_dir`."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(out)
        (out / "patch_log.json").write_text(json.dumps(self.log, indent=2))
        return out
