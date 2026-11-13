"""Apply a rank-1 projection removal or column-nulling edit and export the patched checkpoint.

Examples:
  # remove the mean layer-15 MLP output direction captured by inspect_weights.py
  python scripts/apply_patch.py --param model.layers.15.mlp.down_proj.weight \
      --rank1 --direction acts.pt --direction-key model.layers.15.mlp --alpha 0.5 --out patched/
  # silence MLP neurons 12 and 40 in layer 15
  python scripts/apply_patch.py --param model.layers.15.mlp.down_proj.weight --null-cols 12,40 --out patched/
"""
from __future__ import annotations

import argparse
import logging

import torch

from src import load_config
from src.model.patcher import WeightPatcher


def load_direction(path: str, key: str | None) -> torch.Tensor:
    """Load a direction from a .pt tensor or an inspector cache dict; mean-pool to 1-D."""
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        if key not in obj:
            raise SystemExit(f"--direction-key required; available keys: {list(obj)}")
        obj = obj[key]
    t = torch.as_tensor(obj).float()
    return t.reshape(-1, t.shape[-1]).mean(0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="YAML config (default: config/default_config.yaml)")
    p.add_argument("--model", help="HF model id or checkpoint path (overrides config)")
    p.add_argument("--param", required=True, help="Weight parameter name, e.g. model.layers.15.mlp.down_proj.weight")
    p.add_argument("--out", required=True, help="Output directory for the patched checkpoint")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--rank1", action="store_true", help="W' = W - alpha * u u^T W")
    mode.add_argument("--null-cols", help="Comma-separated column indices to zero")
    p.add_argument("--direction", help="Path to .pt direction tensor or activation cache (for --rank1)")
    p.add_argument("--direction-key", help="Key into the activation cache dict")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--allow-tied", action="store_true", help="Allow editing weights tied to other params")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.rank1 and not args.direction:
        p.error("--rank1 requires --direction")

    cfg = load_config(args.config)
    patcher = WeightPatcher(args.model or cfg["model"]["id"], device=cfg["model"].get("device", "auto"))
    try:
        if args.rank1:
            u = load_direction(args.direction, args.direction_key)
            patcher.rank1_project_out(args.param, u, args.alpha, args.allow_tied)
        else:
            cols = [int(c) for c in args.null_cols.split(",") if c.strip()]
            patcher.null_columns(args.param, cols, args.allow_tied)
    except (KeyError, ValueError, IndexError) as e:
        raise SystemExit(f"error: {e}") from None  # nothing is saved when validation fails
    print(f"patched checkpoint -> {patcher.save(args.out)}")


if __name__ == "__main__":
    main()
