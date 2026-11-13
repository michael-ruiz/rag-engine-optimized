"""Print parameter shapes/memory and capture activations from target modules.

Example:
  python scripts/inspect_weights.py --filter layers.15. --targets model.layers.15.mlp \
      --prompt "3x + 5 = 20. Solve for x." --save-activations acts.pt
"""
from __future__ import annotations

import argparse
import logging

from src import load_config
from src.model.inspector import ModelInspector


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="YAML config (default: config/default_config.yaml)")
    p.add_argument("--model", help="HF model id or local path (overrides config)")
    p.add_argument("--filter", help="Only print parameters whose name contains this string")
    p.add_argument("--targets", nargs="*", help="Module names to hook (default: config inspect.targets)")
    p.add_argument("--prompt", help="Run one forward pass on this prompt and report captured activations")
    p.add_argument("--save-activations", help="torch.save the activation cache to this path")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    inspector = ModelInspector(model_id=args.model or cfg["model"]["id"], device=cfg["model"].get("device", "auto"))
    inspector.print_summary(args.filter)

    if not args.prompt:
        return
    with inspector:
        inspector.register_hooks(args.targets if args.targets is not None else cfg["inspect"]["targets"])
        logits = inspector.run(args.prompt)
    print(f"\nlogits: {tuple(logits.shape)}")
    for name, t in inspector.cache.items():
        print(f"{name:<45} {tuple(t.shape)}  mean={t.mean():.4f} std={t.std():.4f} absmax={t.abs().max():.4f}")
    if args.save_activations:
        inspector.save_cache(args.save_activations)
        print(f"saved activations -> {args.save_activations}")


if __name__ == "__main__":
    main()
