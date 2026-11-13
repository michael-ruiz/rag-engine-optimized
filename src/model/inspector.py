"""Layer inspection, tensor logging and activation capture via PyTorch forward hooks."""
from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from src.utils.device import DeviceManager

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-0.8B"


@dataclass(frozen=True)
class ParamInfo:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    numel: int
    bytes: int


def load_model(model_id: str, device: str = "auto", dtype: torch.dtype | None = None):
    """Load a causal LM and tokenizer onto `device` ("auto" = cuda -> mps -> cpu).

    `dtype` defaults to the device's safe dtype from DeviceManager.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dm = DeviceManager.detect(device)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype or dm.dtype, device_map=dm.device)
    except OSError as e:
        raise RuntimeError(f"Could not load model '{model_id}': {e}") from e
    return model.eval(), tokenizer


def resolve_module(model: nn.Module, name: str) -> nn.Module:
    """Return the submodule called `name`, or raise KeyError with close matches."""
    modules = dict(model.named_modules())
    if name not in modules:
        close = difflib.get_close_matches(name, modules.keys(), n=5)
        raise KeyError(f"Module '{name}' not found. Close matches: {close}")
    return modules[name]


class ModelInspector:
    """Inspect parameters and capture intermediate activations of a model.

    Pass an already-loaded `model` (e.g. in tests), or let it load `model_id`.
    Use as a context manager to guarantee hooks are removed.
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        tokenizer=None,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        dtype: torch.dtype | None = None,
    ) -> None:
        if model is None:
            model, tokenizer = load_model(model_id, device, dtype)
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.cache: dict[str, torch.Tensor] = {}
        self._handles: list[RemovableHandle] = []

    # ---- parameter inspection -------------------------------------------------
    def scan(self, name_filter: str | None = None) -> list[ParamInfo]:
        """List parameters (optionally only those whose name contains `name_filter`)."""
        return [
            ParamInfo(n, tuple(p.shape), p.dtype, p.numel(), p.numel() * p.element_size())
            for n, p in self.model.named_parameters()
            if name_filter is None or name_filter in n
        ]

    def print_summary(self, name_filter: str | None = None) -> list[ParamInfo]:
        rows = self.scan(name_filter)
        for r in rows:
            print(f"{r.name:<60} {str(r.shape):<18} {str(r.dtype):<15} {r.bytes / 2**20:9.2f} MB")
        total = sum(r.bytes for r in rows)
        print(f"\n{len(rows)} tensors | {sum(r.numel for r in rows) / 1e6:.2f}M params | {total / 2**20:.2f} MB")
        return rows

    # ---- activation capture ---------------------------------------------------
    def register_hooks(self, targets: list[str]) -> None:
        """Register forward hooks that store each target module's output in `self.cache`."""
        modules = [(t, resolve_module(self.model, t)) for t in targets]  # validate all before hooking any
        for name, module in modules:
            self._handles.append(module.register_forward_hook(self._make_hook(name)))
            logger.info("Hooked %s (%s)", name, type(module).__name__)

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            # Attention blocks return tuples (hidden, weights, ...); MLPs/Linears return a tensor.
            out = output[0] if isinstance(output, (tuple, list)) else output
            self.cache[name] = out.detach().float().cpu()

        return hook

    @torch.no_grad()
    def run(self, prompt: str) -> torch.Tensor:
        """Tokenize `prompt`, run one forward pass, return logits. Hooks fill `self.cache`."""
        if self.tokenizer is None:
            raise RuntimeError("No tokenizer available; pass one or use forward() with input_ids.")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        return self.model(**inputs).logits

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids.to(self.model.device)).logits

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def save_cache(self, path: str | Path) -> None:
        torch.save(self.cache, path)

    def __enter__(self) -> ModelInspector:
        return self

    def __exit__(self, *exc) -> None:
        self.remove_hooks()
