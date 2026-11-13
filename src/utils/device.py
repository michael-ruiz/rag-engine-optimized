"""Hardware auto-detection: cuda -> mps (Apple Silicon) -> cpu, with a safe dtype per device."""
from __future__ import annotations

from dataclasses import dataclass

import torch

# bf16 is fast and safe on CUDA and Apple Silicon; many CPU kernels are slow or missing in bf16.
_DTYPES = {"cuda": torch.bfloat16, "mps": torch.bfloat16, "cpu": torch.float32}


def _available(device: str) -> bool:
    if device == "cuda":
        return torch.cuda.is_available()
    if device == "mps":
        return torch.backends.mps.is_available()
    return device == "cpu"


@dataclass(frozen=True)
class DeviceManager:
    device: str
    dtype: torch.dtype

    @classmethod
    def detect(cls, prefer: str | None = "auto") -> DeviceManager:
        """Pick the best available device, or validate an explicit one ("cuda" | "mps" | "cpu")."""
        if prefer in (None, "auto"):
            device = next(d for d in ("cuda", "mps", "cpu") if _available(d))
        elif prefer not in _DTYPES:
            raise ValueError(f"Unknown device '{prefer}'; expected auto, cuda, mps or cpu.")
        elif not _available(prefer):
            raise RuntimeError(f"Device '{prefer}' requested but not available on this machine.")
        else:
            device = prefer
        return cls(device, _DTYPES[device])
