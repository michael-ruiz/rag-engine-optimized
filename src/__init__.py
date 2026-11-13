"""patch-rag: RAG engine with surgical weight-space interventions."""
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "default_config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config file (defaults to config/default_config.yaml)."""
    with open(path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f) or {}
