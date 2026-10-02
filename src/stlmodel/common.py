"""Reproducibility and artifact helpers shared across the pipeline."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FORMAT_VERSION = 1


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def device_for(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA is unavailable. Install a CUDA-enabled PyTorch build or use --device cpu."
        )
    return torch.device(name)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, kind: str | None = None) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported artifact version in {path}")
    if kind and state.get("kind") != kind:
        raise ValueError(f"Expected {kind} artifact, got {state.get('kind')!r}")
    return state
