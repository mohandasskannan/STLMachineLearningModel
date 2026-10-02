"""Frozen, CPU-based text embeddings; loaded only when needed."""

from pathlib import Path

import numpy as np

from .common import TEXT_MODEL


class TextEncoder:
    def __init__(self, model_name: str = TEXT_MODEL, revision: str | None = None):
        from huggingface_hub import try_to_load_from_cache
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        cache = str(Path(".cache") / "sentence-transformers")
        # Reuse a complete cached model offline. First use may download it.
        try:
            self.model = SentenceTransformer(
                model_name,
                device="cpu",
                revision=revision,
                cache_folder=cache,
                local_files_only=True,
            )
        except OSError:
            self.model = SentenceTransformer(
                model_name,
                device="cpu",
                revision=revision,
                cache_folder=cache,
            )
        cached_config = try_to_load_from_cache(
            model_name, "config.json", revision=revision or "main", cache_dir=cache
        )
        self.revision = (
            Path(cached_config).parent.name if isinstance(cached_config, str) else revision
        )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        dimension = (
            self.model.get_embedding_dimension()
            if hasattr(self.model, "get_embedding_dimension")
            else self.model.get_sentence_embedding_dimension()
        )
        if dimension != 384:
            raise ValueError("This pipeline requires a 384-dimensional text encoder")

    def encode(self, captions: list[str]) -> np.ndarray:
        if not captions or any(not caption.strip() for caption in captions):
            raise ValueError("Prompts must contain text")
        return self.model.encode(
            captions,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
