"""Text embeddings for the semantic cache.

`HashingEmbedder` is the default: it needs no model download, so everything runs
offline. It hashes word unigrams/bigrams and character trigrams into a fixed-size
vector (the "hashing trick"). It catches near-duplicates (punctuation, casing,
small rewordings) but not true paraphrases.

`SentenceTransformerEmbedder` is optional (pip install sentence-transformers) and
captures meaning, e.g. "What's the capital of France?" ~ "Name France's capital city".
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

_WORD_RE = re.compile(r"[a-z0-9<>_]+")


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> np.ndarray: ...


class HashingEmbedder:
    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _bucket(self, token: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "little")
        return h % self.dim, (1.0 if (h >> 63) & 1 else -1.0)

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        words = _WORD_RE.findall(text.lower())
        features = list(words)
        features += [f"{a} {b}" for a, b in zip(words, words[1:])]
        joined = " ".join(words)
        features += [f"#{joined[i:i + 3]}" for i in range(max(0, len(joined) - 2))]
        for feat in features:
            idx, sign = self._bucket(feat)
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        import torch
        from sentence_transformers import SentenceTransformer

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True).astype(np.float32)
