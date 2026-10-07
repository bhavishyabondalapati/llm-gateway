"""Response cache: exact-match first, then semantic (embedding similarity).

Important design points:
- Entries are stored *after* PII redaction, so the cache never holds raw PII. A
  hit is restored with the *current* request's mapping.
- Entries are scoped per team, so one team can never read another team's answers.
- Semantic matches must also share the same model, sampling params and set of
  PII placeholders; only the wording of the conversation may differ.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from gateway.embeddings import Embedder
from gateway.pii import PLACEHOLDER_RE
from gateway.schemas import ChatCompletionRequest, message_text

# Request fields that don't change the answer and so aren't part of the cache key.
_IGNORED_FIELDS = {"stream", "user", "messages", "stream_options"}


@dataclass
class CachedResponse:
    content: str
    finish_reason: str
    model: str
    prompt_tokens: int
    completion_tokens: int


@dataclass
class _Entry:
    value: CachedResponse
    expires_at: float
    scope: str
    vector: np.ndarray | None


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def conversation_text(req: ChatCompletionRequest) -> str:
    return "\n".join(f"{m.role}: {message_text(m)}" for m in req.messages)


class ResponseCache:
    def __init__(
        self,
        embedder: Embedder | None,
        ttl_seconds: int = 3600,
        max_entries: int = 10_000,
        semantic_threshold: float = 0.95,
    ):
        self.embedder = embedder
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self.threshold = semantic_threshold
        self._entries: OrderedDict[str, _Entry] = OrderedDict()

    # --- keys -----------------------------------------------------------
    def exact_key(self, team: str, req: ChatCompletionRequest) -> str:
        params = req.model_dump(exclude=_IGNORED_FIELDS, exclude_none=True)
        msgs = [m.model_dump(exclude_none=True) for m in req.messages]
        return _hash({"team": team, "params": params, "messages": msgs})

    def semantic_scope(self, team: str, req: ChatCompletionRequest) -> str:
        params = req.model_dump(exclude=_IGNORED_FIELDS, exclude_none=True)
        placeholders = sorted(set(PLACEHOLDER_RE.findall(conversation_text(req))))
        return _hash({"team": team, "params": params, "placeholders": placeholders})

    # --- lookups --------------------------------------------------------
    def get_exact(self, key: str) -> CachedResponse | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.expires_at < time.time():
            del self._entries[key]
            return None
        self._entries.move_to_end(key)
        return entry.value

    def embed(self, text: str) -> np.ndarray | None:
        return self.embedder.embed(text) if self.embedder is not None else None

    def get_semantic(self, scope: str, query: np.ndarray | None) -> tuple[CachedResponse, float] | None:
        if query is None:
            return None
        now = time.time()
        candidates = [
            (k, e) for k, e in self._entries.items()
            if e.scope == scope and e.vector is not None and e.expires_at >= now
        ]
        if not candidates:
            return None
        matrix = np.stack([e.vector for _, e in candidates])
        sims = matrix @ query  # vectors are unit-length, so dot product = cosine
        best = int(np.argmax(sims))
        if sims[best] < self.threshold:
            return None
        key, entry = candidates[best]
        self._entries.move_to_end(key)
        return entry.value, float(sims[best])

    # --- writes ---------------------------------------------------------
    def put(self, key: str, scope: str, vector: np.ndarray | None, value: CachedResponse) -> None:
        self._entries[key] = _Entry(value, time.time() + self.ttl, scope, vector)
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)  # evict least recently used

    def __len__(self) -> int:
        return len(self._entries)
