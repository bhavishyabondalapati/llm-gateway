"""Per-team API keys.

The config stores only the SHA-256 hash of each key, so the YAML file is safe to
commit. Clients send `Authorization: Bearer <key>` (what the OpenAI SDK does).
"""

from __future__ import annotations

import hashlib
import hmac

from gateway.config import TeamConfig


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class KeyStore:
    def __init__(self, teams: list[TeamConfig]):
        self._by_hash = {t.api_key_sha256: t for t in teams}

    def lookup(self, authorization: str | None) -> TeamConfig | None:
        if not authorization or not authorization.lower().startswith("bearer "):
            return None
        digest = hash_key(authorization[7:].strip())
        for stored, team in self._by_hash.items():
            if hmac.compare_digest(stored, digest):  # constant-time comparison
                return team
        return None
