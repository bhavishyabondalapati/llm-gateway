"""Per-team rate limiting with a token bucket.

Each team has a bucket holding up to `requests_per_minute` tokens. Every request
takes one token; tokens refill continuously at rpm/60 per second. This allows
short bursts while keeping the average rate under the limit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class _Bucket:
    tokens: float
    updated: float


class RateLimiter:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self._buckets: dict[str, _Bucket] = {}

    def check(self, team: str, rpm: int) -> tuple[bool, int, float]:
        """Try to take a token. Returns (allowed, remaining, retry_after_seconds)."""
        now = self.clock()
        b = self._buckets.setdefault(team, _Bucket(tokens=float(rpm), updated=now))
        rate = rpm / 60.0
        b.tokens = min(float(rpm), b.tokens + (now - b.updated) * rate)
        b.updated = now
        if b.tokens >= 1:
            b.tokens -= 1
            return True, int(b.tokens), 0.0
        return False, 0, (1 - b.tokens) / rate
