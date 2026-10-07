"""SQLite request log: per-request cost/latency/cache/redaction records, budget
usage, and the aggregates behind the dashboard.

sqlite3 is synchronous, so calls run in a worker thread (asyncio.to_thread) to
keep the event loop free. Only counts are stored here, never prompt content.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id TEXT PRIMARY KEY,
    ts REAL NOT NULL,
    team TEXT NOT NULL,
    model TEXT NOT NULL,
    provider TEXT,
    status INTEGER NOT NULL,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    saved_usd REAL NOT NULL DEFAULT 0,    -- estimated cost avoided by a cache hit
    latency_ms REAL NOT NULL,
    cache TEXT NOT NULL DEFAULT 'miss',   -- miss | exact | semantic | bypass
    redactions INTEGER NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0,
    stream INTEGER NOT NULL DEFAULT 0,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_requests_team_ts ON requests(team, ts);
CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts);
CREATE TABLE IF NOT EXISTS redactions (
    request_id TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    count INTEGER NOT NULL
);
"""


@dataclass
class RequestRecord:
    id: str
    team: str
    model: str
    status: int
    latency_ms: float
    provider: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    saved_usd: float = 0.0
    cache: str = "miss"
    attempts: int = 0
    stream: bool = False
    error: str | None = None
    redaction_counts: Counter = field(default_factory=Counter)
    ts: float = field(default_factory=time.time)


def month_start(ts: float | None = None) -> float:
    d = datetime.fromtimestamp(ts or time.time(), tz=timezone.utc)
    return datetime(d.year, d.month, 1, tzinfo=timezone.utc).timestamp()


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


class Store:
    def __init__(self, path: str):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.lock = threading.Lock()
        # team -> (month_start, tokens used). Avoids a SUM query on every request.
        self._usage: dict[str, tuple[float, int]] = {}

    # --- budgets --------------------------------------------------------
    def _tokens_this_month(self, team: str) -> int:
        start = month_start()
        cached = self._usage.get(team)
        if cached and cached[0] == start:
            return cached[1]
        with self.lock:
            row = self.conn.execute(
                "SELECT COALESCE(SUM(prompt_tokens + completion_tokens), 0) FROM requests WHERE team=? AND ts>=?",
                (team, start),
            ).fetchone()
        self._usage[team] = (start, row[0])
        return row[0]

    async def tokens_this_month(self, team: str) -> int:
        return await asyncio.to_thread(self._tokens_this_month, team)

    # --- writes ---------------------------------------------------------
    def record_sync(self, r: RequestRecord) -> None:
        with self.lock:
            self.conn.execute(
                "INSERT INTO requests VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (r.id, r.ts, r.team, r.model, r.provider, r.status, r.prompt_tokens, r.completion_tokens,
                 r.cost_usd, r.saved_usd, r.latency_ms, r.cache, sum(r.redaction_counts.values()), r.attempts,
                 int(r.stream), r.error),
            )
            self.conn.executemany(
                "INSERT INTO redactions VALUES (?,?,?)",
                [(r.id, etype, n) for etype, n in r.redaction_counts.items()],
            )
            self.conn.commit()
        start, used = self._usage.get(r.team, (None, None))
        if start == month_start(r.ts):
            self._usage[r.team] = (start, used + r.prompt_tokens + r.completion_tokens)

    async def record(self, r: RequestRecord) -> None:
        await asyncio.to_thread(self.record_sync, r)

    # --- dashboard ------------------------------------------------------
    def _stats(self, window_seconds: int) -> dict:
        since = time.time() - window_seconds
        with self.lock:
            rows = [dict(r) for r in self.conn.execute(
                "SELECT * FROM requests WHERE ts>=? ORDER BY ts", (since,)
            ).fetchall()]
            entity_rows = self.conn.execute(
                "SELECT d.entity_type, SUM(d.count) AS n FROM redactions d JOIN requests r ON r.id=d.request_id "
                "WHERE r.ts>=? GROUP BY d.entity_type ORDER BY n DESC", (since,),
            ).fetchall()

        ok = [r for r in rows if r["status"] == 200]
        hits = [r for r in ok if r["cache"] in ("exact", "semantic")]
        latencies = [r["latency_ms"] for r in ok]
        upstream_latencies = [r["latency_ms"] for r in ok if r["cache"] not in ("exact", "semantic")]

        def group(key: str) -> list[dict]:
            out: dict[str, dict] = {}
            for r in rows:
                g = out.setdefault(r[key] or "-", {key: r[key] or "-", "requests": 0, "cost_usd": 0.0,
                                                   "tokens": 0, "errors": 0, "_lat": []})
                g["requests"] += 1
                g["cost_usd"] += r["cost_usd"]
                g["tokens"] += r["prompt_tokens"] + r["completion_tokens"]
                g["errors"] += r["status"] != 200
                if r["status"] == 200:
                    g["_lat"].append(r["latency_ms"])
            for g in out.values():
                g["p95_latency_ms"] = round(percentile(g.pop("_lat"), 95), 1)
                g["cost_usd"] = round(g["cost_usd"], 6)
            return sorted(out.values(), key=lambda g: -g["requests"])

        # Time series with a bucket size that suits the window; empty buckets are kept
        # (as zeros) so charts show gaps in traffic honestly.
        step = 60 if window_seconds <= 3600 else 900 if window_seconds <= 86400 else 7200
        now = time.time()
        buckets: dict[int, dict] = {}
        for ts in range(int(since // step * step), int(now // step * step) + 1, step):
            buckets[ts] = {"requests": 0, "cost_usd": 0.0, "cache_hits": 0, "redactions": 0, "errors": 0, "_lat": []}
        for r in rows:
            b = buckets.setdefault(int(r["ts"] // step * step), {"requests": 0, "cost_usd": 0.0, "cache_hits": 0,
                                                                 "redactions": 0, "errors": 0, "_lat": []})
            b["requests"] += 1
            b["cost_usd"] += r["cost_usd"]
            b["cache_hits"] += r["cache"] in ("exact", "semantic")
            b["redactions"] += r["redactions"]
            b["errors"] += r["status"] != 200
            if r["status"] == 200:
                b["_lat"].append(r["latency_ms"])
        series = []
        for ts in sorted(buckets):
            b = buckets[ts]
            lat = b.pop("_lat")
            series.append({"ts": ts, **b, "cost_usd": round(b["cost_usd"], 6),
                           "p50_latency_ms": round(percentile(lat, 50), 1) if lat else None,
                           "p95_latency_ms": round(percentile(lat, 95), 1) if lat else None})

        return {
            "window_seconds": window_seconds,
            "bucket_seconds": step,
            "totals": {
                "requests": len(rows),
                "errors": len(rows) - len(ok),
                "cost_usd": round(sum(r["cost_usd"] for r in rows), 6),
                "saved_usd": round(sum(r["saved_usd"] for r in rows), 6),
                "tokens": sum(r["prompt_tokens"] + r["completion_tokens"] for r in rows),
                "cache_hits": len(hits),
                "cache_hits_exact": sum(r["cache"] == "exact" for r in hits),
                "cache_hits_semantic": sum(r["cache"] == "semantic" for r in hits),
                "cache_hit_rate": round(len(hits) / len(ok), 4) if ok else 0.0,
                "redactions": sum(r["redactions"] for r in rows),
                "requests_with_pii": sum(r["redactions"] > 0 for r in rows),
                "p50_latency_ms": round(percentile(latencies, 50), 1),
                "p95_latency_ms": round(percentile(latencies, 95), 1),
                "p50_upstream_latency_ms": round(percentile(upstream_latencies, 50), 1),
                "fallbacks": sum(r["attempts"] > 1 for r in ok),
            },
            "by_team": group("team"),
            "by_model": group("model"),
            "by_provider": group("provider"),
            "redactions_by_entity": [{"entity_type": e["entity_type"], "count": e["n"]} for e in entity_rows],
            "series": series,
            "recent": [
                {k: r[k] for k in ("ts", "team", "model", "provider", "status", "cache", "latency_ms",
                                   "cost_usd", "prompt_tokens", "completion_tokens", "redactions", "attempts")}
                for r in rows[-25:][::-1]
            ],
        }

    async def stats(self, window_seconds: int = 3600) -> dict:
        return await asyncio.to_thread(self._stats, window_seconds)

    def close(self) -> None:
        self.conn.close()
