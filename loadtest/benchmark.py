"""Benchmark suite: starts the gateway with different settings and load-tests each.

    python loadtest/benchmark.py              # full run (~2 minutes)
    python loadtest/benchmark.py --quick      # fewer requests

Each scenario runs in a fresh gateway process (new SQLite file, empty cache) on
port 8091, using loadtest/bench.yaml (fake providers with fixed latency). Results
are printed as a Markdown table and saved to loadtest/results.md.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import tempfile
import time
from argparse import Namespace
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from loadtest import run  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PORT = 8091
OFF = {"PII_ENABLED": "false", "CACHE_ENABLED": "false"}

# name, what it shows, env overrides, load args, uvicorn workers
SCENARIOS = [
    ("Gateway overhead (all features off)", "Raw FastAPI + routing + logging cost, instant provider",
     OFF, dict(model="instant", repeat_ratio=0, pii_ratio=0, stream_ratio=0), 1),
    ("+ PII redaction", "Presidio runs on every request (30% contain PII)",
     {"CACHE_ENABLED": "false"}, dict(model="instant", repeat_ratio=0, pii_ratio=0.3, stream_ratio=0), 1),
    ("+ PII redaction, 4 workers", "Same, with 4 uvicorn processes",
     {"CACHE_ENABLED": "false"}, dict(model="instant", repeat_ratio=0, pii_ratio=0.3, stream_ratio=0), 4),
    ("50 ms provider, no cache", "Typical path: all features on except cache",
     {"CACHE_ENABLED": "false"}, dict(model="realistic", repeat_ratio=0.8, pii_ratio=0.3, stream_ratio=0.1), 1),
    ("50 ms provider, cache on", "Same traffic; 80% of non-PII prompts come from a pool of 20",
     {}, dict(model="realistic", repeat_ratio=0.8, pii_ratio=0.3, stream_ratio=0.1), 1),
    ("Flaky provider (30% failures)", "Retries + fallback to a backup provider",
     {"CACHE_ENABLED": "false"}, dict(model="flaky", repeat_ratio=0, pii_ratio=0.3, stream_ratio=0), 1),
]


def start_gateway(env_overrides: dict, workers: int, db: str) -> subprocess.Popen:
    env = {**os.environ, "GATEWAY_CONFIG": str(ROOT / "loadtest/bench.yaml"), "DATABASE_PATH": db,
           "OTEL_EXPORTER_OTLP_ENDPOINT": "", "RETRY_BACKOFF_SECONDS": "0.05", **env_overrides}
    cmd = [sys.executable, "-m", "uvicorn", "gateway.main:create_app", "--factory", "--port", str(PORT),
           "--workers", str(workers), "--log-level", "warning", "--no-access-log"]
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env)
    for _ in range(240):
        try:
            if httpx.get(f"http://127.0.0.1:{PORT}/healthz", timeout=1).status_code == 200:
                time.sleep(1 if workers == 1 else 5)  # let every worker finish loading spaCy
                return proc
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    proc.kill()
    raise RuntimeError("gateway did not start")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--processes", type=int, default=8, help="load-generator processes")
    ap.add_argument("--only", help="run only scenarios whose name contains this text")
    ap.add_argument("--no-save", action="store_true", help="don't overwrite loadtest/results.md")
    args = ap.parse_args()
    n = 1000 if args.quick else 5000

    rows = []
    for name, desc, env, load, workers in SCENARIOS:
        if args.only and args.only.lower() not in name.lower():
            continue
        with tempfile.TemporaryDirectory() as tmp:
            proc = start_gateway(env, workers, f"{tmp}/bench.db")
            try:
                # short warm-up so first-request costs don't skew results
                run(Namespace(url=f"http://127.0.0.1:{PORT}", key="gw-bench-key", requests=200,
                              concurrency=8, seed=999, processes=1, **load))
                r = run(Namespace(url=f"http://127.0.0.1:{PORT}", key="gw-bench-key", requests=n,
                                  concurrency=args.concurrency, seed=1, processes=args.processes, **load))
            finally:
                proc.terminate()
                proc.wait()
        hits = r["cache"].get("exact", 0) + r["cache"].get("semantic", 0)
        rows.append((name, desc, r, hits))
        print(f"done: {name}: {r['throughput_rps']} req/s, p50 {r['latency_ms'][50]} ms", file=sys.stderr)

    lines = [
        f"Machine: {platform.machine()} {platform.system()}, Python {platform.python_version()}. "
        f"{n} requests per scenario, concurrency {args.concurrency} spread over {args.processes} load-generator "
        f"processes on the same machine.",
        "",
        "| Scenario | Throughput (req/s) | p50 ms | p95 ms | p99 ms | Success | Cache hits |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, _, r, hits in rows:
        lat = r["latency_ms"]
        lines.append(f"| {name} | {r['throughput_rps']} | {lat[50]} | {lat[95]} | {lat[99]} | "
                     f"{r['success_rate']:.1%} | {hits / r['requests']:.0%} |")
    lines += ["", "What each scenario shows:", ""] + [f"- **{name}**: {desc}" for name, desc, _, _ in rows]
    out = "\n".join(lines)
    print(out)
    if not args.no_save:
        (ROOT / "loadtest/results.md").write_text(out + "\n")


if __name__ == "__main__":
    main()
