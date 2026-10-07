"""Async load test for the gateway.

Sends N requests with a fixed concurrency and reports throughput, latency
percentiles, status codes and cache hits. Uses only httpx, so no extra install.

    python loadtest/loadtest.py --requests 2000 --concurrency 50
    python loadtest/loadtest.py --requests 5000 --concurrency 64 --processes 4   # for >400 req/s
    python loadtest/loadtest.py --repeat-ratio 0.5 --pii-ratio 0.3 --stream-ratio 0.1

Prompt mix:
  --repeat-ratio  share of prompts drawn from a small pool of 20 (cache-friendly)
  --pii-ratio     share of prompts containing names/emails/phones (exercises Presidio)
  --stream-ratio  share of requests sent with stream=true
"""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import httpx

NAMES = ["John Smith", "Priya Patel", "Maria Garcia", "Wei Chen", "Ahmed Hassan", "Olivia Brown"]
TOPICS = ["the invoice", "the quarterly report", "our meeting", "the contract renewal", "the shipping delay"]
POOL = [f"Explain concept number {i} of distributed systems in two sentences." for i in range(20)]


def make_prompt(rng: random.Random, i: int, repeat_ratio: float, pii_ratio: float) -> str:
    if rng.random() < pii_ratio:
        name = rng.choice(NAMES)
        email = name.lower().replace(" ", ".") + "@example.com"
        return f"Draft a short note to {name} ({email}, 617-555-01{rng.randint(10, 99)}) about {rng.choice(TOPICS)}. #{i}"
    if rng.random() < repeat_ratio:
        return rng.choice(POOL)
    return f"Give me one interesting fact about the number {i} and {rng.random():.6f}."


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


async def _run_shard(args, shard: int, shards: int) -> dict:
    """Run every `shards`-th job starting at `shard`; return raw measurements."""
    rng = random.Random(args.seed)
    jobs = [
        (make_prompt(rng, i, args.repeat_ratio, args.pii_ratio), rng.random() < args.stream_ratio)
        for i in range(args.requests)
    ][shard::shards]
    concurrency = max(1, args.concurrency // shards)
    raw = {"latencies": [], "ttfb": [], "by_cache": {}, "statuses": Counter(), "cache": Counter(),
           "providers": Counter()}
    queue: asyncio.Queue = asyncio.Queue()
    for j in jobs:
        queue.put_nowait(j)

    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(base_url=args.url, timeout=60, limits=limits,
                                 headers={"Authorization": f"Bearer {args.key}"}) as client:

        async def worker():
            while True:
                try:
                    prompt, stream = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                body = {"model": args.model, "messages": [{"role": "user", "content": prompt}], "stream": stream}
                t0 = time.perf_counter()
                try:
                    async with client.stream("POST", "/v1/chat/completions", json=body) as r:
                        first = None
                        async for _ in r.aiter_bytes():
                            first = first or time.perf_counter()
                        raw["statuses"][r.status_code] += 1
                        raw["cache"][r.headers.get("x-gateway-cache", "-")] += 1
                        raw["providers"][r.headers.get("x-gateway-provider", "-")] += 1
                        if r.status_code == 200:
                            ms = (time.perf_counter() - t0) * 1000
                            raw["latencies"].append(ms)
                            hit = r.headers.get("x-gateway-cache") in ("exact", "semantic")
                            raw["by_cache"].setdefault("hit" if hit else "miss", []).append(ms)
                            raw["ttfb"].append(((first or time.perf_counter()) - t0) * 1000)
                except httpx.HTTPError as e:
                    raw["statuses"][type(e).__name__] += 1

        raw["start"] = time.time()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        raw["end"] = time.time()
    return raw


def _shard_entry(args_dict: dict, shard: int, shards: int) -> dict:
    return asyncio.run(_run_shard(argparse.Namespace(**args_dict), shard, shards))


def run(args) -> dict:
    """Run the load test, split over `args.processes` client processes.

    One Python process tops out around 350-400 req/s because the client itself
    becomes the bottleneck (measured against the trivial /healthz endpoint), so
    several processes are needed to measure a fast gateway.
    """
    processes = getattr(args, "processes", 1)
    if processes == 1:
        shards = [asyncio.run(_run_shard(args, 0, 1))]
    else:
        with ProcessPoolExecutor(processes, mp_context=multiprocessing.get_context("spawn")) as pool:
            shards = list(pool.map(_shard_entry, [vars(args)] * processes, range(processes), [processes] * processes))

    latencies = [x for s in shards for x in s["latencies"]]
    ttfb = [x for s in shards for x in s["ttfb"]]
    merged = {k: sum((s[k] for s in shards), Counter()) for k in ("statuses", "cache", "providers")}
    elapsed = max(s["end"] for s in shards) - min(s["start"] for s in shards)
    by_cache = {k: [x for s in shards for x in s["by_cache"].get(k, [])] for k in ("hit", "miss")}
    return {
        "requests": args.requests,
        "concurrency": args.concurrency,
        "processes": processes,
        "elapsed_s": round(elapsed, 2),
        "throughput_rps": round(args.requests / elapsed, 1),
        "success_rate": round(merged["statuses"].get(200, 0) / args.requests, 4),
        "latency_ms": {p: round(pct(latencies, p), 1) for p in (50, 90, 95, 99)},
        "ttfb_ms_p50": round(pct(ttfb, 50), 1),
        "p50_ms_cache_hit": round(pct(by_cache["hit"], 50), 1),
        "p50_ms_cache_miss": round(pct(by_cache["miss"], 50), 1),
        "statuses": dict(merged["statuses"]),
        "cache": dict(merged["cache"]),
        "providers": dict(merged["providers"]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--key", default="gw-demo-alpha")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--requests", type=int, default=1000)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--repeat-ratio", type=float, default=0.3)
    ap.add_argument("--pii-ratio", type=float, default=0.3)
    ap.add_argument("--stream-ratio", type=float, default=0.1)
    ap.add_argument("--processes", type=int, default=1, help="client processes (use 4+ for >400 req/s)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", action="store_true", help="print the result as JSON only")
    args = ap.parse_args()
    result = run(args)
    if args.json:
        print(json.dumps(result))
        return
    lat = result["latency_ms"]
    print(f"{result['requests']} requests @ concurrency {result['concurrency']} ({result['processes']} procs) in {result['elapsed_s']}s "
          f"-> {result['throughput_rps']} req/s, success {result['success_rate']:.1%}")
    print(f"latency ms  p50 {lat[50]}  p90 {lat[90]}  p95 {lat[95]}  p99 {lat[99]}   (ttfb p50 {result['ttfb_ms_p50']})")
    if result["cache"].get("exact") or result["cache"].get("semantic"):
        print(f"p50 by cache: hit {result['p50_ms_cache_hit']} ms, miss {result['p50_ms_cache_miss']} ms")
    print(f"statuses {result['statuses']}")
    print(f"cache    {result['cache']}")
    print(f"provider {result['providers']}")


if __name__ == "__main__":
    main()
