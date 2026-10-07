# LLM Gateway

A self-hosted gateway that sits between your apps and LLM providers. Apps call
one **OpenAI-compatible `/v1/chat/completions` endpoint**; the gateway strips
personal data (PII) before anything leaves your network and puts it back in the
answer. It also enforces per-team keys, rate limits and token budgets, caches
repeated questions, retries and fails over between providers, and records the
cost, latency and a trace for every request.

It runs **fully offline** with fake providers (that's how it's tested). Add an
OpenAI or Anthropic key to `.env` and real providers are used first.

```
client (OpenAI SDK) ──► /v1/chat/completions
                         │ 1. auth (team API key)
                         │ 2. rate limit (token bucket) + monthly token budget
                         │ 3. PII redaction  "John Smith" → <PERSON_1>    (Presidio)
                         │ 4. cache lookup   exact → semantic             (redacted text only)
                         │ 5. provider call  retries + fallback chain     (OpenAI / Anthropic / fake)
                         │ 6. PII restoration <PERSON_1> → "John Smith"   (works while streaming)
                         │ 7. record cost + latency (SQLite), emit trace (OpenTelemetry GenAI)
                         ▼
                      dashboard  /dashboard
```

## Tools and libraries

| Tool | Why |
|---|---|
| **FastAPI + uvicorn** | Async web framework: one process can wait on many slow LLM calls at once. Pydantic validates the OpenAI request format for free. |
| **httpx** | Async HTTP client for the upstream providers, with streaming. Also powers the test client. |
| **Microsoft Presidio (analyzer)** | Proven PII detector (names, emails, phones, cards, SSNs, IPs, IBANs, locations) combining regex rules, checksums and NER. |
| **spaCy `en_core_web_md`** | The NER model Presidio uses to find names and places. Medium size: much better name boundaries than `sm`, a tenth the size of `lg`. |
| **NumPy** | Vector math for the semantic cache (cosine similarity). |
| **OpenTelemetry SDK + OTLP exporter** | Vendor-neutral tracing. Uses the GenAI semantic conventions, so any compatible backend understands the spans. |
| **SQLite** | Request log for cost/latency/budgets. Zero setup, file-based. |
| **pydantic-settings, PyYAML, python-dotenv** | Settings from env/.env; routing and teams from a readable YAML file. |
| **pytest + pytest-asyncio** | Test runner; async tests for the router. |
| **Docker Compose + Jaeger** | One command to run the gateway with a trace viewer. |

## File structure

```
llm-gateway/
├── gateway/
│   ├── main.py              App factory and the request pipeline for /v1/chat/completions
│   ├── config.py            Settings (env) and YAML config models (teams, providers, routes, prices)
│   ├── schemas.py           OpenAI-compatible request/response models, token estimate
│   ├── auth.py              Team API keys, stored as SHA-256 hashes
│   ├── ratelimit.py         Token-bucket rate limiter (requests/minute per team)
│   ├── pii.py               Presidio redaction, placeholder restoration, streaming restorer
│   ├── cache.py             Exact + semantic response cache (per team, PII-free, LRU + TTL)
│   ├── embeddings.py        Offline hashing embedder; optional sentence-transformers embedder
│   ├── router.py            Fallback chain with retries, backoff and per-attempt spans
│   ├── pricing.py           Cost from tokens and the price table
│   ├── store.py             SQLite request log, budget usage, dashboard aggregates
│   ├── telemetry.py         OpenTelemetry setup and GenAI semantic-convention attributes
│   ├── dashboard.py         Dashboard page and its JSON stats API
│   ├── static/dashboard.html  The dashboard (vanilla JS + inline SVG charts, no CDN)
│   └── providers/
│       ├── base.py          Provider interface and error classification (retry? fall back?)
│       ├── fake.py          Offline fake LLM with configurable latency and failure rate
│       ├── openai_compat.py OpenAI (and any OpenAI-compatible API) adapter
│       └── anthropic.py     Anthropic Messages API adapter (translates formats)
├── config/gateway.yaml      Teams, providers, model fallback chains, prices
├── tests/
│   ├── conftest.py          Fixtures: test config, shared spaCy model, in-memory span exporter
│   ├── test_api.py          End-to-end HTTP tests (PII, cache, limits, fallback, streaming, traces)
│   ├── test_pii.py          Redaction, restoration, split-placeholder streaming
│   ├── test_cache.py        Exact/semantic hits, team isolation, guards, TTL, LRU
│   └── test_router.py       Retries, fallback rules, streaming fallback, rate limiter
├── loadtest/
│   ├── loadtest.py          Async load generator (throughput, latency percentiles, cache hits)
│   ├── benchmark.py         Runs the gateway in several configurations and load-tests each
│   ├── bench.yaml           Config used by the benchmark (fixed-latency fake providers)
│   └── results.md           Latest benchmark output
├── scripts/
│   ├── hash_key.py          Generate a team API key and its hash
│   ├── eval_semantic_threshold.py  Labeled pairs used to pick the semantic-cache threshold
│   └── bench_pii.py         Redaction latency vs message length
├── Dockerfile               Gateway image (non-root, healthcheck)
├── docker-compose.yml       Gateway + Jaeger (+ optional load-test service)
├── DECISIONS.md             Every design choice and the reason for it
├── requirements.txt         Pinned dependencies
├── pytest.ini               Test settings
└── .env.example             Optional settings and provider keys (placeholders only)
```

## Setup

```bash
conda create -n llm-gateway python=3.11 -y
```

```bash
conda activate llm-gateway
```

```bash
pip install -r requirements.txt
```

`requirements.txt` includes the spaCy model download (~40 MB). Optional: copy
`.env.example` to `.env` and add `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` to use
real providers.

## How to run

Start the gateway:

```bash
uvicorn gateway.main:create_app --factory --port 8080
```

Open the dashboard at http://localhost:8080/dashboard, then send a request
(the demo keys are listed at the top of `config/gateway.yaml`):

```bash
curl -s http://localhost:8080/v1/chat/completions -H "Authorization: Bearer gw-demo-alpha" -H "Content-Type: application/json" -d '{"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Email John Smith at john.smith@acme.com about the invoice."}]}'
```

The fake provider echoes what it received, so you'll see the original name and
email restored. Add `-i` to see the gateway headers (`x-gateway-cache`,
`x-gateway-cost-usd`, `x-gateway-redactions`...). Run the same command twice and
the second one is a cache hit.

It works with the official OpenAI SDK too (`pip install openai`; the gateway
itself doesn't need it). Streaming, `models.list()` and error classes like
`RateLimitError` all behave as they do against OpenAI:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="gw-demo-alpha")
client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
```

Things to try:
- `"model": "flaky-model"`: its first provider fails 50% of the time, so you'll see retries and fallback (`x-gateway-provider: fake-backup`).
- Key `gw-demo-tiny`: 5 requests/minute and a 500-token monthly budget, so you'll quickly get `429`s.
- `"stream": true`: placeholders are restored even when split across chunks.
- Header `Cache-Control: no-cache` skips the cache.

**With Docker** (gateway + Jaeger trace viewer at http://localhost:16686):

```bash
docker compose up --build
```

**Run the tests:**

```bash
python -m pytest
```

**Load test** (with the gateway running):

```bash
python loadtest/loadtest.py --requests 2000 --concurrency 50
```

**Benchmarks** (starts its own gateway processes on port 8091):

```bash
python loadtest/benchmark.py
```

## Load test and benchmarks

All numbers below come from my laptop (Apple M4 Pro, 24 GB, Python 3.11). The
load generator runs on the **same machine**, so treat them as relative, not
absolute. Fake providers have fixed latency, so these measure the **gateway**,
not an LLM.

### Load test

`loadtest/loadtest.py` sends a configurable mix of prompts: some repeated
(cache-friendly), some with names/emails/phones (exercises Presidio), some
streamed. It reports throughput, latency percentiles, status codes, cache hits
and which provider served each request:

```
$ python loadtest/loadtest.py --requests 1000 --concurrency 32 --processes 4
1000 requests @ concurrency 32 (4 procs) in 2.79s -> 358.8 req/s, success 62.6%
latency ms  p50 153.3  p90 186.4  p95 198.9  p99 257.0   (ttfb p50 152.0)
p50 by cache: hit 1.6 ms, miss 158.0 ms
statuses {200: 626, 429: 374}
cache    {'miss': 529, 'exact': 97, '-': 374}
provider {'fake-primary': 529, 'cache': 97, '-': 374}
```

The 374 `429`s are the rate limiter doing its job: the demo team `alpha`
allows 600 requests/minute, so a 1,000-request burst gets its first ~600 (plus
a little refill) and the rest are rejected. Use `--key` with your own team, or
raise `requests_per_minute`, to load-test without limits. Use `--processes 4`
or more above ~400 req/s, because one Python client process can't generate
more load than that.

### Benchmark results

`python loadtest/benchmark.py`: 5,000 requests per scenario, 64 concurrent
connections across 8 client processes, a fresh gateway process per scenario.

| Scenario | Throughput (req/s) | p50 ms | p95 ms | p99 ms | Success | Cache hits | p50 hit / miss ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gateway overhead (all features off) | 3,085 | 19.8 | 31.2 | 35.8 | 100% | 0% | - |
| + PII redaction | 208 | 305.1 | 337.6 | 353.4 | 100% | 0% | - |
| + PII redaction, 4 workers | 712 | 98.2 | 115.7 | 135.9 | 100% | 0% | - |
| 50 ms provider, no cache | 341 | 74.1 | 390.9 | 414.2 | 100% | 0% | - |
| 50 ms provider, cache on | 362 | 2.6 | 433.8 | 461.4 | 100% | 56% | 1.6 / 394.3 |
| Flaky provider (30% failures) | 186 | 315.2 | 520.2 | 591.6 | 100% | 0% | - |

PII redaction cost vs message length (`python scripts/bench_pii.py`, one
message, memo disabled):

| Characters | p50 ms | p95 ms |
|---:|---:|---:|
| 126 | 4.6 | 5.4 |
| 418 | 13.0 | 13.5 |
| 1,513 | 43.1 | 44.5 |
| 5,893 | 168.6 | 171.4 |

### What the numbers say

- **The gateway itself is cheap.** With features off, one worker serves ~3,000
  req/s at ~20 ms p50 under 64 concurrent connections (ApacheBench measured
  ~3,400 req/s for the same path). Compared with an LLM call that takes
  hundreds of milliseconds, the overhead is negligible.
- **PII redaction is the bottleneck.** Presidio/spaCy costs ~5 ms of CPU per
  short message and grows linearly with length (~170 ms for a 6k-char RAG
  prompt). It is CPU-bound, so one process tops out around 200 req/s of *new*
  text. **More workers scale it almost linearly** (4 workers: 3.4×). The
  memo makes repeated text (system prompts, chat history) nearly free.
- **Provider latency doesn't limit throughput, CPU does.** Async I/O means
  waiting 50 ms on a provider costs no CPU, so the 50 ms scenarios are limited
  by Presidio, not the provider.
- **Cache hits are ~250× faster than misses** (1.6 ms vs 394 ms p50 under load),
  and they cost $0. The overall p95 doesn't improve because the misses still
  queue behind Presidio; the win shows up in the hit latency and the bill.
- **Fallback works: 100% success with 30% upstream failures.** The price is
  tail latency (p99 ~590 ms) from retries and backoff.

### Bugs the load tests found

The load tests were useful well beyond timing numbers:

1. **Semantic cache false hits**: prompts differing only by a number ("fact
   about 12" vs "13") were served each other's answers. Fixed by requiring
   identical numbers (and placeholders) for a semantic match.
2. **PII false positives**: decimals like `0.651593` were redacted as phone
   numbers. Fixed with a plausibility filter.
3. **Cache hits as slow as misses** (~180 ms): they were queueing behind
   Presidio on a shared thread pool and lock. Fixed with a dedicated Presidio
   thread and inline memo hits: 1.6 ms.
4. **The benchmark was measuring itself**: a single Python client process maxed
   out at ~350 req/s even against `/healthz`. Fixed with a multi-process load
   generator, cross-checked with ApacheBench.

## How it was built

1. **Scaffolding**: conda env, `.gitignore`, config models (env settings plus YAML for teams/models/providers), OpenAI request/response schemas.
2. **PII redaction**: tried Presidio with spaCy `sm`, switched to `md` after it mis-bounded names; wrote reversible numbered placeholders and a streaming restorer.
3. **Providers and routing**: provider interface, fake provider, OpenAI and Anthropic adapters, a router with retries, backoff and a fallback chain, plus one GenAI-convention span per attempt.
4. **Cache**: exact-match key on the redacted request, plus a semantic cache using an offline hashing embedder.
5. **The endpoint**: auth → rate limit → budget → redact → cache → route → restore → record, with streaming and OpenAI-style errors.
6. **Dashboard**: stats API over the SQLite log and a dependency-free HTML page.
7. **Tests**: 50+ offline tests, including "no raw PII reached the provider".
8. **Load testing found real bugs**: semantic-cache false hits on numbers and PII false positives on decimals (both fixed, see above).
9. **Threshold tuning**: labeled pairs to pick the semantic threshold (0.96).
10. **Docker Compose with Jaeger**; OTLP export verified against a local receiver.
11. **Benchmarking and performance fixes**: found the load generator was the bottleneck and made it multi-process; memoized Presidio; moved it to a dedicated thread so cache hits stay fast.
12. **This README and `DECISIONS.md`**.

## Key concepts

- **API gateway pattern**: one choke point for cross-cutting concerns (auth, limits, privacy, cost, observability), so every app doesn't reimplement them. Being OpenAI-compatible means switching an app over is just a `base_url` change.
- **Reversible pseudonymization**: replace PII with placeholders, keep the mapping in memory for the request only, restore it in the output. The LLM provider never sees the real values, yet the user gets a natural answer. It also has to survive streaming, where a placeholder can be split across chunks.
- **Caching LLM responses**: an exact cache is safe and cheap. A semantic cache trades correctness for hit rate, and its threshold is a precision/recall decision. I picked it with labeled data and added hard guards (numbers, placeholders) where embeddings are known to be weak. Caching the *redacted* text means the cache never stores PII and different users can share entries.
- **Token bucket rate limiting**: allows bursts but caps the average rate; simple to implement with two numbers per team.
- **Retries vs fallback**: retry transient errors (429/5xx/timeouts) with exponential backoff and jitter; fall back to another provider when retries run out; never retry a request the provider rejected as invalid (4xx). Streaming can only fall back before the first byte is sent.
- **Observability with OpenTelemetry**: traces made of spans (parent/child, trace IDs). The GenAI semantic conventions standardize attribute names like `gen_ai.usage.input_tokens`, so tools can chart token usage across vendors. Prompt content is kept out of traces for privacy.
- **Async I/O vs CPU-bound work**: the event loop handles thousands of waiting HTTP calls, but Presidio/spaCy is CPU-bound and runs in a thread. That makes it the throughput limit per process, so you scale with more worker processes (see benchmarks).
- **Per-process state**: rate limits, cache and budget counters live in memory, which is fine for one process. With multiple replicas they would move to Redis and a shared database.
- **Testing with fakes**: a fake provider that echoes its input makes the tests deterministic and offline, and lets them assert exactly what left the gateway.
