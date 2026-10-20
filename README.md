# LLM Gateway

A self-hosted, OpenAI-compatible gateway between applications and LLM providers. It redacts personal data before requests leave the network and restores it in responses, enforces per-team API keys, rate limits, and token budgets, caches repeated requests, retries and fails over between providers, and records cost, latency, and an OpenTelemetry trace for every request.

It runs fully offline with fake providers, which is how it is tested. With an OpenAI or Anthropic key in `.env`, real providers are used first.

## Features

- **OpenAI-compatible API**: `/v1/chat/completions` and `/v1/models`, including streaming and OpenAI-style errors; existing clients switch by changing `base_url`.
- **Reversible PII redaction** with Microsoft Presidio and spaCy: values are replaced with numbered placeholders (`John Smith` → `<PERSON_1>`) and restored in the response, including across streamed chunk boundaries.
- **Team auth and limits**: SHA-256-hashed API keys, a per-team token-bucket rate limiter, and monthly token budgets.
- **Response cache**: exact and semantic matching on redacted text, isolated per team, with LRU eviction and TTL.
- **Routing**: per-model fallback chains with retries, exponential backoff, and jitter across OpenAI, Anthropic, and fake providers.
- **Observability**: SQLite request log for cost and latency, OpenTelemetry spans using the GenAI semantic conventions, and a built-in dashboard.
- **Docker Compose** setup with a Jaeger trace viewer.

## Architecture

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

- The PII mapping is held in memory for the duration of a single request; providers never see the original values, and the cache never stores them.
- Transient errors (429, 5xx, timeouts) are retried and then fall back to the next provider; invalid requests (other 4xx) are not retried. Streaming requests can fall back only before the first byte is sent.
- Presidio is CPU-bound and runs on a dedicated thread, with a memo for repeated text, so cache hits and other async I/O are not blocked behind it.
- Prompt content is excluded from traces.

| Component | Role |
|---|---|
| FastAPI, uvicorn | Async API; Pydantic validation of the OpenAI request format |
| httpx | Async, streaming client for upstream providers |
| Presidio analyzer | PII detection (names, emails, phones, cards, SSNs, IPs, IBANs, locations) via rules, checksums, and NER |
| spaCy `en_core_web_md` | NER model for Presidio; more accurate name boundaries than `sm`, a tenth the size of `lg` |
| NumPy | Cosine similarity for the semantic cache |
| OpenTelemetry SDK, OTLP exporter | Vendor-neutral tracing with GenAI semantic conventions |
| SQLite | Request log, budget usage, dashboard aggregates |
| pydantic-settings, PyYAML, python-dotenv | Environment settings and YAML config for teams, providers, routes, and prices |
| pytest, pytest-asyncio | Test suite |
| Docker Compose, Jaeger | Containerized deployment with a trace viewer |

## Results

All numbers were measured on an Apple M4 Pro (24 GB, Python 3.11) with the load generator on the same machine, so they are relative rather than absolute. Fake providers have fixed latency, so the numbers measure the gateway, not an LLM. Latest output: [loadtest/results.md](loadtest/results.md).

### Benchmarks

`python loadtest/benchmark.py`: 5,000 requests per scenario, 64 concurrent connections across 8 client processes, and a fresh gateway process per scenario.

| Scenario | Throughput (req/s) | p50 ms | p95 ms | p99 ms | Success | Cache hits | p50 hit / miss ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gateway overhead (all features off) | 3,085 | 19.8 | 31.2 | 35.8 | 100% | 0% | - |
| + PII redaction | 208 | 305.1 | 337.6 | 353.4 | 100% | 0% | - |
| + PII redaction, 4 workers | 712 | 98.2 | 115.7 | 135.9 | 100% | 0% | - |
| 50 ms provider, no cache | 341 | 74.1 | 390.9 | 414.2 | 100% | 0% | - |
| 50 ms provider, cache on | 362 | 2.6 | 433.8 | 461.4 | 100% | 56% | 1.6 / 394.3 |
| Flaky provider (30% failures) | 186 | 315.2 | 520.2 | 591.6 | 100% | 0% | - |

PII redaction cost by message length (`python scripts/bench_pii.py`, single message, memo disabled):

| Characters | p50 ms | p95 ms |
|---:|---:|---:|
| 126 | 4.6 | 5.4 |
| 418 | 13.0 | 13.5 |
| 1,513 | 43.1 | 44.5 |
| 5,893 | 168.6 | 171.4 |

Findings:

- **Gateway overhead is small.** With features off, one worker serves about 3,000 req/s at ~20 ms p50 under 64 concurrent connections (ApacheBench measured ~3,400 req/s on the same path), negligible next to an LLM call.
- **PII redaction is the bottleneck.** Presidio/spaCy costs ~5 ms of CPU per short message and grows linearly with length (~170 ms for a 6k-character prompt), capping one process at around 200 req/s of new text. Additional workers scale nearly linearly (4 workers: 3.4×), and the memo makes repeated text such as system prompts and chat history nearly free.
- **Throughput is CPU-bound, not provider-bound.** Waiting on a 50 ms provider costs no CPU under async I/O, so those scenarios are limited by Presidio.
- **Cache hits are about 250× faster than misses** (1.6 ms vs 394 ms p50 under load) and cost nothing. Overall p95 does not improve because misses still queue behind Presidio.
- **Fallback keeps success at 100% with 30% upstream failures**, at the cost of tail latency (p99 ~590 ms) from retries and backoff.

### Issues found through load testing

1. **Semantic-cache false hits**: prompts differing only by a number ("fact about 12" vs "13") were served each other's answers. Semantic matches now require identical numbers and placeholders.
2. **PII false positives**: decimals such as `0.651593` were redacted as phone numbers. Fixed with a plausibility filter.
3. **Slow cache hits** (~180 ms): hits were queueing behind Presidio on a shared thread pool and lock. A dedicated Presidio thread and inline memo hits reduced them to 1.6 ms.
4. **Client-bound benchmark**: a single Python client process saturated at ~350 req/s even against `/healthz`. The load generator is now multi-process and was cross-checked with ApacheBench.

## Quickstart

```bash
conda create -n llm-gateway python=3.11 -y
conda activate llm-gateway
pip install -r requirements.txt          # includes the spaCy model (~40 MB)
cp .env.example .env                     # optional: OPENAI_API_KEY / ANTHROPIC_API_KEY
uvicorn gateway.main:create_app --factory --port 8080
```

The dashboard is at http://localhost:8080/dashboard. Demo team keys are listed at the top of `config/gateway.yaml`.

```bash
curl -s http://localhost:8080/v1/chat/completions -H "Authorization: Bearer gw-demo-alpha" -H "Content-Type: application/json" -d '{"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Email John Smith at john.smith@acme.com about the invoice."}]}'
```

The fake provider echoes its (redacted) input, so the response shows the original name and email restored. Add `-i` to see gateway headers such as `x-gateway-cache`, `x-gateway-cost-usd`, and `x-gateway-redactions`; repeating the request produces a cache hit.

The official OpenAI SDK works unchanged, including streaming, `models.list()`, and error classes such as `RateLimitError`:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="gw-demo-alpha")
client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
```

Demo scenarios:

- `"model": "flaky-model"`: the first provider fails 50% of the time, showing retries and fallback (`x-gateway-provider: fake-backup`).
- Key `gw-demo-tiny`: 5 requests/minute and a 500-token monthly budget, producing `429` responses.
- `"stream": true`: placeholders are restored even when split across chunks.
- Header `Cache-Control: no-cache` bypasses the cache.

Generate a new team key and hash:

```bash
python scripts/hash_key.py
```

### Docker

```bash
docker compose up --build
```

Starts the gateway and Jaeger (trace viewer at http://localhost:16686).

### Load testing

With the gateway running:

```bash
python loadtest/loadtest.py --requests 2000 --concurrency 50
```

The load generator mixes repeated prompts, prompts containing PII, and streamed requests, and reports throughput, latency percentiles, status codes, cache hits, and provider usage. Example:

```
$ python loadtest/loadtest.py --requests 1000 --concurrency 32 --processes 4
1000 requests @ concurrency 32 (4 procs) in 2.79s -> 358.8 req/s, success 62.6%
latency ms  p50 153.3  p90 186.4  p95 198.9  p99 257.0   (ttfb p50 152.0)
p50 by cache: hit 1.6 ms, miss 158.0 ms
statuses {200: 626, 429: 374}
cache    {'miss': 529, 'exact': 97, '-': 374}
provider {'fake-primary': 529, 'cache': 97, '-': 374}
```

The 429 responses come from the rate limiter: team `alpha` allows 600 requests/minute. Use `--key` with a team that has higher limits, or raise `requests_per_minute`, to test without limiting. Use `--processes 4` or more above ~400 req/s, the limit of a single Python client process.

Run the full benchmark suite (starts its own gateway processes on port 8091):

```bash
python loadtest/benchmark.py
```

## Running tests

```bash
python -m pytest
```

53 offline tests:

| File | Coverage |
|---|---|
| `test_api.py` | End-to-end HTTP: PII, cache, limits, fallback, streaming, traces, and that no raw PII reaches the provider |
| `test_pii.py` | Redaction, restoration, placeholders split across stream chunks |
| `test_cache.py` | Exact and semantic hits, team isolation, number/placeholder guards, TTL, LRU |
| `test_router.py` | Retries, fallback rules, streaming fallback, rate limiter |

## Project structure

```
llm-gateway/
├── gateway/
│   ├── main.py              App factory and the /v1/chat/completions pipeline
│   ├── config.py            Environment settings and YAML config models
│   ├── schemas.py           OpenAI-compatible request/response models, token estimate
│   ├── auth.py              Team API keys stored as SHA-256 hashes
│   ├── ratelimit.py         Per-team token-bucket rate limiter
│   ├── pii.py               Presidio redaction, placeholder restoration, streaming restorer
│   ├── cache.py             Exact and semantic response cache (per team, LRU + TTL)
│   ├── embeddings.py        Offline hashing embedder; optional sentence-transformers embedder
│   ├── router.py            Fallback chain with retries, backoff, and per-attempt spans
│   ├── pricing.py           Cost from token counts and the price table
│   ├── store.py             SQLite request log, budget usage, dashboard aggregates
│   ├── telemetry.py         OpenTelemetry setup and GenAI attributes
│   ├── dashboard.py         Dashboard page and JSON stats API
│   ├── static/dashboard.html  Dashboard (vanilla JS, inline SVG charts, no CDN)
│   └── providers/
│       ├── base.py          Provider interface and error classification
│       ├── fake.py          Offline fake provider with configurable latency and failure rate
│       ├── openai_compat.py OpenAI and OpenAI-compatible API adapter
│       └── anthropic.py     Anthropic Messages API adapter
├── config/gateway.yaml      Teams, providers, model fallback chains, prices
├── tests/                   pytest suite (see Running tests)
├── loadtest/
│   ├── loadtest.py          Async multi-process load generator
│   ├── benchmark.py         Runs and load-tests several gateway configurations
│   ├── bench.yaml           Benchmark config (fixed-latency fake providers)
│   └── results.md           Latest benchmark output
├── scripts/
│   ├── hash_key.py          Generates a team API key and its hash
│   ├── eval_semantic_threshold.py  Labeled pairs for choosing the semantic-cache threshold
│   └── bench_pii.py         Redaction latency vs message length
├── Dockerfile               Non-root image with healthcheck
├── docker-compose.yml       Gateway, Jaeger, optional load-test service
├── DECISIONS.md             Design decision log
├── requirements.txt         Pinned dependencies
├── pytest.ini
└── .env.example             Optional settings and provider keys (placeholders only)
```

## Design decisions

See [DECISIONS.md](DECISIONS.md) for the full log. Highlights:

- **Cache on redacted text.** The cache never stores PII, and entries can be shared across users within a team.
- **Semantic threshold chosen from labeled data.** `scripts/eval_semantic_threshold.py` selects 0.96; hard guards on numbers and placeholders cover cases where embeddings are known to be weak.
- **spaCy `md` over `sm`.** The small model mis-bounded names in early testing.
- **Token bucket rate limiting** allows short bursts while capping the average rate, with two numbers of state per team.
- **Fake providers that echo their input** make tests deterministic and let them assert exactly what left the gateway.

## Limitations and future work

- **Single node.** Rate-limit buckets, cache, and budget counters are per process; with several workers, effective limits multiply by the worker count. Multi-node deployment needs Redis for limits and cache and a shared database.
- **Real providers have not been called.** The OpenAI and Anthropic adapters follow the public APIs but are untested against live endpoints.
- **Docker Compose has not been run.** The Jaeger image tag and ports follow Jaeger's getting-started documentation; OTLP export was verified against a local receiver.
- Token counts from fake providers are estimates (~4 characters per token).
