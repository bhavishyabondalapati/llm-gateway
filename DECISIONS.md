# Design decisions

A log of the choices made while building the gateway, and why. Newest
learnings are folded into the relevant section rather than appended.

## Stack

| Choice | Why | Alternatives considered |
|---|---|---|
| **FastAPI + uvicorn** | Async (many concurrent upstream calls on one process), Pydantic validation for the OpenAI schema, built-in streaming responses. | Flask (sync), Starlette alone (less validation). |
| **httpx** for upstream calls | Async, streaming, one client reused per provider (connection pooling). | aiohttp (fine, but httpx also gives us the test client). |
| **Raw HTTP for OpenAI/Anthropic, not their SDKs** | Keeps the gateway thin and makes retry/fallback logic ours. The adapters are ~80 lines each. | `openai` / `anthropic` SDKs (they add their own retries, which would stack with ours). |
| **SQLite** for the request log | Zero setup, survives restarts, enough for one node. Writes run in a worker thread so the event loop isn't blocked. | Postgres/ClickHouse (needed for multiple replicas, see "Limits"). |
| **YAML for teams/models/providers, env for secrets** | Routing config is reviewable and committable; secrets never are. | Everything in env vars (unreadable for routing chains). |
| **Python 3.11 in a conda env** | Matches the rest of my projects; all deps have wheels for Apple Silicon. | |

## API compatibility

- The endpoint accepts the OpenAI chat-completions body and **passes unknown
  fields through** (`extra="allow"`), so new OpenAI params don't break clients.
- **Errors use OpenAI's shape** (`{"error": {"message", "type", "code"}}`) and
  status codes (401 bad key, 404 `model_not_found`, 429 `rate_limit_exceeded`
  and 429 `insufficient_quota` for budgets). The OpenAI SDK maps these to the
  right exception classes. FastAPI's default 422 is converted to 400.
- **Streaming** (`stream: true`) is supported, including
  `stream_options.include_usage`, because most chat UIs stream.
- Gateway metadata goes in **response headers** (`x-gateway-cache`,
  `x-gateway-provider`, `x-gateway-cost-usd`, `x-gateway-redactions`,
  `x-ratelimit-*`) so the response body stays exactly OpenAI-shaped.

## PII redaction (Presidio)

- **Presidio AnalyzerEngine finds spans; our own code replaces them** with
  numbered placeholders (`<PERSON_1>`). presidio-anonymizer was not used
  because we need *reversible*, *numbered* placeholders that are *consistent
  across all messages* in a conversation (same name → same placeholder).
  The mapping lives only in memory for the duration of the request.
- **spaCy `en_core_web_md`** (~40 MB). Tested `sm` first: it tagged "Email John
  Smith" as a PERSON (eating the verb), which mangled restored text. `md` got
  the boundaries right and was also faster in a quick test. `lg` (~400 MB,
  Presidio's default) was not needed for this and would bloat the Docker image.
- **Entity allow-list** (PERSON, EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD,
  US_SSN, IP_ADDRESS, IBAN_CODE, LOCATION). Presidio's full set flagged a
  phone number as `UK_NHS` and fragments of emails as `URL`. DATE_TIME and NRP
  were excluded: redacting dates breaks most scheduling prompts.
- **Score threshold 0.4.** At 0.5, phone numbers without a nearby word like
  "phone" (Presidio scores those 0.4) were missed.
- **Plausibility filter after Presidio.** The load test found decimals like
  `0.651593` tagged as PHONE_NUMBER and once as LOCATION. Rule: PERSON/LOCATION
  must contain a letter; phone numbers must not be a bare decimal and need ≥ 7
  digits. Cheap, and removed every false positive seen in the load-test mix.
- **Overlaps**: keep the higher-scoring span, then the longer one.
- **Streaming restoration** holds back text after an unclosed `<` (up to 32
  chars) so a placeholder split across chunks (`<PER` + `SON_1>`) is still
  restored. A literal `<` in normal text is released as soon as it's clearly
  not a placeholder.
- Redaction runs in a **thread** (`asyncio.to_thread`) behind a lock, because
  spaCy is CPU-bound and not guaranteed thread-safe. This makes PII the main
  per-process throughput limit (see benchmarks); scale with more uvicorn workers.
- Placeholders are applied to **all roles** (system, assistant history too),
  since PII in history leaks just as much.
- Known limit: if the model rewrites a placeholder (e.g. `<Person 1>`), it
  won't be restored. Real models preserve `<TYPE_N>` tokens well in practice.

## Auth, rate limits, budgets

- **Keys stored as SHA-256 hashes** in the YAML, compared in constant time.
  Plain SHA-256 (not bcrypt) is fine because keys are long random strings, not
  human passwords. `scripts/hash_key.py --new` generates one.
- **Token bucket** per team for requests/minute: allows short bursts, smooth
  average. Returns `retry-after`.
- **Monthly token budget** per team, summed from the SQLite log (cached in
  memory and incremented on each write). Checked *before* the call, so
  concurrent in-flight requests can overshoot the budget slightly; a strict
  version would reserve `max_tokens` up front.
- **Cache hits cost 0 tokens** against the budget (no provider was called),
  and the avoided cost is recorded as `saved_usd`.

## Caching

- **Cache key is computed on the redacted request**, and entries store the
  redacted response. Two consequences, both deliberate:
  1. The cache never contains raw PII.
  2. "Email John at john@x" and "Email Jane at jane@y" share one entry, and
     each caller gets *their own* values restored.
- **Scoped per team**: one team can never be served another team's answer.
- Key includes model and sampling params (temperature, max_tokens, tools...);
  excludes `stream` and `user`, which don't change the answer.
- **Cached by default; opt out per request** with `Cache-Control: no-cache` or
  `x-gateway-cache: off`. A product team that relies on randomness at
  temperature > 0 should send the header.
- **Semantic cache embedder: feature hashing** (word uni/bigrams + char
  trigrams → 1024-dim vector) instead of a neural model, so everything runs
  offline with no model download. It catches casing, punctuation and small
  rewordings, not true paraphrases. `SentenceTransformerEmbedder` (MPS on
  Apple Silicon) is included for real semantic matching.
- **Threshold 0.96**, picked with `scripts/eval_semantic_threshold.py` (10
  should-hit and 10 should-miss pairs). 0.96 is the lowest threshold with zero
  false hits, catching 8/10 rewordings; at 0.92–0.95 a long prompt with one
  meaningful word changed ("…churn" vs "…costs", similarity 0.959) was served
  the wrong answer. A wrong answer is worse than a miss, so precision wins.
  The set is tiny; a real deployment should label pairs from its own traffic.
- **Semantic hits must have identical numbers and placeholder sets.** The load
  test showed "fact about the number 12" being served the answer for "13":
  embeddings, even neural ones, blur numbers. Numbers and placeholders are
  now part of the semantic scope, which also makes lookups cheaper (entries
  are indexed by scope).
- In-memory LRU with TTL (default 1 h, 10k entries). Linear similarity scan
  within a scope is fine at this size; a vector index (FAISS, Redis VSS) would
  be needed for millions of entries.
- Only `stop`/`length` responses are cached (not errors or tool calls).

## Routing, retries, fallback

- Each model maps to an **ordered chain of provider targets** in YAML.
  Real providers without an API key are skipped, so the same config works
  offline (fake providers) and online (real first, fake as last resort).
- Per target: up to `1 + MAX_RETRIES` attempts with **exponential backoff +
  jitter**, then the next target.
- Error classes: 429/5xx/timeouts → retry then fall back; 401/403 → skip
  straight to the next provider (our key is bad, retrying won't help);
  other 4xx → return to the client immediately (the request itself is wrong,
  every provider would reject it).
- **Streaming falls back only before the first chunk.** After that the
  response is committed; a mid-stream failure ends the stream with an error
  event.

## Observability

- **OpenTelemetry with the GenAI semantic conventions**: one CLIENT span per
  provider attempt named `chat {model}` with `gen_ai.operation.name`,
  `gen_ai.provider.name` (plus the older `gen_ai.system`),
  `gen_ai.request.*`, `gen_ai.response.*`, `gen_ai.usage.input_tokens/
  output_tokens`, and `error.type` on failures. A SERVER root span carries
  gateway attributes (team, cache result, cost, redaction count, attempts).
- **No prompt/response content in spans** (the spec makes content opt-in;
  here it could contain PII).
- We create our **own TracerProvider instead of the global one** so each test
  can attach an in-memory exporter.
- Export via **OTLP/HTTP** to Jaeger in Docker Compose. Verified locally by
  pointing the exporter at a small Python OTLP receiver and decoding the
  protobuf payload (Docker isn't installed on the dev machine).

## Dashboard

- A single static HTML page with **inline SVG charts and no CDN**, so it
  works offline and has no build step. It polls `/dashboard/api/stats`
  every 5 s.
- Time buckets scale with the window (1 min up to 1 h, 15 min up to 24 h,
  2 h beyond) and **empty buckets are kept**, so gaps in traffic are visible
  rather than lines drawn across them.
- Light and dark mode via CSS variables; colors from a CVD-checked palette.
- Optional `DASHBOARD_TOKEN`; open by default for local dev. Only counts are
  stored and shown, never content.

## Testing

- Everything runs offline against `FakeProvider`, which echoes the last user
  message. Echoing is what makes PII tests meaningful: placeholders sent
  upstream come back and must be restored. It also records `last_request`,
  so tests can assert that **no raw PII left the gateway**.
- One spaCy model is loaded per test session (it takes seconds).

## Not verified / limits

- **Docker Compose was written but not run**: Docker isn't installed on the
  machine this was built on. The image's Jaeger tag and ports come from the
  Jaeger getting-started docs.
- **Real providers were not called** (no keys used during development). The
  OpenAI and Anthropic adapters follow the public APIs but are untested
  against live endpoints.
- **Single node.** Rate-limit buckets, cache and budget counters are per
  process. With several workers, each has its own (so effective limits are
  multiplied by the worker count). Multi-node needs Redis for limits/cache and
  a shared database.
- Token counts from fake providers are estimates (~4 chars/token).
