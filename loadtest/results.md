Machine: arm64 Darwin, Python 3.11.17. 5000 requests per scenario, concurrency 64 spread over 8 load-generator processes on the same machine.

| Scenario | Throughput (req/s) | p50 ms | p95 ms | p99 ms | Success | Cache hits | p50 hit / miss ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gateway overhead (all features off) | 3084.6 | 19.8 | 31.2 | 35.8 | 100.0% | 0% | - |
| + PII redaction | 207.6 | 305.1 | 337.6 | 353.4 | 100.0% | 0% | - |
| + PII redaction, 4 workers | 712.2 | 98.2 | 115.7 | 135.9 | 100.0% | 0% | - |
| 50 ms provider, no cache | 340.6 | 74.1 | 390.9 | 414.2 | 100.0% | 0% | - |
| 50 ms provider, cache on | 362.3 | 2.6 | 433.8 | 461.4 | 100.0% | 56% | 1.6 / 394.3 |
| Flaky provider (30% failures) | 185.7 | 315.2 | 520.2 | 591.6 | 100.0% | 0% | - |

What each scenario shows:

- **Gateway overhead (all features off)**: Raw FastAPI + routing + logging cost, instant provider
- **+ PII redaction**: Presidio runs on every request (30% contain PII)
- **+ PII redaction, 4 workers**: Same, with 4 uvicorn processes
- **50 ms provider, no cache**: Typical path: all features on except cache
- **50 ms provider, cache on**: Same traffic; 80% of non-PII prompts come from a pool of 20
- **Flaky provider (30% failures)**: Retries + fallback to a backup provider
