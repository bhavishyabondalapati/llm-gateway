"""End-to-end tests through the HTTP API (fake providers, offline)."""

import json

from tests.conftest import AUTH_A, AUTH_B, AUTH_TINY, chat

PII_TEXT = "Email John Smith at john.smith@acme.com about the invoice."


def provider(client, name):
    return client.app.state.gw.providers[name]


def sse_events(text):
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: {")]


# --- auth -----------------------------------------------------------------

def test_missing_or_bad_key_is_401(client):
    assert client.post("/v1/chat/completions", json=chat("hi")).status_code == 401
    r = client.post("/v1/chat/completions", json=chat("hi"), headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_api_key"


def test_invalid_body_is_openai_style_400(client):
    r = client.post("/v1/chat/completions", json={"model": "m-fast", "messages": []}, headers=AUTH_A)
    assert r.status_code == 400 and "error" in r.json()


# --- happy path + PII -----------------------------------------------------

def test_completion_redacts_upstream_and_restores_response(client):
    r = client.post("/v1/chat/completions", json=chat(PII_TEXT), headers=AUTH_A)
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == f"[fast] You said: {PII_TEXT}"

    sent = provider(client, "fast").last_request.messages[0].content
    assert "John Smith" not in sent and "john.smith@acme.com" not in sent
    assert "<PERSON_1>" in sent and "<EMAIL_ADDRESS_1>" in sent
    assert r.headers["x-gateway-redactions"] == "2"


def test_usage_and_cost_are_reported(client):
    r = client.post("/v1/chat/completions", json=chat("hello world"), headers=AUTH_A)
    usage = r.json()["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"] > 0
    expected = (usage["prompt_tokens"] * 1.0 + usage["completion_tokens"] * 2.0) / 1e6
    assert float(r.headers["x-gateway-cost-usd"]) == round(expected, 8)


def test_models_endpoint(client):
    ids = [m["id"] for m in client.get("/v1/models", headers=AUTH_A).json()["data"]]
    assert "m-fast" in ids


# --- caching --------------------------------------------------------------

def test_exact_cache_hit_skips_provider(client):
    fast = provider(client, "fast")
    client.post("/v1/chat/completions", json=chat("cache me"), headers=AUTH_A)
    calls = fast.calls
    r = client.post("/v1/chat/completions", json=chat("cache me"), headers=AUTH_A)
    assert r.headers["x-gateway-cache"] == "exact" and fast.calls == calls
    assert r.json()["usage"]["total_tokens"] == 0


def test_cache_holds_redacted_text_and_restores_per_request(client):
    """Two people, same template: second hits the cache but gets *their own* PII back."""
    client.post("/v1/chat/completions", json=chat(PII_TEXT), headers=AUTH_A)
    other = "Email Jane Doe at jane@foo.org about the invoice."
    r = client.post("/v1/chat/completions", json=chat(other), headers=AUTH_A)
    assert r.headers["x-gateway-cache"] == "exact"
    content = r.json()["choices"][0]["message"]["content"]
    assert "Jane Doe" in content and "jane@foo.org" in content and "John" not in content


def test_semantic_cache_hit(client):
    client.post("/v1/chat/completions", json=chat("What is the capital of France?"), headers=AUTH_A)
    r = client.post("/v1/chat/completions", json=chat("what is the capital of france"), headers=AUTH_A)
    assert r.headers["x-gateway-cache"] == "semantic"


def test_cache_is_per_team(client):
    client.post("/v1/chat/completions", json=chat("team secret question"), headers=AUTH_A)
    r = client.post("/v1/chat/completions", json=chat("team secret question"), headers=AUTH_B)
    assert r.headers["x-gateway-cache"] == "miss"


def test_cache_bypass_header(client):
    client.post("/v1/chat/completions", json=chat("bypass me"), headers=AUTH_A)
    r = client.post("/v1/chat/completions", json=chat("bypass me"), headers={**AUTH_A, "Cache-Control": "no-cache"})
    assert r.headers["x-gateway-cache"] == "bypass"


# --- limits ---------------------------------------------------------------

def test_rate_limit_returns_429(client):
    codes = [client.post("/v1/chat/completions", json=chat(f"q{i}"), headers=AUTH_TINY).status_code
             for i in range(4)]
    assert codes[3] == 429
    r = client.post("/v1/chat/completions", json=chat("again"), headers=AUTH_TINY)
    assert r.json()["error"]["code"] == "rate_limit_exceeded" and "retry-after" in r.headers


def test_token_budget_returns_429_insufficient_quota(client):
    long_text = "word " * 200  # ~250 tokens, over the tiny team's 40-token budget
    assert client.post("/v1/chat/completions", json=chat(long_text), headers=AUTH_TINY).status_code == 200
    r = client.post("/v1/chat/completions", json=chat("small"), headers=AUTH_TINY)
    assert r.status_code == 429 and r.json()["error"]["code"] == "insufficient_quota"


# --- routing --------------------------------------------------------------

def test_fallback_to_backup_provider(client):
    r = client.post("/v1/chat/completions", json=chat("hi", model="m-fallback"), headers=AUTH_A)
    assert r.status_code == 200 and r.headers["x-gateway-provider"] == "backup"


def test_all_providers_down_is_502(client):
    r = client.post("/v1/chat/completions", json=chat("hi", model="m-dead"), headers=AUTH_A)
    assert r.status_code == 502 and r.json()["error"]["code"] == "upstream_error"


def test_unknown_model_is_404(client):
    r = client.post("/v1/chat/completions", json=chat("hi", model="nope"), headers=AUTH_A)
    assert r.status_code == 404 and r.json()["error"]["code"] == "model_not_found"


def test_real_provider_without_key_is_skipped(client):
    r = client.post("/v1/chat/completions", json=chat("hi", model="m-nokey"), headers=AUTH_A)
    assert r.status_code == 503


# --- streaming ------------------------------------------------------------

def test_streaming_restores_pii_across_chunks(client):
    body = chat(PII_TEXT, stream=True, stream_options={"include_usage": True})
    with client.stream("POST", "/v1/chat/completions", json=body, headers=AUTH_A) as r:
        assert r.status_code == 200
        raw = "".join(r.iter_text())
    events = sse_events(raw)
    text = "".join(e["choices"][0]["delta"].get("content", "") for e in events if e["choices"])
    assert text == f"[fast] You said: {PII_TEXT}"
    assert raw.rstrip().endswith("data: [DONE]")
    assert events[-1]["usage"]["total_tokens"] > 0
    assert "<PERSON" not in raw


def test_streaming_cache_replay(client):
    client.post("/v1/chat/completions", json=chat("stream cache test"), headers=AUTH_A)
    with client.stream("POST", "/v1/chat/completions", json=chat("stream cache test", stream=True),
                       headers=AUTH_A) as r:
        raw = "".join(r.iter_text())
    assert r.headers["x-gateway-cache"] == "exact"
    assert "[fast] You said: stream cache test" in raw


# --- observability --------------------------------------------------------

def test_traces_follow_genai_conventions(client, spans):
    client.post("/v1/chat/completions", json=chat(PII_TEXT, model="m-fallback", max_tokens=100), headers=AUTH_A)
    finished = spans.get_finished_spans()
    names = [s.name for s in finished]
    assert "POST /v1/chat/completions" in names and "pii.redact" in names

    attempts = [s for s in finished if s.name == "chat m-fallback"]
    assert len(attempts) == 3  # down x2 (1 retry), then backup
    ok = attempts[-1]
    assert ok.attributes["gen_ai.operation.name"] == "chat"
    assert ok.attributes["gen_ai.request.model"] == "m-fallback"
    assert ok.attributes["gen_ai.request.max_tokens"] == 100
    assert ok.attributes["gen_ai.usage.input_tokens"] > 0
    assert ok.attributes["gen_ai.response.finish_reasons"] == ("stop",)
    assert attempts[0].attributes["error.type"] == "503"

    root = next(s for s in finished if s.name == "POST /v1/chat/completions")
    assert all(s.context.trace_id == root.context.trace_id for s in attempts)
    # No prompt content in any span
    assert not any("John Smith" in str(v) for s in finished for v in s.attributes.values())


def test_dashboard_stats(client):
    client.post("/v1/chat/completions", json=chat(PII_TEXT), headers=AUTH_A)
    client.post("/v1/chat/completions", json=chat(PII_TEXT), headers=AUTH_A)
    client.post("/v1/chat/completions", json=chat("hi", model="m-dead"), headers=AUTH_A)
    stats = client.get("/dashboard/api/stats").json()
    t = stats["totals"]
    assert t["requests"] == 3 and t["errors"] == 1 and t["cache_hits_exact"] == 1
    assert t["redactions"] == 4 and t["saved_usd"] > 0
    assert {e["entity_type"] for e in stats["redactions_by_entity"]} == {"PERSON", "EMAIL_ADDRESS"}
    assert client.get("/dashboard").status_code == 200
