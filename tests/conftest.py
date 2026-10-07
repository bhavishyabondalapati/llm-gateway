"""Shared fixtures. Everything runs offline against fake providers."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from gateway.auth import hash_key
from gateway.config import GatewayConfig, Settings
from gateway.main import create_app
from gateway.pii import PIIRedactor

KEY_A, KEY_B, KEY_TINY = "test-key-a", "test-key-b", "test-key-tiny"
AUTH_A = {"Authorization": f"Bearer {KEY_A}"}
AUTH_B = {"Authorization": f"Bearer {KEY_B}"}
AUTH_TINY = {"Authorization": f"Bearer {KEY_TINY}"}


def make_config(**overrides) -> GatewayConfig:
    raw = {
        "teams": [
            {"name": "a", "api_key_sha256": hash_key(KEY_A), "requests_per_minute": 1000, "monthly_token_budget": 10**9},
            {"name": "b", "api_key_sha256": hash_key(KEY_B), "requests_per_minute": 1000, "monthly_token_budget": 10**9},
            {"name": "tiny", "api_key_sha256": hash_key(KEY_TINY), "requests_per_minute": 3, "monthly_token_budget": 40},
        ],
        "providers": {
            "fast": {"type": "fake", "latency_ms": 1},
            "backup": {"type": "fake", "latency_ms": 1},
            "down": {"type": "fake", "latency_ms": 1, "failure_rate": 1.0},
            "openai": {"type": "openai", "api_key_env": "TEST_UNSET_OPENAI_KEY"},  # no key -> skipped
        },
        "models": {
            "m-fast": {"targets": [{"provider": "openai"}, {"provider": "fast"}]},
            "m-fallback": {"targets": [{"provider": "down"}, {"provider": "backup"}]},
            "m-dead": {"targets": [{"provider": "down"}]},
            "m-nokey": {"targets": [{"provider": "openai"}]},
        },
        "pricing": {"m-fast": {"input_per_million": 1.0, "output_per_million": 2.0}},
    }
    raw.update(overrides)
    return GatewayConfig.model_validate(raw)


@pytest.fixture(scope="session")
def redactor() -> PIIRedactor:
    """Loading spaCy takes seconds, so share one redactor across all tests."""
    r = PIIRedactor()
    r.analyzer  # noqa: B018 - force load
    return r


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        database_path=str(tmp_path / "test.db"),
        retry_backoff_seconds=0.0,
        max_retries=1,
        otel_exporter_otlp_endpoint="",
    )


@pytest.fixture
def spans() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def client(settings, redactor, spans):
    app = create_app(settings=settings, config=make_config(), span_exporter=spans, redactor=redactor)
    with TestClient(app) as c:
        yield c


def chat(content: str, model: str = "m-fast", **kw) -> dict:
    return {"model": model, "messages": [{"role": "user", "content": content}], **kw}
