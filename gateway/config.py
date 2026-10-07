"""Configuration.

Two sources:
- Environment variables / .env (secrets and runtime knobs) -> `Settings`
- A YAML file (teams, models, providers, prices)            -> `GatewayConfig`
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gateway_config: str = "config/gateway.yaml"
    database_path: str = "data/gateway.db"

    # PII
    pii_enabled: bool = True
    pii_score_threshold: float = 0.4
    spacy_model: str = "en_core_web_md"

    # Caching
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    cache_max_entries: int = 10_000
    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = 0.96

    # Provider calls
    provider_timeout_seconds: float = 30.0
    max_retries: int = 2  # retries per provider, on top of the first attempt
    retry_backoff_seconds: float = 0.2

    # Telemetry: OTLP endpoint (e.g. http://jaeger:4318). Empty = no export.
    otel_exporter_otlp_endpoint: str = ""
    otel_console_exporter: bool = False
    otel_service_name: str = "llm-gateway"

    # Optional token to protect the dashboard. Empty = open (dev mode).
    dashboard_token: str = ""


class TeamConfig(BaseModel):
    name: str
    api_key_sha256: str
    requests_per_minute: int = 60
    monthly_token_budget: int = 1_000_000


class ProviderConfig(BaseModel):
    type: Literal["fake", "openai", "anthropic"]
    base_url: str | None = None
    api_key_env: str | None = None  # name of the env var holding the key
    # Fake-provider knobs (useful for testing fallback and load tests)
    latency_ms: float = 20.0
    failure_rate: float = 0.0
    seed: int | None = None


class RouteTarget(BaseModel):
    provider: str
    model: str | None = None  # upstream model name; defaults to requested model


class ModelRoute(BaseModel):
    targets: list[RouteTarget]


class Price(BaseModel):
    input_per_million: float
    output_per_million: float


class GatewayConfig(BaseModel):
    teams: list[TeamConfig] = Field(default_factory=list)
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    models: dict[str, ModelRoute] = Field(default_factory=dict)
    pricing: dict[str, Price] = Field(default_factory=dict)
    default_price: Price = Price(input_per_million=1.0, output_per_million=2.0)

    def provider_enabled(self, name: str) -> bool:
        """A real provider is only enabled when its API key is in the environment."""
        p = self.providers.get(name)
        if p is None:
            return False
        if p.type == "fake":
            return True
        return bool(p.api_key_env and os.environ.get(p.api_key_env))


def load_gateway_config(path: str | Path) -> GatewayConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return GatewayConfig.model_validate(raw)


@lru_cache
def get_settings() -> Settings:
    return Settings()
