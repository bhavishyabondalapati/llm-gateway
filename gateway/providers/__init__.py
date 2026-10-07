"""Build provider instances from config."""

from __future__ import annotations

import os

from gateway.config import GatewayConfig, Settings
from gateway.providers.anthropic import AnthropicProvider
from gateway.providers.base import Provider, ProviderError, ProviderResponse, StreamChunk
from gateway.providers.fake import FakeProvider
from gateway.providers.openai_compat import OpenAICompatProvider

__all__ = ["Provider", "ProviderError", "ProviderResponse", "StreamChunk", "build_providers"]


def build_providers(config: GatewayConfig, settings: Settings) -> dict[str, Provider]:
    """Instantiate every enabled provider. Real providers without an API key are skipped."""
    providers: dict[str, Provider] = {}
    for name, p in config.providers.items():
        if not config.provider_enabled(name):
            continue
        if p.type == "fake":
            providers[name] = FakeProvider(name, p.latency_ms, p.failure_rate, p.seed)
        elif p.type == "openai":
            providers[name] = OpenAICompatProvider(
                name, p.base_url or "https://api.openai.com/v1", os.environ[p.api_key_env],
                settings.provider_timeout_seconds,
            )
        elif p.type == "anthropic":
            providers[name] = AnthropicProvider(
                name, os.environ[p.api_key_env], p.base_url or "https://api.anthropic.com",
                settings.provider_timeout_seconds,
            )
    return providers
