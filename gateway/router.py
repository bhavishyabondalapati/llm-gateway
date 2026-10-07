"""Routing with retries and provider fallback.

For a requested model, the config lists an ordered chain of targets, e.g.
    gpt-4o-mini -> [openai, fake-primary, fake-backup]
For each target we try up to (1 + max_retries) times with exponential backoff,
then move on to the next. Errors caused by the client's request (HTTP 400 etc.)
stop immediately, since another provider would reject it too.

Streaming can only fall back *before* the first chunk is sent to the client;
after that the response is committed.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from opentelemetry.trace import SpanKind, Status, StatusCode, Tracer

from gateway import telemetry as tm
from gateway.config import GatewayConfig, Settings
from gateway.providers import Provider, ProviderError, ProviderResponse, StreamChunk
from gateway.schemas import ChatCompletionRequest


class UnknownModelError(Exception):
    pass


class AllProvidersFailedError(Exception):
    def __init__(self, errors: list[str], status: int | None = None, client_error: bool = False):
        super().__init__("; ".join(errors) or "no provider available")
        self.errors = errors
        self.status = status
        self.client_error = client_error


@dataclass
class RouteResult:
    response: ProviderResponse
    provider: str
    attempts: int


@dataclass
class StreamHandle:
    provider: str
    model: str
    attempts: int
    first: StreamChunk
    rest: AsyncIterator[StreamChunk]
    span: object
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None
    errors: list[str] = field(default_factory=list)

    async def chunks(self) -> AsyncIterator[StreamChunk]:
        try:
            self._track(self.first)
            yield self.first
            async for chunk in self.rest:
                self._track(chunk)
                yield chunk
            tm.set_response_attributes(
                self.span, self.model, self.finish_reason, self.prompt_tokens, self.completion_tokens
            )
        except ProviderError as e:  # failure mid-stream: too late to fall back
            self.span.set_status(Status(StatusCode.ERROR, str(e)))
            self.span.set_attribute(tm.ERROR_TYPE, "ProviderError")
            raise
        finally:
            self.span.end()

    def _track(self, c: StreamChunk) -> None:
        if c.prompt_tokens is not None:
            self.prompt_tokens = c.prompt_tokens
        if c.completion_tokens is not None:
            self.completion_tokens = c.completion_tokens
        if c.finish_reason:
            self.finish_reason = c.finish_reason


class Router:
    def __init__(self, config: GatewayConfig, providers: dict[str, Provider], settings: Settings, tracer: Tracer):
        self.config = config
        self.providers = providers
        self.settings = settings
        self.tracer = tracer

    def targets(self, model: str) -> list[tuple[Provider, str]]:
        route = self.config.models.get(model)
        if route is None:
            raise UnknownModelError(model)
        return [
            (self.providers[t.provider], t.model or model)
            for t in route.targets
            if t.provider in self.providers  # skips real providers without API keys
        ]

    async def _backoff(self, attempt: int) -> None:
        delay = self.settings.retry_backoff_seconds * (2**attempt)
        await asyncio.sleep(delay * random.uniform(0.5, 1.0))  # jitter avoids retry stampedes

    def _attempt_span(self, provider: Provider, model: str, req: ChatCompletionRequest, attempt: int):
        span = self.tracer.start_span(f"chat {model}", kind=SpanKind.CLIENT)
        tm.set_request_attributes(span, provider.system, model, req)
        span.set_attribute("gateway.provider", provider.name)
        span.set_attribute("gateway.attempt", attempt)
        return span

    @staticmethod
    def _fail_span(span, e: ProviderError) -> None:
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.set_attribute(tm.ERROR_TYPE, str(e.status) if e.status else "ProviderError")
        span.end()

    async def complete(self, req: ChatCompletionRequest) -> RouteResult:
        errors: list[str] = []
        attempts = 0
        for provider, model in self.targets(req.model):
            for attempt in range(self.settings.max_retries + 1):
                attempts += 1
                span = self._attempt_span(provider, model, req, attempt)
                try:
                    resp = await provider.complete(req, model)
                except ProviderError as e:
                    self._fail_span(span, e)
                    errors.append(f"{provider.name}#{attempt + 1}: {e}")
                    if not e.fallback:
                        raise AllProvidersFailedError(errors, e.status, client_error=True) from e
                    if not e.retryable or attempt == self.settings.max_retries:
                        break
                    await self._backoff(attempt)
                    continue
                tm.set_response_attributes(
                    span, resp.model, resp.finish_reason, resp.prompt_tokens, resp.completion_tokens, resp.id
                )
                span.end()
                return RouteResult(resp, provider.name, attempts)
        raise AllProvidersFailedError(errors)

    async def open_stream(self, req: ChatCompletionRequest) -> StreamHandle:
        errors: list[str] = []
        attempts = 0
        for provider, model in self.targets(req.model):
            for attempt in range(self.settings.max_retries + 1):
                attempts += 1
                span = self._attempt_span(provider, model, req, attempt)
                it = provider.stream(req, model).__aiter__()
                try:
                    first = await it.__anext__()
                except StopAsyncIteration:
                    first = StreamChunk(finish_reason="stop")
                except ProviderError as e:
                    self._fail_span(span, e)
                    errors.append(f"{provider.name}#{attempt + 1}: {e}")
                    if not e.fallback:
                        raise AllProvidersFailedError(errors, e.status, client_error=True) from e
                    if not e.retryable or attempt == self.settings.max_retries:
                        break
                    await self._backoff(attempt)
                    continue
                return StreamHandle(provider.name, model, attempts, first, it, span, errors=errors)
        raise AllProvidersFailedError(errors)
