"""Provider interface. Every upstream (fake, OpenAI, Anthropic) implements this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx

from gateway.schemas import ChatCompletionRequest


class ProviderError(Exception):
    """An upstream failure.

    retryable: worth trying the same provider again (timeouts, 429, 5xx).
    fallback:  worth trying the next provider (true for nearly everything except
               errors caused by the client's own request, like a 400).
    """

    def __init__(self, message: str, status: int | None = None, retryable: bool = True, fallback: bool = True):
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.fallback = fallback

    @classmethod
    def from_status(cls, status: int, body: str) -> ProviderError:
        if status == 429 or status >= 500:
            return cls(f"upstream {status}: {body[:200]}", status, retryable=True, fallback=True)
        if status in (401, 403):  # our key for this provider is bad -> try another provider
            return cls(f"upstream auth error {status}", status, retryable=False, fallback=True)
        return cls(f"upstream rejected request {status}: {body[:200]}", status, retryable=False, fallback=False)

    @classmethod
    def from_httpx(cls, exc: httpx.HTTPError) -> ProviderError:
        return cls(f"upstream connection error: {type(exc).__name__}", None, retryable=True, fallback=True)


@dataclass
class ProviderResponse:
    content: str
    finish_reason: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    id: str | None = None


@dataclass
class StreamChunk:
    delta: str = ""
    finish_reason: str | None = None
    # Usage normally arrives only with the final chunk.
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class Provider(ABC):
    #: value for the OpenTelemetry `gen_ai.provider.name` attribute
    system: str = "unknown"

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def complete(self, req: ChatCompletionRequest, model: str) -> ProviderResponse: ...

    @abstractmethod
    def stream(self, req: ChatCompletionRequest, model: str) -> AsyncIterator[StreamChunk]: ...

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        pass
