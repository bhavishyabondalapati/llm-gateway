"""A fake LLM provider so the gateway runs and is tested fully offline.

It replies by echoing the last user message, which is handy: any PII placeholders
the gateway sent (e.g. <PERSON_1>) come back in the reply, so restoration is
exercised end to end. Latency and failure rate are configurable to simulate a
slow or flaky upstream.
"""

from __future__ import annotations

import asyncio
import random
import uuid
from collections.abc import AsyncIterator

from gateway.providers.base import Provider, ProviderError, ProviderResponse, StreamChunk
from gateway.schemas import ChatCompletionRequest, estimate_tokens, message_text


class FakeProvider(Provider):
    system = "fake"

    def __init__(self, name: str, latency_ms: float = 20.0, failure_rate: float = 0.0, seed: int | None = None):
        super().__init__(name)
        self.latency_ms = latency_ms
        self.failure_rate = failure_rate
        self.rng = random.Random(seed)
        self.calls = 0  # handy in tests
        self.last_request: ChatCompletionRequest | None = None  # what actually left the gateway

    def _reply(self, req: ChatCompletionRequest) -> str:
        last_user = next((message_text(m) for m in reversed(req.messages) if m.role == "user"), "")
        return f"[{self.name}] You said: {last_user}"

    def _usage(self, req: ChatCompletionRequest, reply: str) -> tuple[int, int]:
        prompt = sum(estimate_tokens(message_text(m)) + 4 for m in req.messages)
        return prompt, estimate_tokens(reply)

    async def _maybe_fail(self, req: ChatCompletionRequest) -> None:
        self.calls += 1
        self.last_request = req
        await asyncio.sleep(self.latency_ms / 1000 * self.rng.uniform(0.5, 1.5))
        if self.rng.random() < self.failure_rate:
            raise ProviderError(f"{self.name}: simulated outage", status=503)

    async def complete(self, req: ChatCompletionRequest, model: str) -> ProviderResponse:
        await self._maybe_fail(req)
        reply = self._reply(req)
        if req.max_tokens is not None:
            reply = reply[: req.max_tokens * 4]
        prompt_tokens, completion_tokens = self._usage(req, reply)
        return ProviderResponse(
            content=reply,
            finish_reason="stop",
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            id=f"fake-{uuid.uuid4().hex[:12]}",
        )

    async def stream(self, req: ChatCompletionRequest, model: str) -> AsyncIterator[StreamChunk]:
        await self._maybe_fail(req)
        reply = self._reply(req)
        # Emit 5-character pieces so placeholders get split across chunks.
        for i in range(0, len(reply), 5):
            await asyncio.sleep(0.001)
            yield StreamChunk(delta=reply[i : i + 5])
        prompt_tokens, completion_tokens = self._usage(req, reply)
        yield StreamChunk(finish_reason="stop", prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
