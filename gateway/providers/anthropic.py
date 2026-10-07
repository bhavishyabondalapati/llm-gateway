"""Anthropic Messages API, translated to/from the OpenAI chat format."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from gateway.providers.base import Provider, ProviderError, ProviderResponse, StreamChunk
from gateway.schemas import ChatCompletionRequest, message_text

STOP_REASONS = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length", "tool_use": "tool_calls"}


class AnthropicProvider(Provider):
    system = "anthropic"

    def __init__(self, name: str, api_key: str, base_url: str = "https://api.anthropic.com", timeout: float = 30.0):
        super().__init__(name)
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            timeout=timeout,
        )

    def _body(self, req: ChatCompletionRequest, model: str, stream: bool) -> dict:
        system = "\n\n".join(message_text(m) for m in req.messages if m.role in ("system", "developer"))
        messages = [
            {"role": "assistant" if m.role == "assistant" else "user", "content": message_text(m)}
            for m in req.messages
            if m.role not in ("system", "developer")
        ]
        body: dict = {"model": model, "messages": messages, "max_tokens": req.max_tokens or 1024, "stream": stream}
        if system:
            body["system"] = system
        if req.temperature is not None:
            body["temperature"] = req.temperature
        if req.top_p is not None:
            body["top_p"] = req.top_p
        return body

    async def complete(self, req: ChatCompletionRequest, model: str) -> ProviderResponse:
        try:
            r = await self.client.post("/v1/messages", json=self._body(req, model, False))
        except httpx.HTTPError as e:
            raise ProviderError.from_httpx(e) from e
        if r.status_code != 200:
            raise ProviderError.from_status(r.status_code, r.text)
        data = r.json()
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage") or {}
        return ProviderResponse(
            content=text,
            finish_reason=STOP_REASONS.get(data.get("stop_reason"), "stop"),
            model=data.get("model", model),
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            id=data.get("id"),
        )

    async def stream(self, req: ChatCompletionRequest, model: str) -> AsyncIterator[StreamChunk]:
        prompt_tokens = 0
        try:
            async with self.client.stream("POST", "/v1/messages", json=self._body(req, model, True)) as r:
                if r.status_code != 200:
                    raise ProviderError.from_status(r.status_code, (await r.aread()).decode())
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    kind = event.get("type")
                    if kind == "message_start":
                        prompt_tokens = event["message"].get("usage", {}).get("input_tokens", 0)
                    elif kind == "content_block_delta" and event["delta"].get("type") == "text_delta":
                        yield StreamChunk(delta=event["delta"]["text"])
                    elif kind == "message_delta":
                        yield StreamChunk(
                            finish_reason=STOP_REASONS.get(event["delta"].get("stop_reason"), "stop"),
                            prompt_tokens=prompt_tokens,
                            completion_tokens=event.get("usage", {}).get("output_tokens", 0),
                        )
        except httpx.HTTPError as e:
            raise ProviderError.from_httpx(e) from e

    async def aclose(self) -> None:
        await self.client.aclose()
