"""Any OpenAI-compatible upstream (OpenAI itself, Groq, Together, vLLM, Ollama...)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from gateway.providers.base import Provider, ProviderError, ProviderResponse, StreamChunk
from gateway.schemas import ChatCompletionRequest


class OpenAICompatProvider(Provider):
    system = "openai"

    def __init__(self, name: str, base_url: str, api_key: str, timeout: float = 30.0):
        super().__init__(name)
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def _body(self, req: ChatCompletionRequest, model: str, stream: bool) -> dict:
        body = req.model_dump(exclude_none=True, exclude={"user"})
        body["model"] = model
        body["stream"] = stream
        if stream:
            body["stream_options"] = {"include_usage": True}
        return body

    async def complete(self, req: ChatCompletionRequest, model: str) -> ProviderResponse:
        try:
            r = await self.client.post("/chat/completions", json=self._body(req, model, False))
        except httpx.HTTPError as e:
            raise ProviderError.from_httpx(e) from e
        if r.status_code != 200:
            raise ProviderError.from_status(r.status_code, r.text)
        data = r.json()
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        return ProviderResponse(
            content=choice["message"].get("content") or "",
            finish_reason=choice.get("finish_reason") or "stop",
            model=data.get("model", model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            id=data.get("id"),
        )

    async def stream(self, req: ChatCompletionRequest, model: str) -> AsyncIterator[StreamChunk]:
        try:
            async with self.client.stream("POST", "/chat/completions", json=self._body(req, model, True)) as r:
                if r.status_code != 200:
                    raise ProviderError.from_status(r.status_code, (await r.aread()).decode())
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        break
                    data = json.loads(payload)
                    chunk = StreamChunk()
                    if data.get("choices"):
                        c = data["choices"][0]
                        chunk.delta = (c.get("delta") or {}).get("content") or ""
                        chunk.finish_reason = c.get("finish_reason")
                    if data.get("usage"):
                        chunk.prompt_tokens = data["usage"].get("prompt_tokens")
                        chunk.completion_tokens = data["usage"].get("completion_tokens")
                    yield chunk
        except httpx.HTTPError as e:
            raise ProviderError.from_httpx(e) from e

    async def aclose(self) -> None:
        await self.client.aclose()
