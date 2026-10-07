"""FastAPI app: the OpenAI-compatible /v1/chat/completions endpoint.

Request pipeline:
    auth -> rate limit -> token budget -> PII redaction -> cache lookup
         -> provider (retries + fallback) -> PII restoration -> cache store
         -> cost/latency record + trace
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from opentelemetry import context, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.trace import SpanKind, Status, StatusCode, Tracer

from gateway import telemetry as tm
from gateway.auth import KeyStore
from gateway.cache import CachedResponse, ResponseCache, conversation_text
from gateway.config import GatewayConfig, Settings, get_settings, load_gateway_config
from gateway.embeddings import HashingEmbedder
from gateway.pii import PIIRedactor, RedactionResult, StreamRestorer, restore
from gateway.pricing import compute_cost
from gateway.providers import Provider, ProviderError, build_providers
from gateway.ratelimit import RateLimiter
from gateway.router import AllProvidersFailedError, Router, UnknownModelError
from gateway.schemas import ChatCompletionRequest, ChatCompletionResponse, Choice, ResponseMessage, Usage
from gateway.store import RequestRecord, Store

log = logging.getLogger("gateway")


@dataclass
class GatewayState:
    settings: Settings
    config: GatewayConfig
    keys: KeyStore
    limiter: RateLimiter
    store: Store
    redactor: PIIRedactor | None
    cache: ResponseCache | None
    providers: dict[str, Provider]
    router: Router
    tracer_provider: TracerProvider
    tracer: Tracer


def error_response(status: int, message: str, type_: str, code: str | None = None, headers=None) -> JSONResponse:
    """Errors in the same shape OpenAI uses, so SDKs raise the right exception."""
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": type_, "code": code}},
        headers=headers,
    )


def create_app(
    settings: Settings | None = None,
    config: GatewayConfig | None = None,
    span_exporter: SpanExporter | None = None,
    redactor: PIIRedactor | None = None,
) -> FastAPI:
    load_dotenv()  # so real-provider API keys in .env are visible via os.environ
    settings = settings or get_settings()
    config = config or load_gateway_config(settings.gateway_config)
    tracer_provider, tracer = tm.setup_tracing(settings, span_exporter)
    providers = build_providers(config, settings)

    if settings.pii_enabled and redactor is None:
        redactor = PIIRedactor(score_threshold=settings.pii_score_threshold, spacy_model=settings.spacy_model)
    cache = None
    if settings.cache_enabled:
        cache = ResponseCache(
            embedder=HashingEmbedder() if settings.semantic_cache_enabled else None,
            ttl_seconds=settings.cache_ttl_seconds,
            max_entries=settings.cache_max_entries,
            semantic_threshold=settings.semantic_cache_threshold,
        )

    state = GatewayState(
        settings=settings,
        config=config,
        keys=KeyStore(config.teams),
        limiter=RateLimiter(),
        store=Store(settings.database_path),
        redactor=redactor if settings.pii_enabled else None,
        cache=cache,
        providers=providers,
        router=Router(config, providers, settings, tracer),
        tracer_provider=tracer_provider,
        tracer=tracer,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state.redactor is not None:
            await asyncio.to_thread(lambda: state.redactor.analyzer)  # load spaCy before first request
        log.info("providers enabled: %s", ", ".join(providers) or "none")
        yield
        for p in providers.values():
            await p.aclose()
        tracer_provider.shutdown()
        state.store.close()

    app = FastAPI(title="LLM Gateway", version="0.1.0", lifespan=lifespan)
    app.state.gw = state

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError):
        return error_response(400, str(exc.errors()), "invalid_request_error")

    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse("/dashboard")

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "providers": list(providers)}

    @app.get("/v1/models")
    async def list_models(request: Request):
        if state.keys.lookup(request.headers.get("authorization")) is None:
            return error_response(401, "Invalid API key", "invalid_request_error", "invalid_api_key")
        return {
            "object": "list",
            "data": [{"id": m, "object": "model", "owned_by": "llm-gateway"} for m in config.models],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, body: ChatCompletionRequest):
        return await handle_chat(state, request, body)

    from gateway.dashboard import router as dashboard_router

    app.include_router(dashboard_router)
    return app


async def handle_chat(state: GatewayState, request: Request, body: ChatCompletionRequest):
    started = time.perf_counter()
    team = state.keys.lookup(request.headers.get("authorization"))
    if team is None:
        return error_response(401, "Invalid API key", "invalid_request_error", "invalid_api_key")

    rec = RequestRecord(id=f"req-{uuid.uuid4().hex[:16]}", team=team.name, model=body.model,
                        status=200, latency_ms=0, stream=body.stream)
    root = state.tracer.start_span("POST /v1/chat/completions", kind=SpanKind.SERVER)
    root.set_attribute(tm.GEN_AI_OPERATION_NAME, "chat")
    root.set_attribute(tm.GEN_AI_REQUEST_MODEL, body.model)
    root.set_attribute("gateway.team", team.name)
    root.set_attribute("gateway.request_id", rec.id)
    root.set_attribute("gateway.stream", body.stream)
    token = context.attach(trace.set_span_in_context(root))
    ended_by_stream = False

    def finish(status: int, error: str | None = None) -> None:
        rec.status, rec.error = status, error
        rec.latency_ms = round((time.perf_counter() - started) * 1000, 2)
        root.set_attribute("http.response.status_code", status)
        root.set_attribute("gateway.cache", rec.cache)
        root.set_attribute("gateway.cost_usd", rec.cost_usd)
        root.set_attribute("gateway.pii.redactions", sum(rec.redaction_counts.values()))
        root.set_attribute("gateway.attempts", rec.attempts)
        root.set_attribute(tm.GEN_AI_USAGE_INPUT_TOKENS, rec.prompt_tokens)
        root.set_attribute(tm.GEN_AI_USAGE_OUTPUT_TOKENS, rec.completion_tokens)
        if rec.provider:
            root.set_attribute("gateway.provider", rec.provider)
        if status >= 400:
            root.set_status(Status(StatusCode.ERROR, error or str(status)))
            root.set_attribute(tm.ERROR_TYPE, str(status))
        root.end()

    async def fail(status: int, message: str, type_: str, code: str | None = None, headers=None):
        finish(status, message)
        await state.store.record(rec)
        return error_response(status, message, type_, code, headers)

    try:
        # 1. Rate limit
        allowed, remaining, retry_after = state.limiter.check(team.name, team.requests_per_minute)
        rl_headers = {"x-ratelimit-limit-requests": str(team.requests_per_minute),
                      "x-ratelimit-remaining-requests": str(remaining)}
        if not allowed:
            return await fail(429, f"Rate limit exceeded: {team.requests_per_minute} requests/minute",
                              "rate_limit_error", "rate_limit_exceeded",
                              {**rl_headers, "retry-after": str(max(1, round(retry_after)))})

        # 2. Token budget
        used = await state.store.tokens_this_month(team.name)
        if used >= team.monthly_token_budget:
            return await fail(429, f"Monthly token budget of {team.monthly_token_budget} exhausted",
                              "insufficient_quota", "insufficient_quota", rl_headers)

        # 3. Is the model routable?
        try:
            if not state.router.targets(body.model):
                return await fail(503, f"No enabled provider for model '{body.model}'", "server_error")
        except UnknownModelError:
            return await fail(404, f"The model '{body.model}' does not exist", "invalid_request_error",
                              "model_not_found")

        # 4. PII redaction
        redaction = RedactionResult(messages=body.messages)
        if state.redactor is not None:
            with state.tracer.start_as_current_span("pii.redact") as span:
                redaction = await state.redactor.aredact(body.messages)
                span.set_attribute("gateway.pii.redactions", redaction.total)
                for etype, n in redaction.counts.items():
                    span.set_attribute(f"gateway.pii.{etype.lower()}", n)
        rec.redaction_counts = redaction.counts
        upstream_req = body.model_copy(update={"messages": redaction.messages})

        # 5. Cache lookup (on the *redacted* request)
        cache_key = scope = vector = None
        hit: CachedResponse | None = None
        bypass = (
            state.cache is None
            or "no-cache" in request.headers.get("cache-control", "")
            or request.headers.get("x-gateway-cache", "").lower() == "off"
        )
        rec.cache = "bypass" if bypass else "miss"
        if not bypass:
            with state.tracer.start_as_current_span("cache.lookup") as span:
                cache_key = state.cache.exact_key(team.name, upstream_req)
                hit = state.cache.get_exact(cache_key)
                if hit:
                    rec.cache = "exact"
                elif state.settings.semantic_cache_enabled:
                    scope = state.cache.semantic_scope(team.name, upstream_req)
                    vector = state.cache.embed(conversation_text(upstream_req))
                    found = state.cache.get_semantic(scope, vector)
                    if found:
                        hit, similarity = found
                        rec.cache = "semantic"
                        span.set_attribute("gateway.cache.similarity", round(similarity, 4))
                span.set_attribute("gateway.cache", rec.cache)
            if hit:
                rec.provider = "cache"
                rec.saved_usd = compute_cost(state.config, hit.model, hit.prompt_tokens, hit.completion_tokens)

        def store_in_cache(value: CachedResponse) -> None:
            if not bypass and value.finish_reason in ("stop", "length"):
                s = scope or state.cache.semantic_scope(team.name, upstream_req)
                v = vector if vector is not None else state.cache.embed(conversation_text(upstream_req))
                state.cache.put(cache_key, s, v, value)

        def headers() -> dict[str, str]:
            return {**rl_headers, "x-request-id": rec.id, "x-gateway-cache": rec.cache,
                    "x-gateway-provider": rec.provider or "", "x-gateway-redactions": str(redaction.total),
                    "x-gateway-cost-usd": f"{rec.cost_usd:.8f}",
                    "x-gateway-budget-remaining-tokens": str(max(0, team.monthly_token_budget - used))}

        # 6a. Streaming
        if body.stream:
            if hit:
                replay, handle = _replay_stream(hit, redaction.mapping), None
            else:
                try:
                    handle = await state.router.open_stream(upstream_req)
                except AllProvidersFailedError as e:
                    rec.attempts = len(e.errors)
                    return await _provider_failure(fail, e)
                rec.provider, rec.attempts = handle.provider, handle.attempts
                replay = None
            ended_by_stream = True
            return StreamingResponse(
                _stream_body(state, body, rec, redaction, handle, replay, finish, store_in_cache),
                media_type="text/event-stream",
                headers=headers(),
            )

        # 6b. Non-streaming
        if hit:
            content, finish_reason, model = hit.content, hit.finish_reason, hit.model
        else:
            try:
                result = await state.router.complete(upstream_req)
            except AllProvidersFailedError as e:
                rec.attempts = len(e.errors)
                return await _provider_failure(fail, e)
            r = result.response
            rec.provider, rec.attempts = result.provider, result.attempts
            rec.prompt_tokens, rec.completion_tokens = r.prompt_tokens, r.completion_tokens
            rec.cost_usd = compute_cost(state.config, r.model, r.prompt_tokens, r.completion_tokens)
            content, finish_reason, model = r.content, r.finish_reason, r.model
            store_in_cache(CachedResponse(content, finish_reason, model, r.prompt_tokens, r.completion_tokens))

        resp = ChatCompletionResponse(
            model=model,
            choices=[Choice(message=ResponseMessage(content=restore(content, redaction.mapping)),
                            finish_reason=finish_reason)],
            usage=Usage(prompt_tokens=rec.prompt_tokens, completion_tokens=rec.completion_tokens,
                        total_tokens=rec.prompt_tokens + rec.completion_tokens),
        )
        finish(200)
        await state.store.record(rec)
        return JSONResponse(resp.model_dump(), headers=headers())
    except Exception as e:  # last-resort: never leak a span or skip the record
        if not root.is_recording():
            raise
        log.exception("unhandled error")
        ended_by_stream = False
        return await fail(500, f"Internal gateway error: {type(e).__name__}", "server_error")
    finally:
        context.detach(token)
        if not ended_by_stream and root.is_recording():
            root.end()


async def _provider_failure(fail, e: AllProvidersFailedError):
    if e.client_error:  # e.g. upstream said 400: pass that through
        return await fail(e.status or 400, str(e), "invalid_request_error")
    if not e.errors:
        return await fail(503, "No provider available", "server_error")
    return await fail(502, f"All providers failed: {e}", "server_error", "upstream_error")


async def _replay_stream(hit: CachedResponse, mapping) -> AsyncIterator[tuple[str, str | None]]:
    yield restore(hit.content, mapping), None
    yield "", hit.finish_reason


def _sse(obj) -> str:
    return f"data: {json.dumps(obj)}\n\n"


async def _stream_body(state: GatewayState, body: ChatCompletionRequest, rec: RequestRecord,
                       redaction: RedactionResult, handle, replay, finish, store_in_cache):
    """Yield OpenAI-style SSE chunks, restoring PII as text streams through."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def chunk(delta: dict, finish_reason: str | None = None) -> str:
        return _sse({"id": chunk_id, "object": "chat.completion.chunk", "created": created, "model": body.model,
                     "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]})

    status, error = 200, None
    try:
        yield chunk({"role": "assistant", "content": ""})
        finish_reason = "stop"
        if replay is not None:
            async for text, fr in replay:
                if text:
                    yield chunk({"content": text})
                finish_reason = fr or finish_reason
        else:
            restorer = StreamRestorer(redaction.mapping)
            parts: list[str] = []
            async for c in handle.chunks():
                if c.delta:
                    parts.append(c.delta)
                    out = restorer.feed(c.delta)
                    if out:
                        yield chunk({"content": out})
            tail = restorer.flush()
            if tail:
                yield chunk({"content": tail})
            finish_reason = handle.finish_reason or "stop"
            rec.prompt_tokens, rec.completion_tokens = handle.prompt_tokens, handle.completion_tokens
            rec.cost_usd = compute_cost(state.config, handle.model, rec.prompt_tokens, rec.completion_tokens)
            store_in_cache(CachedResponse("".join(parts), finish_reason, handle.model,
                                          rec.prompt_tokens, rec.completion_tokens))
        yield chunk({}, finish_reason)
        if (getattr(body, "stream_options", None) or {}).get("include_usage"):
            yield _sse({"id": chunk_id, "object": "chat.completion.chunk", "created": created, "model": body.model,
                        "choices": [], "usage": {"prompt_tokens": rec.prompt_tokens,
                                                 "completion_tokens": rec.completion_tokens,
                                                 "total_tokens": rec.prompt_tokens + rec.completion_tokens}})
        yield "data: [DONE]\n\n"
    except ProviderError as e:
        status, error = 502, str(e)
        yield _sse({"error": {"message": f"Upstream failed mid-stream: {e}", "type": "server_error"}})
    except (asyncio.CancelledError, GeneratorExit):
        status, error = 499, "client disconnected"
        raise
    finally:
        finish(status, error)
        # Synchronous on purpose: awaiting inside a cancelled generator isn't reliable.
        state.store.record_sync(rec)

