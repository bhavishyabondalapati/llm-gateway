"""OpenTelemetry tracing with the GenAI semantic conventions.

Each request produces a trace like:

    POST /v1/chat/completions            (SERVER, gateway.* attributes)
    ├── pii.redact
    ├── cache.lookup
    └── chat gpt-4o-mini                 (CLIENT, one span per provider attempt)

The provider spans use the standard `gen_ai.*` attribute names, so backends that
understand GenAI conventions (Jaeger, Grafana Tempo, Langfuse, Arize...) can show
model, token usage and finish reasons without custom parsing.
Prompt/response *content* is deliberately not recorded (it may contain PII).
"""

from __future__ import annotations

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor, SpanExporter
from opentelemetry.trace import Span, Tracer

from gateway.config import Settings

# GenAI semantic convention attribute names
# (https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/)
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
GEN_AI_SYSTEM = "gen_ai.system"  # older name for provider, kept for compatibility
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
GEN_AI_RESPONSE_ID = "gen_ai.response.id"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
ERROR_TYPE = "error.type"


def setup_tracing(settings: Settings, extra_exporter: SpanExporter | None = None) -> tuple[TracerProvider, Tracer]:
    """Create a TracerProvider for the app.

    We don't set the global provider, so tests can create many apps, each with its
    own in-memory exporter.
    """
    provider = TracerProvider(resource=Resource.create({"service.name": settings.otel_service_name}))
    if settings.otel_exporter_otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        endpoint = settings.otel_exporter_otlp_endpoint.rstrip("/") + "/v1/traces"
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    if settings.otel_console_exporter:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    if extra_exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(extra_exporter))
    return provider, provider.get_tracer("llm-gateway")


def set_request_attributes(span: Span, provider_system: str, model: str, req) -> None:
    span.set_attribute(GEN_AI_OPERATION_NAME, "chat")
    span.set_attribute(GEN_AI_PROVIDER_NAME, provider_system)
    span.set_attribute(GEN_AI_SYSTEM, provider_system)
    span.set_attribute(GEN_AI_REQUEST_MODEL, model)
    if req.max_tokens is not None:
        span.set_attribute(GEN_AI_REQUEST_MAX_TOKENS, req.max_tokens)
    if req.temperature is not None:
        span.set_attribute(GEN_AI_REQUEST_TEMPERATURE, req.temperature)
    if req.top_p is not None:
        span.set_attribute(GEN_AI_REQUEST_TOP_P, req.top_p)


def set_response_attributes(
    span: Span, response_model: str, finish_reason: str | None, input_tokens: int, output_tokens: int,
    response_id: str | None = None,
) -> None:
    span.set_attribute(GEN_AI_RESPONSE_MODEL, response_model)
    if finish_reason:
        span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, [finish_reason])
    span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
    span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)
    if response_id:
        span.set_attribute(GEN_AI_RESPONSE_ID, response_id)
