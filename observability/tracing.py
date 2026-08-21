"""
OpenTelemetry Instrumentation

Instruments every LLM call in the agentic RCA pipeline using OpenTelemetry
GenAI semantic conventions. Each sub-agent call becomes a span capturing:
  - gen_ai.system (provider: ollama or anthropic)
  - gen_ai.request.model
  - gen_ai.usage.input_tokens
  - gen_ai.usage.output_tokens
  - gen_ai.request.max_tokens
  - Custom attributes: agent name, incident ID, confidence score

Traces are exported to Arize Phoenix via OTLP HTTP exporter.

Usage:
    from observability.tracing import init_tracing, get_tracer
    init_tracing()  # call once at startup
    tracer = get_tracer()

    with tracer.start_as_current_span("log_reader") as span:
        span.set_attribute("incident.id", incident_id)
        ...
"""

import os
from functools import wraps
from typing import Callable

from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

load_dotenv()

# ---------------------------------------------------------------------------
# Constants — OpenTelemetry GenAI semantic conventions
# ---------------------------------------------------------------------------

GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"

# Custom attributes for this project
INCIDENT_ID = "rca.incident.id"
AGENT_NAME = "rca.agent.name"
AGENT_CONFIDENCE = "rca.agent.confidence"
AGENT_SOURCE = "rca.agent.source"
HYPOTHESIS_TOP_CAUSE = "rca.hypothesis.top_cause"
HYPOTHESIS_OVERALL_CONFIDENCE = "rca.hypothesis.overall_confidence"


# ---------------------------------------------------------------------------
# Tracer initialisation
# ---------------------------------------------------------------------------

def init_tracing(service_name: str = "agentic-rca-system") -> TracerProvider:
    """
    Initialise OpenTelemetry tracing and export to Arize Phoenix.

    Call once at application startup before any agents run.

    Args:
        service_name: Name shown in Phoenix's service list

    Returns:
        The configured TracerProvider
    """
    phoenix_endpoint = os.getenv(
        "PHOENIX_COLLECTOR_ENDPOINT",
        "http://localhost:6006/v1/traces"
    )

    resource = Resource.create({
        "service.name": service_name,
        "service.version": "0.1.0",
        "deployment.environment": "development",
    })

    provider = TracerProvider(resource=resource)

    # Export to Arize Phoenix via OTLP HTTP
    otlp_exporter = OTLPSpanExporter(endpoint=phoenix_endpoint)
    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    # Also print to console in debug mode
    if os.getenv("OTEL_DEBUG", "false").lower() == "true":
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    print(f"OTel tracing initialised → {phoenix_endpoint}")
    return provider


def get_tracer(name: str = "agentic-rca") -> trace.Tracer:
    """Get a tracer instance for creating spans."""
    return trace.get_tracer(name)


# ---------------------------------------------------------------------------
# Instrumented LLM call wrappers
# ---------------------------------------------------------------------------

def trace_llm_call(
    agent_name: str,
    incident_id: str,
    prompt: str,
    raw_response: str,
    confidence: float | None = None,
    model: str | None = None,
    provider: str | None = None,
) -> None:
    """
    Record a completed LLM call as an OTel span with GenAI semantic attributes.

    Call this immediately after each LLM response is received.
    """
    tracer = get_tracer()
    resolved_model = model or os.getenv("OLLAMA_MODEL", "llama3.1")
    resolved_provider = provider or os.getenv("LLM_PROVIDER", "ollama")

    # Rough token estimate — character count / 4 is a standard approximation
    input_tokens = len(prompt) // 4
    output_tokens = len(raw_response) // 4

    with tracer.start_as_current_span(f"{agent_name}.llm_call") as span:
        # GenAI semantic conventions
        span.set_attribute(GEN_AI_SYSTEM, resolved_provider)
        span.set_attribute(GEN_AI_REQUEST_MODEL, resolved_model)
        span.set_attribute(GEN_AI_OPERATION_NAME, "chat")
        span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
        span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)
        span.set_attribute(GEN_AI_REQUEST_MAX_TOKENS, 1024)

        # Custom RCA attributes
        span.set_attribute(INCIDENT_ID, incident_id)
        span.set_attribute(AGENT_NAME, agent_name)
        if confidence is not None:
            span.set_attribute(AGENT_CONFIDENCE, confidence)


def trace_agent_run(agent_name: str, incident_id: str):
    """
    Decorator factory that wraps an agent function in a parent span,
    capturing the full agent execution including LLM call and parsing.

    Usage:
        @trace_agent_run("log_reader", incident_id)
        def run():
            ...
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.start_as_current_span(agent_name) as span:
                span.set_attribute(INCIDENT_ID, incident_id)
                span.set_attribute(AGENT_NAME, agent_name)
                try:
                    result = fn(*args, **kwargs)
                    if hasattr(result, "confidence"):
                        span.set_attribute(AGENT_CONFIDENCE, result.confidence)
                    if hasattr(result, "source"):
                        span.set_attribute(AGENT_SOURCE, result.source.value)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace.StatusCode.ERROR, str(e))
                    raise
        return wrapper
    return decorator


def trace_hypothesis(incident_id: str, hypothesis) -> None:
    """
    Record the final hypothesis as a span with top cause and confidence.
    Call after run_hypothesis_generator completes.
    """
    tracer = get_tracer()
    with tracer.start_as_current_span("hypothesis_generator") as span:
        span.set_attribute(INCIDENT_ID, incident_id)
        span.set_attribute(AGENT_NAME, "hypothesis_generator")
        span.set_attribute(HYPOTHESIS_TOP_CAUSE, hypothesis.top_cause.cause)
        span.set_attribute(
            HYPOTHESIS_OVERALL_CONFIDENCE, hypothesis.overall_confidence
        )
        span.set_attribute(
            "rca.hypothesis.ranked_causes_count", len(hypothesis.ranked_causes)
        )
