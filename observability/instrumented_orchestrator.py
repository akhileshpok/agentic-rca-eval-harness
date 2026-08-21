"""
Instrumented Orchestrator

Extends the base orchestrator with OpenTelemetry tracing via Phoenix's native
instrumentation (arize-phoenix-otel). Every agent call is wrapped in a span —
orchestrator → sub-agents → hypothesis generator — so you can view the full
trace in Arize Phoenix including model name, token counts, and latency.

Install dependencies:
    pip install arize-phoenix-otel openinference-semantic-conventions

Run:
    python observability/instrumented_orchestrator.py

Then open http://localhost:6006 to view traces in Phoenix.
"""

import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Initialise Phoenix tracing — must happen before any agent imports
# ---------------------------------------------------------------------------

from phoenix.otel import register

tracer_provider = register(
    project_name="agentic-rca-system",
    endpoint="http://localhost:6006/v1/traces",
)

from opentelemetry import trace
from opentelemetry.trace import SpanKind
from openinference.semconv.trace import SpanAttributes

from agents.hypothesis_generator import run_hypothesis_generator
from agents.log_reader import run_log_reader
from agents.metrics_analyst import run_metrics_analyst
from agents.trace_inspector import run_trace_inspector
from schemas.state import TriageState

tracer = trace.get_tracer("agentic-rca")

# ---------------------------------------------------------------------------
# Helper — set GenAI + custom attributes on a span
# ---------------------------------------------------------------------------

def _set_llm_attributes(span, agent_name: str, incident_id: str, confidence: float | None = None, summary: str | None = None):
    """Set OpenInference + custom RCA attributes on a span."""
    provider = os.getenv("LLM_PROVIDER", "ollama")
    model = os.getenv("OLLAMA_MODEL", "llama3.1") if provider == "ollama" else os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    # OpenInference semantic conventions — renders in Phoenix UI natively
    span.set_attribute(SpanAttributes.LLM_MODEL_NAME, model)
    span.set_attribute(SpanAttributes.LLM_PROVIDER, provider)
    span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "CHAIN")

    # Custom RCA attributes
    span.set_attribute("rca.agent.name", agent_name)
    span.set_attribute("rca.incident.id", incident_id)
    if confidence is not None:
        span.set_attribute("rca.agent.confidence", confidence)
    if summary is not None:
        span.set_attribute("rca.agent.summary", summary)


# ---------------------------------------------------------------------------
# Instrumented triage pipeline
# ---------------------------------------------------------------------------

def run_instrumented_triage(incident: dict) -> dict:
    """
    Run the full triage pipeline with Phoenix-native OTel tracing.

    Each sub-agent and the hypothesis generator emit spans visible in Phoenix,
    nested under a root span for the full incident triage.
    """
    incident_id = incident.get("incident_id", "unknown")
    ground_truth = incident.get("ground_truth", {})

    with tracer.start_as_current_span("triage_incident", kind=SpanKind.INTERNAL) as root_span:
        root_span.set_attribute("rca.incident.id", incident_id)
        root_span.set_attribute(
            "rca.ground_truth.root_cause",
            ground_truth.get("root_cause", "unknown")
        )
        root_span.set_attribute(
            "rca.ground_truth.affected_service",
            ground_truth.get("affected_service", "unknown")
        )
        root_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "AGENT")

        triage = TriageState(incident_id=incident_id)

        # --- Log Reader ---
        with tracer.start_as_current_span("log_reader", kind=SpanKind.INTERNAL) as span:
            try:
                log_obs = run_log_reader(incident)
                triage.add_observation(log_obs)
                _set_llm_attributes(span, "log_reader", incident_id, log_obs.confidence, log_obs.summary)
                span.set_attribute(SpanAttributes.OUTPUT_VALUE, log_obs.evidence)
                print(f"  ✓ Log Reader (confidence={log_obs.confidence:.2f})")
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.StatusCode.ERROR, str(e))
                raise

        # --- Metrics Analyst ---
        with tracer.start_as_current_span("metrics_analyst", kind=SpanKind.INTERNAL) as span:
            try:
                metrics_obs = run_metrics_analyst(incident)
                triage.add_observation(metrics_obs)
                _set_llm_attributes(span, "metrics_analyst", incident_id, metrics_obs.confidence, metrics_obs.summary)
                span.set_attribute(SpanAttributes.OUTPUT_VALUE, metrics_obs.evidence)
                print(f"  ✓ Metrics Analyst (confidence={metrics_obs.confidence:.2f})")
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.StatusCode.ERROR, str(e))
                raise

        # --- Trace Inspector ---
        with tracer.start_as_current_span("trace_inspector", kind=SpanKind.INTERNAL) as span:
            try:
                trace_obs = run_trace_inspector(incident)
                triage.add_observation(trace_obs)
                _set_llm_attributes(span, "trace_inspector", incident_id, trace_obs.confidence, trace_obs.summary)
                span.set_attribute(SpanAttributes.OUTPUT_VALUE, trace_obs.evidence)
                print(f"  ✓ Trace Inspector (confidence={trace_obs.confidence:.2f})")
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.StatusCode.ERROR, str(e))
                raise

        # --- Hypothesis Generator ---
        with tracer.start_as_current_span("hypothesis_generator", kind=SpanKind.INTERNAL) as span:
            try:
                hypothesis = run_hypothesis_generator(triage)
                _set_llm_attributes(span, "hypothesis_generator", incident_id, hypothesis.overall_confidence)
                span.set_attribute("rca.hypothesis.top_cause", hypothesis.top_cause.cause)
                span.set_attribute("rca.hypothesis.overall_confidence", hypothesis.overall_confidence)
                span.set_attribute("rca.hypothesis.ranked_causes_count", len(hypothesis.ranked_causes))
                span.set_attribute(SpanAttributes.OUTPUT_VALUE, hypothesis.top_cause.cause)
                print(f"  ✓ Hypothesis Generator (confidence={hypothesis.overall_confidence:.2f})")
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.StatusCode.ERROR, str(e))
                raise

        # Record final result on root span
        root_span.set_attribute("rca.hypothesis.top_cause", hypothesis.top_cause.cause)
        root_span.set_attribute("rca.hypothesis.overall_confidence", hypothesis.overall_confidence)
        root_span.set_attribute(SpanAttributes.OUTPUT_VALUE, hypothesis.top_cause.cause)

        return {
            "incident_id": incident_id,
            "ground_truth": ground_truth,
            "top_cause": hypothesis.top_cause.cause,
            "overall_confidence": hypothesis.overall_confidence,
            "hypothesis": hypothesis.model_dump(),
        }


# ---------------------------------------------------------------------------
# Quick test — run on first 3 incidents and view traces in Phoenix
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data_path = Path(__file__).parent.parent / "data" / "incidents.json"
    if not data_path.exists():
        print(f"ERROR: {data_path} not found.")
        sys.exit(1)

    with open(data_path) as f:
        incidents = json.load(f)

    print(f"\nRunning instrumented triage on 3 incidents...")
    print("Open http://localhost:6006 to view traces in Arize Phoenix\n")

    for incident in incidents[:3]:
        incident_id = incident["incident_id"]
        print(f"\n{'='*60}")
        print(f"Incident: {incident_id}")
        print(f"Ground truth: {incident['ground_truth']['root_cause']}")
        print(f"{'='*60}")

        result = run_instrumented_triage(incident)
        print(f"\nTop cause:  {result['top_cause']}")
        print(f"Confidence: {result['overall_confidence']:.2f}")

    print("\nDone — check Phoenix at http://localhost:6006 for traces")
