"""
Trace Inspector Sub-Agent

Single responsibility: walk the distributed trace from a synthetic incident
and emit a typed Observation identifying the failing span most likely
responsible for the incident.

The agent uses a structured prompt to extract:
  - The most significant failing span (evidence)
  - A plain-language summary of what the span failure means
  - A confidence score (0.0 - 1.0)

Output is a validated Pydantic Observation — not free text — so the eval
layer can check every field programmatically.
"""

import json
import os
import sys
from pathlib import Path

# Allow running from project root or agents/ directory
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from schemas.state import AgentSource, Observation

load_dotenv()

# ---------------------------------------------------------------------------
# LLM client factory — swap between Ollama and Anthropic via .env
# ---------------------------------------------------------------------------

def _get_llm_response(prompt: str) -> str:
    """
    Route the prompt to the configured LLM provider and return the raw
    text response. Swapping providers requires only a .env change.
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "anthropic":
        return _call_anthropic(prompt)
    elif provider == "ollama":
        return _call_ollama(prompt)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}. Set to 'anthropic' or 'ollama' in .env")


def _call_anthropic(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def _call_ollama(prompt: str) -> str:
    import requests
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    response = requests.post(
        f"{base_url}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
    )
    response.raise_for_status()
    return response.json()["response"]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

TRACE_INSPECTOR_PROMPT = """You are a senior site reliability engineer analysing a distributed trace from a production incident.

Your job is to identify the single most significant failing span — the one most likely to be the root cause of the incident, not just a downstream effect.

Focus on spans with status "error" and look for the deepest failing span in the call chain, or the one with the most descriptive error message.

Return ONLY a JSON object with exactly these fields — no preamble, no explanation, no markdown:
{{
  "summary": "<one sentence describing what the span failure means in plain English and which service is the likely origin>",
  "evidence": "<span details in this format: span_id=<id> service=<service> operation=<operation> duration_ms=<duration> error=<error_message>>",
  "confidence": <float between 0.0 and 1.0 indicating how confident you are this span points to the root cause>
}}

TRACE:
trace_id={trace_id}
{spans}
"""

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

def run_trace_inspector(incident: dict) -> Observation:
    """
    Run the Trace Inspector sub-agent on a single incident dict.

    Args:
        incident: A single incident loaded from data/incidents.json

    Returns:
        A validated Observation Pydantic model
    """
    trace = incident.get("trace", {})
    spans = trace.get("spans", [])
    if not spans:
        raise ValueError(f"Incident {incident.get('incident_id')} has no trace spans")

    # Format spans as a readable block for the prompt
    span_lines = []
    for span in spans:
        line = (
            f"span_id={span.get('span_id', 'unknown')} "
            f"service={span.get('service', 'unknown')} "
            f"operation={span.get('operation', 'unknown')} "
            f"duration_ms={span.get('duration_ms', 'unknown')} "
            f"status={span.get('status', 'unknown')}"
        )
        if span.get("parent_span_id"):
            line += f" parent={span['parent_span_id']}"
        if span.get("error_message"):
            line += f" error=\"{span['error_message']}\""
        span_lines.append(line)

    span_block = "\n".join(span_lines)
    prompt = TRACE_INSPECTOR_PROMPT.format(
        trace_id=trace.get("trace_id", "unknown"),
        spans=span_block,
    )

    # Call the LLM
    raw_response = _get_llm_response(prompt)

    # Parse and validate — patch truncated JSON from smaller local models
    try:
        cleaned = raw_response.strip()
        if not cleaned.endswith("}"):
            cleaned = cleaned + "}"
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Trace Inspector returned invalid JSON: {e}\nRaw response:\n{raw_response}"
        )

    return Observation(
        source=AgentSource.TRACE_INSPECTOR,
        summary=parsed["summary"],
        evidence=parsed["evidence"],
        confidence=float(parsed["confidence"]),
    )


# ---------------------------------------------------------------------------
# Quick test — run directly to verify the agent works on one incident
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data_path = Path(__file__).parent.parent / "data" / "incidents.json"
    if not data_path.exists():
        print(f"ERROR: {data_path} not found. Run generate_incidents.py first.")
        sys.exit(1)

    with open(data_path) as f:
        incidents = json.load(f)

    # Test on the first incident
    incident = incidents[0]
    print(f"\nTesting Trace Inspector on incident: {incident['incident_id']}")
    print(f"Ground truth: {incident['ground_truth']['root_cause']}\n")

    observation = run_trace_inspector(incident)

    print("--- Trace Inspector Observation ---")
    print(f"Summary:    {observation.summary}")
    print(f"Evidence:   {observation.evidence}")
    print(f"Confidence: {observation.confidence:.2f}")
    print(f"Source:     {observation.source.value}")
    print(f"\nObservation ID: {observation.id}")
