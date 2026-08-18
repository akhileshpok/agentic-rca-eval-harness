"""
Log Reader Sub-Agent

Single responsibility: parse log lines from a synthetic incident and emit
a typed Observation identifying the most significant error signal found.

The agent uses a structured prompt to extract:
  - The most relevant log line (evidence)
  - A plain-language summary of what it means
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

LOG_READER_PROMPT = """You are a senior site reliability engineer analysing log lines from a production incident.

Your job is to identify the single most significant error or warning signal in the logs below that best explains the incident.

Return ONLY a JSON object with exactly these fields — no preamble, no explanation, no markdown:
{{
  "summary": "<one sentence describing what the log signal means in plain English>",
  "evidence": "<the exact log line that is the strongest signal>",
  "confidence": <float between 0.0 and 1.0 indicating how confident you are this is the root signal>
}}

LOGS:
{logs}
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

def run_log_reader(incident: dict) -> Observation:
    """
    Run the Log Reader sub-agent on a single incident dict.

    Args:
        incident: A single incident loaded from data/incidents.json

    Returns:
        A validated Observation Pydantic model
    """
    logs = incident.get("logs", [])
    if not logs:
        raise ValueError(f"Incident {incident.get('incident_id')} has no logs")

    # Format logs as a readable block for the prompt
    log_block = "\n".join(
        f"[{entry.get('timestamp', '')}] {entry.get('level', '')} {entry.get('service', '')} - {entry.get('message', '')}"
        for entry in logs
    )

    prompt = LOG_READER_PROMPT.format(logs=log_block)

    # Call the LLM
    raw_response = _get_llm_response(prompt)

    # Parse and validate the response
    # Ollama sometimes truncates the closing brace — patch it if needed
    try:
        cleaned = raw_response.strip()
        if not cleaned.endswith("}"):
            cleaned = cleaned + "}"
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Log Reader returned invalid JSON: {e}\nRaw response:\n{raw_response}"
        )

    return Observation(
        source=AgentSource.LOG_READER,
        summary=parsed["summary"],
        evidence=parsed["evidence"],
        confidence=float(parsed["confidence"]),
    )


# ---------------------------------------------------------------------------
# Quick test — run directly to verify the agent works on one incident
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    data_path = Path(__file__).parent.parent / "data" / "incidents.json"
    if not data_path.exists():
        print(f"ERROR: {data_path} not found. Run generate_incidents.py first.")
        sys.exit(1)

    with open(data_path) as f:
        incidents = json.load(f)

    # Test on the first incident
    incident = incidents[0]
    print(f"\nTesting Log Reader on incident: {incident['incident_id']}")
    print(f"Ground truth: {incident['ground_truth']['root_cause']}\n")

    observation = run_log_reader(incident)

    print("--- Log Reader Observation ---")
    print(f"Summary:    {observation.summary}")
    print(f"Evidence:   {observation.evidence}")
    print(f"Confidence: {observation.confidence:.2f}")
    print(f"Source:     {observation.source.value}")
    print(f"\nObservation ID: {observation.id}")
