"""
Hypothesis Generator

Single responsibility: combine Observations from all three sub-agents into
a ranked, confidence-scored Hypothesis identifying the most likely root cause.

This is the final step in the triage pipeline. It receives the aggregated
TriageState (containing all Observations) and produces a Hypothesis with:
  - One or more ranked root-cause candidates
  - Supporting observation IDs for each candidate (grounding)
  - An overall confidence score

The grounding (linking causes back to specific Observation IDs) is what
makes the eval layer able to check not just correctness but also
groundedness — did the agent cite real evidence, or hallucinate?

Output is a validated Pydantic Hypothesis model.
"""

import json
import os
import sys
from pathlib import Path

# Allow running from project root or agents/ directory
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from schemas.state import AgentSource, Hypothesis, Observation, RankedCause, TriageState

load_dotenv()

# ---------------------------------------------------------------------------
# LLM client factory — swap between Ollama and Anthropic via .env
# ---------------------------------------------------------------------------

def _get_llm_response(prompt: str) -> str:
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

HYPOTHESIS_GENERATOR_PROMPT = """You are a senior site reliability engineer synthesising findings from multiple specialist agents to identify the root cause of a production incident.

You have received the following observations, each from a different evidence source:

{observations}

Your job is to synthesise these observations into a ranked list of root-cause hypotheses.

Rules:
- Rank 1 should be your most confident hypothesis
- Each hypothesis must cite the observation IDs that support it
- Be specific — name the service and failure mode, not just "there was an error"
- If all observations point to the same cause, one ranked cause is fine
- Overall confidence should reflect how strongly the evidence converges

Return ONLY a JSON object with exactly these fields — no preamble, no explanation, no markdown:
{{
  "ranked_causes": [
    {{
      "rank": 1,
      "cause": "<specific description of the root cause, naming the service and failure mode>",
      "supporting_observation_ids": ["<observation_id_1>", "<observation_id_2>"],
      "confidence": <float 0.0-1.0>
    }}
  ],
  "overall_confidence": <float 0.0-1.0>
}}
"""

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

def run_hypothesis_generator(state: TriageState) -> Hypothesis:
    """
    Run the Hypothesis Generator on a completed TriageState.

    Args:
        state: A TriageState with observations from all three sub-agents

    Returns:
        A validated Hypothesis Pydantic model
    """
    if not state.observations:
        raise ValueError(f"TriageState for incident {state.incident_id} has no observations")

    # Format observations as a readable block for the prompt
    obs_lines = []
    for obs in state.observations:
        obs_lines.append(
            f"observation_id={obs.id}\n"
            f"  source={obs.source.value}\n"
            f"  summary={obs.summary}\n"
            f"  evidence={obs.evidence}\n"
            f"  confidence={obs.confidence:.2f}"
        )

    obs_block = "\n\n".join(obs_lines)
    prompt = HYPOTHESIS_GENERATOR_PROMPT.format(observations=obs_block)

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
            f"Hypothesis Generator returned invalid JSON: {e}\nRaw response:\n{raw_response}"
        )

    # Build RankedCause models
    ranked_causes = [
        RankedCause(
            rank=cause["rank"],
            cause=cause["cause"],
            supporting_observation_ids=cause.get("supporting_observation_ids", []),
            confidence=float(cause["confidence"]),
        )
        for cause in parsed["ranked_causes"]
    ]

    return Hypothesis(
        incident_id=state.incident_id,
        ranked_causes=ranked_causes,
        overall_confidence=float(parsed["overall_confidence"]),
    )


# ---------------------------------------------------------------------------
# Quick test — run all three sub-agents then generate a hypothesis
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from agents.log_reader import run_log_reader
    from agents.metrics_analyst import run_metrics_analyst
    from agents.trace_inspector import run_trace_inspector

    data_path = Path(__file__).parent.parent / "data" / "incidents.json"
    if not data_path.exists():
        print(f"ERROR: {data_path} not found. Run generate_incidents.py first.")
        sys.exit(1)

    with open(data_path) as f:
        incidents = json.load(f)

    # Test on the first incident
    incident = incidents[0]
    incident_id = incident["incident_id"]
    print(f"\nRunning full pipeline on incident: {incident_id}")
    print(f"Ground truth: {incident['ground_truth']['root_cause']}\n")

    # Run all three sub-agents
    print("Running sub-agents...")
    log_obs = run_log_reader(incident)
    metrics_obs = run_metrics_analyst(incident)
    trace_obs = run_trace_inspector(incident)

    # Aggregate into TriageState
    state = TriageState(incident_id=incident_id)
    state.add_observation(log_obs)
    state.add_observation(metrics_obs)
    state.add_observation(trace_obs)

    print(f"All agents completed: {[a.value for a in state.completed_agents]}")
    print(f"Ready for hypothesis: {state.is_ready_for_hypothesis}\n")

    # Generate hypothesis
    hypothesis = run_hypothesis_generator(state)

    print("--- Hypothesis ---")
    for cause in hypothesis.ranked_causes:
        print(f"Rank {cause.rank}: {cause.cause}")
        print(f"  Confidence:    {cause.confidence:.2f}")
        print(f"  Supported by:  {cause.supporting_observation_ids}")
    print(f"\nOverall confidence: {hypothesis.overall_confidence:.2f}")
    print(f"Hypothesis ID:      {hypothesis.id}")
    print(f"\nTop cause: {hypothesis.top_cause.cause}")
