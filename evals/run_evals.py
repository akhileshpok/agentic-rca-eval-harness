"""
Eval Runner — Two-Tier Evaluation Layer

Tier 1: Component-level eval — scores each sub-agent's Observation
        against the labeled ground truth for each incident.

Tier 2: End-to-end eval — scores the orchestrator's final Hypothesis
        against ground truth using:
          - Correctness (string + semantic match)
          - Groundedness (did the hypothesis cite real observation IDs?)
          - LLM-as-judge (llama3.2 scores hypothesis quality on a rubric)

Results are written to:
  evals/results/component_eval.csv   — Tier 1 scores
  evals/results/e2e_eval.csv         — Tier 2 scores
  evals/results/summary.json         — aggregate metrics

Run:
    python evals/run_evals.py --incidents data/incidents.json --limit 25
"""

import argparse
import json
import os
import sys
import csv
from pathlib import Path
from datetime import datetime, timezone

sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from agents.log_reader import run_log_reader
from agents.metrics_analyst import run_metrics_analyst
from agents.trace_inspector import run_trace_inspector
from agents.hypothesis_generator import run_hypothesis_generator
from orchestrator import run_triage
from schemas.state import AgentSource, Observation, TriageState

load_dotenv()

RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# LLM-as-judge (llama3.2 via Ollama)
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are an expert SRE evaluating the quality of an AI-generated root cause analysis hypothesis.

Score the hypothesis below against the ground truth root cause on a scale of 0.0 to 1.0.

Scoring rubric:
- 1.0: Hypothesis correctly identifies the root cause and affected service
- 0.75: Hypothesis identifies the correct service but describes the cause imprecisely
- 0.5: Hypothesis is partially correct — right domain but wrong service or cause
- 0.25: Hypothesis is plausible but does not match the ground truth
- 0.0: Hypothesis is wrong or hallucinates a cause not supported by evidence

Ground truth: {ground_truth}
Hypothesis: {hypothesis}

Return ONLY a JSON object with exactly these fields — no preamble, no markdown:
{{
  "score": <float 0.0-1.0>,
  "reasoning": "<one sentence explaining the score>"
}}
"""

def llm_judge_score(hypothesis: str, ground_truth: str) -> dict:
    """Run the LLM-as-judge and return a score + reasoning."""
    import requests
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    judge_model = os.getenv("JUDGE_MODEL", "llama3.2")

    prompt = JUDGE_PROMPT.format(
        ground_truth=ground_truth,
        hypothesis=hypothesis,
    )

    response = requests.post(
        f"{base_url}/api/generate",
        json={"model": judge_model, "prompt": prompt, "stream": False},
    )
    response.raise_for_status()
    raw = response.json()["response"].strip()

    try:
        if not raw.endswith("}"):
            raw = raw + "}"
        parsed = json.loads(raw)
        return {
            "judge_score": float(parsed["score"]),
            "judge_reasoning": parsed.get("reasoning", ""),
        }
    except (json.JSONDecodeError, KeyError):
        return {"judge_score": 0.0, "judge_reasoning": "Failed to parse judge response"}


# ---------------------------------------------------------------------------
# Groundedness check
# ---------------------------------------------------------------------------

def check_groundedness(hypothesis, triage_state: TriageState) -> float:
    """
    Check what fraction of supporting_observation_ids in the hypothesis
    actually reference real observation IDs from the TriageState.
    Returns a score between 0.0 and 1.0.
    """
    real_ids = {obs.id for obs in triage_state.observations}
    all_cited = []
    for cause in hypothesis.ranked_causes:
        all_cited.extend(cause.supporting_observation_ids)

    if not all_cited:
        return 0.0

    valid = sum(1 for cid in all_cited if cid in real_ids)
    return valid / len(all_cited)


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------

def check_correctness(top_cause: str, ground_truth: str) -> float:
    """
    Simple keyword-based correctness check.
    Returns 1.0 if key terms from ground truth appear in the hypothesis,
    0.5 for partial match, 0.0 for no match.
    """
    gt_keywords = set(ground_truth.lower().split())
    # Remove common stop words
    stop_words = {"the", "a", "an", "is", "was", "in", "of", "to", "and", "or", "for"}
    gt_keywords -= stop_words

    top_cause_lower = top_cause.lower()
    matches = sum(1 for kw in gt_keywords if kw in top_cause_lower)

    if matches == 0:
        return 0.0
    elif matches / len(gt_keywords) >= 0.5:
        return 1.0
    else:
        return 0.5


# ---------------------------------------------------------------------------
# Tier 1 — Component eval
# ---------------------------------------------------------------------------

def run_component_eval(incidents: list) -> list:
    """
    Score each sub-agent's Observation against ground truth for each incident.
    Returns a list of result dicts.
    """
    results = []
    agents = [
        ("log_reader",       run_log_reader),
        ("metrics_analyst",  run_metrics_analyst),
        ("trace_inspector",  run_trace_inspector),
    ]

    for i, incident in enumerate(incidents):
        incident_id = incident["incident_id"]
        ground_truth = incident["ground_truth"]["root_cause"]
        affected_service = incident["ground_truth"]["affected_service"]

        print(f"  [{i+1}/{len(incidents)}] Component eval: {incident_id}")

        for agent_name, agent_fn in agents:
            try:
                obs = agent_fn(incident)
                correctness = check_correctness(obs.summary, ground_truth)
                service_match = affected_service.lower() in obs.evidence.lower()

                results.append({
                    "incident_id": incident_id,
                    "agent": agent_name,
                    "ground_truth": ground_truth,
                    "affected_service": affected_service,
                    "summary": obs.summary,
                    "evidence": obs.evidence,
                    "confidence": obs.confidence,
                    "correctness": correctness,
                    "service_match": service_match,
                    "error": None,
                })
            except Exception as e:
                results.append({
                    "incident_id": incident_id,
                    "agent": agent_name,
                    "ground_truth": ground_truth,
                    "affected_service": affected_service,
                    "summary": None,
                    "evidence": None,
                    "confidence": None,
                    "correctness": 0.0,
                    "service_match": False,
                    "error": str(e),
                })

    return results


# ---------------------------------------------------------------------------
# Tier 2 — End-to-end eval
# ---------------------------------------------------------------------------

def run_e2e_eval(incidents: list) -> list:
    """
    Score the full orchestrator pipeline against ground truth for each incident.
    Returns a list of result dicts.
    """
    results = []

    for i, incident in enumerate(incidents):
        incident_id = incident["incident_id"]
        ground_truth = incident["ground_truth"]["root_cause"]
        affected_service = incident["ground_truth"]["affected_service"]

        print(f"  [{i+1}/{len(incidents)}] E2E eval: {incident_id}")

        try:
            # Run full pipeline
            triage = TriageState(incident_id=incident_id)
            log_obs = run_log_reader(incident)
            metrics_obs = run_metrics_analyst(incident)
            trace_obs = run_trace_inspector(incident)
            triage.add_observation(log_obs)
            triage.add_observation(metrics_obs)
            triage.add_observation(trace_obs)

            hypothesis = run_hypothesis_generator(triage)
            top_cause = hypothesis.top_cause.cause

            # Score
            correctness = check_correctness(top_cause, ground_truth)
            groundedness = check_groundedness(hypothesis, triage)
            judge = llm_judge_score(top_cause, ground_truth)

            results.append({
                "incident_id": incident_id,
                "ground_truth": ground_truth,
                "affected_service": affected_service,
                "top_cause": top_cause,
                "overall_confidence": hypothesis.overall_confidence,
                "correctness": correctness,
                "groundedness": groundedness,
                "judge_score": judge["judge_score"],
                "judge_reasoning": judge["judge_reasoning"],
                "error": None,
            })

        except Exception as e:
            results.append({
                "incident_id": incident_id,
                "ground_truth": ground_truth,
                "affected_service": affected_service,
                "top_cause": None,
                "overall_confidence": None,
                "correctness": 0.0,
                "groundedness": 0.0,
                "judge_score": 0.0,
                "judge_reasoning": None,
                "error": str(e),
            })

    return results


# ---------------------------------------------------------------------------
# Write results
# ---------------------------------------------------------------------------

def write_csv(results: list, path: Path):
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"  Wrote {len(results)} rows → {path}")


def write_summary(component_results: list, e2e_results: list, path: Path):
    def avg(lst, key):
        vals = [r[key] for r in lst if r[key] is not None]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    # Per-agent component scores
    agent_scores = {}
    for agent in ["log_reader", "metrics_analyst", "trace_inspector"]:
        agent_rows = [r for r in component_results if r["agent"] == agent]
        agent_scores[agent] = {
            "avg_correctness": avg(agent_rows, "correctness"),
            "avg_confidence": avg(agent_rows, "confidence"),
            "service_match_rate": round(
                sum(1 for r in agent_rows if r["service_match"]) / len(agent_rows), 3
            ) if agent_rows else 0.0,
        }

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "incident_count": len(e2e_results),
        "component_eval": agent_scores,
        "e2e_eval": {
            "avg_correctness": avg(e2e_results, "correctness"),
            "avg_groundedness": avg(e2e_results, "groundedness"),
            "avg_judge_score": avg(e2e_results, "judge_score"),
            "avg_overall_confidence": avg(e2e_results, "overall_confidence"),
            "error_rate": round(
                sum(1 for r in e2e_results if r["error"]) / len(e2e_results), 3
            ) if e2e_results else 0.0,
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Wrote summary → {path}")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run two-tier evals on the RCA pipeline")
    parser.add_argument("--incidents", type=str, default="data/incidents.json")
    parser.add_argument("--limit", type=int, default=25, help="Number of incidents to eval")
    parser.add_argument("--tier", choices=["1", "2", "both"], default="both",
                        help="Which eval tier to run")
    args = parser.parse_args()

    incidents_path = Path(args.incidents)
    if not incidents_path.exists():
        print(f"ERROR: {incidents_path} not found. Run generate_incidents.py first.")
        sys.exit(1)

    with open(incidents_path) as f:
        incidents = json.load(f)[:args.limit]

    print(f"\nRunning evals on {len(incidents)} incidents (tier={args.tier})\n")

    component_results, e2e_results = [], []

    if args.tier in ("1", "both"):
        print("=== Tier 1: Component Eval ===")
        component_results = run_component_eval(incidents)
        write_csv(component_results, RESULTS_DIR / "component_eval.csv")

    if args.tier in ("2", "both"):
        print("\n=== Tier 2: End-to-End Eval ===")
        e2e_results = run_e2e_eval(incidents)
        write_csv(e2e_results, RESULTS_DIR / "e2e_eval.csv")

    if component_results or e2e_results:
        print("\n=== Summary ===")
        summary = write_summary(
            component_results, e2e_results, RESULTS_DIR / "summary.json"
        )
        print(json.dumps(summary["e2e_eval"], indent=2))


if __name__ == "__main__":
    main()
