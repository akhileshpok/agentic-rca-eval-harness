"""
Orchestrator

Wires the four agents into a LangGraph state graph:

    START
      │
      ▼
  [run_sub_agents]  ← dispatches Log Reader, Metrics Analyst, Trace Inspector
      │                 in parallel and aggregates Observations into TriageState
      │
      ▼
  [generate_hypothesis]  ← runs Hypothesis Generator once all agents complete
      │
      ▼
    END

The graph uses a typed state dict (OrchestratorState) as the LangGraph
channel, wrapping our Pydantic TriageState so LangGraph can manage
transitions. All inter-node communication is through typed models —
no free-text handoffs.
"""

import json
import sys
from pathlib import Path
from typing import TypedDict

# Allow running from project root
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from agents.hypothesis_generator import run_hypothesis_generator
from agents.log_reader import run_log_reader
from agents.metrics_analyst import run_metrics_analyst
from agents.trace_inspector import run_trace_inspector
from schemas.state import Hypothesis, TriageState

load_dotenv()


# ---------------------------------------------------------------------------
# LangGraph state — wraps our Pydantic TriageState for the graph channel
# ---------------------------------------------------------------------------

class OrchestratorState(TypedDict):
    incident: dict          # raw incident dict loaded from incidents.json
    triage_state: TriageState | None
    hypothesis: Hypothesis | None
    error: str | None


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def node_run_sub_agents(state: OrchestratorState) -> OrchestratorState:
    """
    Dispatch all three sub-agents against the incident and aggregate their
    Observations into a TriageState.

    In a production system these would run in parallel (asyncio or
    ThreadPoolExecutor). For this reference implementation they run
    sequentially to keep the code readable and debuggable.
    """
    incident = state["incident"]
    incident_id = incident.get("incident_id", "unknown")
    triage = TriageState(incident_id=incident_id)

    agents = [
        ("Log Reader",       run_log_reader),
        ("Metrics Analyst",  run_metrics_analyst),
        ("Trace Inspector",  run_trace_inspector),
    ]

    for name, agent_fn in agents:
        try:
            print(f"  → Running {name}...")
            observation = agent_fn(incident)
            triage.add_observation(observation)
            print(f"    ✓ {name} complete (confidence={observation.confidence:.2f})")
        except Exception as e:
            print(f"    ✗ {name} failed: {e}")
            return {**state, "error": f"{name} failed: {e}"}

    return {**state, "triage_state": triage, "error": None}


def node_generate_hypothesis(state: OrchestratorState) -> OrchestratorState:
    """
    Run the Hypothesis Generator once all sub-agents have reported back.
    Guards against running with incomplete observations.
    """
    triage = state.get("triage_state")

    if triage is None:
        return {**state, "error": "No TriageState available — sub-agents may have failed"}

    if not triage.is_ready_for_hypothesis:
        completed = [a.value for a in triage.completed_agents]
        return {**state, "error": f"Not all sub-agents completed. Got: {completed}"}

    try:
        print("  → Running Hypothesis Generator...")
        hypothesis = run_hypothesis_generator(triage)
        print(f"    ✓ Hypothesis generated (overall_confidence={hypothesis.overall_confidence:.2f})")
        return {**state, "hypothesis": hypothesis, "error": None}
    except Exception as e:
        return {**state, "error": f"Hypothesis Generator failed: {e}"}


# ---------------------------------------------------------------------------
# Routing — stop early if a node errored
# ---------------------------------------------------------------------------

def route_after_sub_agents(state: OrchestratorState) -> str:
    if state.get("error"):
        return END
    return "generate_hypothesis"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(OrchestratorState)

    graph.add_node("run_sub_agents", node_run_sub_agents)
    graph.add_node("generate_hypothesis", node_generate_hypothesis)

    graph.add_edge(START, "run_sub_agents")
    graph.add_conditional_edges("run_sub_agents", route_after_sub_agents)
    graph.add_edge("generate_hypothesis", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_triage(incident: dict) -> dict:
    """
    Run the full triage pipeline on a single incident dict.

    Args:
        incident: A single incident loaded from data/incidents.json

    Returns:
        A result dict with keys: incident_id, hypothesis, error
    """
    app = build_graph()

    initial_state: OrchestratorState = {
        "incident": incident,
        "triage_state": None,
        "hypothesis": None,
        "error": None,
    }

    final_state = app.invoke(initial_state)

    hypothesis = final_state.get("hypothesis")
    error = final_state.get("error")

    return {
        "incident_id": incident.get("incident_id"),
        "ground_truth": incident.get("ground_truth", {}),
        "hypothesis": hypothesis.model_dump() if hypothesis else None,
        "top_cause": hypothesis.top_cause.cause if hypothesis else None,
        "overall_confidence": hypothesis.overall_confidence if hypothesis else None,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Quick test — run the orchestrator on the first two incidents
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data_path = Path(__file__).parent / "data" / "incidents.json"
    if not data_path.exists():
        print(f"ERROR: {data_path} not found. Run generate_incidents.py first.")
        sys.exit(1)

    with open(data_path) as f:
        incidents = json.load(f)

    # Test on first two incidents
    for incident in incidents[:2]:
        print(f"\n{'='*60}")
        print(f"Incident: {incident['incident_id']}")
        print(f"Ground truth: {incident['ground_truth']['root_cause']}")
        print(f"{'='*60}")

        result = run_triage(incident)

        if result["error"]:
            print(f"\nERROR: {result['error']}")
        else:
            print(f"\nTop cause:          {result['top_cause']}")
            print(f"Overall confidence: {result['overall_confidence']:.2f}")
            print(f"Match ground truth: {incident['ground_truth']['root_cause'].lower() in result['top_cause'].lower()}")
