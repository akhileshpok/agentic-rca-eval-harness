"""
Pydantic state schemas for the agentic RCA system.

These models are the contract between every component: sub-agents emit
Observations, the orchestrator aggregates them into a TriageState, and the
Hypothesis Generator produces a ranked Hypothesis. Keeping these typed
(rather than passing free-text between agents) is what makes the eval layer
possible — every field here is something the eval framework can check
programmatically.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


class AgentSource(str, Enum):
    """Which sub-agent produced a given observation."""

    LOG_READER = "log_reader"
    METRICS_ANALYST = "metrics_analyst"
    TRACE_INSPECTOR = "trace_inspector"


class Observation(BaseModel):
    """
    A single finding emitted by a sub-agent after inspecting its assigned
    evidence source (logs, metrics, or traces).
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    source: AgentSource = Field(..., description="Sub-agent that produced this observation")
    summary: str = Field(..., description="Short natural-language description of what was found")
    evidence: str = Field(..., description="Raw excerpt supporting the summary (log line, metric window, span id)")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Sub-agent's self-reported confidence in this observation"
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RankedCause(BaseModel):
    """A single candidate root cause within a Hypothesis, ranked against others."""

    rank: int = Field(..., ge=1, description="1 = most likely root cause")
    cause: str = Field(..., description="Natural-language description of the candidate root cause")
    supporting_observation_ids: list[str] = Field(
        default_factory=list, description="IDs of Observations that support this cause"
    )
    confidence: float = Field(..., ge=0.0, le=1.0)


class Hypothesis(BaseModel):
    """
    The orchestrator's final output for an incident: one or more ranked
    root-cause candidates, each grounded in specific observations.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    incident_id: str = Field(..., description="ID of the incident this hypothesis addresses")
    ranked_causes: list[RankedCause] = Field(..., min_length=1)
    overall_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in the top-ranked cause, combining self-report and evidence strength"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def top_cause(self) -> RankedCause:
        return min(self.ranked_causes, key=lambda c: c.rank)


class TriageState(BaseModel):
    """
    Aggregate state for a single incident's triage run, threaded through the
    LangGraph orchestrator. This is the object that gets passed between
    graph nodes and updated as each sub-agent reports back.
    """

    incident_id: str
    observations: list[Observation] = Field(default_factory=list)
    hypothesis: Hypothesis | None = None
    completed_agents: set[AgentSource] = Field(default_factory=set)

    def add_observation(self, observation: Observation) -> None:
        self.observations.append(observation)
        self.completed_agents.add(observation.source)

    @property
    def is_ready_for_hypothesis(self) -> bool:
        """All three sub-agents have reported before the Hypothesis Generator runs."""
        return len(self.completed_agents) == len(AgentSource)
