# Agentic RCA System — Eval & Observability Harness

An open-source reference implementation of an orchestrator/sub-agent system for automated root cause analysis (RCA), built with a production-grade eval and observability layer. This project demonstrates how to design, measure, and monitor an agentic AI system end-to-end — not just build one.

> **Why this project exists:** Most agentic AI demos show a working prototype. Few show how you'd know whether it's actually good, whether it's getting better or worse over time, or how you'd debug it in production. This project treats evaluation and observability as first-class design concerns, not afterthoughts.

---

## Table of Contents
- [Problem](#problem)
- [Architecture](#architecture)
- [Eval Framework](#eval-framework)
- [Observability](#observability)
- [Results](#results)
- [Design Decisions & Tradeoffs](#design-decisions--tradeoffs)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)

---

## Problem

Incident response teams increasingly rely on AI agents to triage and diagnose issues faster than humans can manually correlate logs, metrics, and traces. But most agentic RCA demos stop at "it works on my example." This project asks and answers three harder questions:

1. **How do you decompose RCA into agent responsibilities that are testable in isolation?**
2. **How do you measure whether an agent's hypothesis is actually correct — not just plausible-sounding?**
3. **How do you observe what an agent did, at each step, when something goes wrong?**

---

## Architecture

An orchestrator coordinates single-purpose sub-agents in a ReAct loop, aggregating their outputs into a ranked, confidence-scored root-cause hypothesis.

```
                        ┌─────────────────┐
                        │   Orchestrator    │
                        │  (LangGraph ReAct) │
                        └────────┬──────────┘
                 ┌────────────────┼────────────────┐
                 ▼                ▼                ▼
         ┌───────────────┐ ┌──────────────┐ ┌────────────────┐
         │  Log Reader     │ │ Metrics       │ │ Trace           │
         │  Sub-Agent      │ │ Analyst       │ │ Inspector       │
         │                 │ │ Sub-Agent     │ │ Sub-Agent       │
         └───────┬─────────┘ └──────┬────────┘ └────────┬────────┘
                  └──────────────────┼──────────────────┘
                                     ▼
                        ┌─────────────────────┐
                        │ Hypothesis Generator  │
                        │ (ranked + confidence)  │
                        └─────────────────────┘
```

**Sub-agents (single responsibility each):**
| Agent | Responsibility | Input | Output |
|---|---|---|---|
| Log Reader | Parses synthetic log anomalies, extracts relevant errors | Raw log lines | Structured `Observation` |
| Metrics Analyst | Detects metric spikes/threshold breaches | Synthetic metric series | Structured `Observation` |
| Trace Inspector | Walks distributed trace for latency/error propagation | Synthetic trace tree | Structured `Observation` |
| Hypothesis Generator | Combines observations into ranked root-cause hypotheses | All `Observation`s | `Hypothesis` with confidence score |

**State design:** All agents communicate through typed Pydantic models (`Observation`, `Hypothesis`, `TriageState`), not free-text — this keeps agent outputs machine-checkable, which is what makes the eval layer possible.

**Framework:** [LangGraph](https://github.com/langchain-ai/langgraph) for orchestration — chosen for explicit state graphs and native support for multi-agent routing.

---

## Eval Framework

A two-tier evaluation design, mirroring how you'd need to evaluate this in production:

**Tier 1 — Component-level:** Does each sub-agent do its one job correctly?
- e.g., did the Log Reader extract the actual root-cause error line, not a red herring?

**Tier 2 — End-to-end:** Is the orchestrator's final hypothesis correct and well-justified?
- e.g., does the ranked hypothesis match the labeled ground-truth root cause, with evidence that actually supports it?

**Metrics:**
| Metric | Tier | Tooling |
|---|---|---|
| Correctness (vs. labeled ground truth) | Both | Ragas |
| Groundedness (is the hypothesis supported by cited evidence?) | End-to-end | Ragas + custom LLM-judge |
| Hallucination rate | Both | Ragas |
| Latency / cost per incident | End-to-end | Custom instrumentation |

**Eval set:** 20–30 synthetic, labeled incidents (log/metric/trace triples with a known ground-truth root cause).

**LLM-as-judge:** Runs on a local model via [Ollama](https://ollama.com) to keep the project fully reproducible at zero API cost; optionally swappable for a hosted model.

**CI:** GitHub Actions runs the eval suite on every commit; results are tracked over time so prompt/architecture changes can be compared against a baseline.

---

## Observability

Every orchestrator → sub-agent call is instrumented using [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/), capturing prompts, completions, token usage, and latency as spans.

Traces are viewed in **[Arize Phoenix](https://github.com/Arize-ai/phoenix)** (self-hosted, free), allowing you to inspect exactly which sub-agent contributed which evidence to a given hypothesis — and to correlate a trace with its eval score.

---

## Results

*[To be populated once eval runs are complete — will include: eval score trend chart across prompt/architecture iterations, example trace screenshot paired with its eval score, and a before/after comparison table.]*

---

## Design Decisions & Tradeoffs

*[To be populated: why single-purpose sub-agents over one monolithic agent; why Pydantic-typed state over free-text handoffs; why a two-tier eval design; local vs. hosted judge model tradeoffs; what was cut from scope and why.]*

---

## Getting Started

```bash
git clone https://github.com/<your-username>/agentic-rca-eval-harness.git
cd agentic-rca-eval-harness
pip install -r requirements.txt

# Run the agent on a sample incident
python run_agent.py --incident data/sample_incident_01.json

# Run the eval suite
promptfoo eval

# View traces
docker compose up phoenix
```

*Full setup instructions in [`docs/SETUP.md`](docs/SETUP.md).*

---

## Project Structure

```
├── agents/              # Orchestrator + sub-agent implementations
├── schemas/             # Pydantic models (Observation, Hypothesis, TriageState)
├── data/                # Synthetic incident dataset + labels
├── evals/               # Ragas/Promptfoo configs, LLM-judge prompts
├── observability/        # OTel instrumentation, Phoenix/docker setup
├── docs/                 # Design doc, setup guide
└── .github/workflows/    # CI eval pipeline
```

---

## Roadmap

- [ ] Week 1: Orchestrator + sub-agent architecture
- [ ] Week 2: Eval layer (component + end-to-end)
- [ ] Week 3: Observability integration + design doc

---

*Built as part of a portfolio project exploring production-grade agentic AI system design — architecture, evaluation, and observability treated as equally important disciplines.*
