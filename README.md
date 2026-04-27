# AgentForge — Hierarchical Multi-Agent System with RAG, ReAct, and Production-Grade Eval

> A reference implementation of a **production-style hierarchical multi-agent system** built on LangGraph, demonstrating ReAct reasoning, RAG with chunking optimization, LLM-as-Judge evaluation, conversation-ID tracing, and graceful fallback.
>
> **Why this repo exists**: I led a similar production agent at Microsoft (400+ engineers, >85% MAU, CSAT >4.5/5, zero quality incidents at rollout) built on the **Model Context Protocol (MCP)** + Azure OpenAI. This repo ports the same architecture into the LangGraph idiom to demonstrate the design principles are framework-agnostic.

---

## Table of Contents

- [Architecture](#architecture)
- [Design Principles](#design-principles)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Agent Topology](#agent-topology)
- [RAG Pipeline](#rag-pipeline)
- [Evaluation Framework](#evaluation-framework)
- [Observability](#observability)
- [Production Hardening Checklist](#production-hardening-checklist)
- [Lessons from a Real Production Agent](#lessons-from-a-real-production-agent)

---

## Architecture

```
                       ┌──────────────────────────┐
   user query  ───►    │  Supervisor Agent        │  (router · ReAct loop)
                       │  - intent classification │
                       │  - hierarchical delegate │
                       └──────────┬───────────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            ▼                     ▼                     ▼
    ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
    │ KB Agent     │      │ Code Agent   │      │ Ops Agent    │
    │ (RAG-backed) │      │ (tool-use)   │      │ (action)     │
    └──────┬───────┘      └──────┬───────┘      └──────┬───────┘
           │                     │                     │
           ▼                     ▼                     ▼
    [Vector Store]        [Sandbox Exec]        [HTTP / API tools]
           │                     │                     │
           └─────────────────────┼─────────────────────┘
                                 ▼
                       ┌──────────────────────────┐
                       │  Critic Agent            │  (self-reflection)
                       │  - factuality check      │
                       │  - confidence scoring    │
                       └──────────┬───────────────┘
                                  │
                       confidence ≥ τ ?  ──► answer
                       confidence < τ   ──► fallback path (HITL or template)
```

**Key patterns implemented**:
- **Hierarchical Delegation** — Supervisor routes to specialized sub-agents based on classified intent.
- **ReAct (Reason + Act)** — Each sub-agent runs a Thought → Action → Observation loop with explicit tool selection.
- **Self-Reflection** — Critic agent rescores draft answer against retrieved evidence before commit.
- **Graceful Fallback** — Below-threshold confidence triggers a deterministic template path or HITL queue.

---

## Design Principles

1. **Framework is replaceable; design is not.** This repo runs on LangGraph; the same node graph is structurally equivalent to MCP tool registries, Google ADK agent + tool abstractions, or CrewAI role topologies. Tool selection, state management, eval, and fallback are framework-agnostic.

2. **Eval before launch, eval after launch.** No prompt change ships without an eval-set delta. LLM-as-Judge is a baseline; human sampling is the truth.

3. **Hallucination is a system property, not a model property.** Mitigated across five layers: chunking, retrieval threshold, prompt constraints, output schema, and Critic check.

4. **Observability ≠ logging.** Every reasoning step, tool call, and tool response carries a `conversation_id` traceable across microservices via OpenTelemetry.

5. **Fallback paths are first-class citizens.** Production agents fail open to a degraded but deterministic path — never to a stack trace.

---

## Quick Start

### Prerequisites
- Python 3.11+
- `OPENAI_API_KEY` (or Vertex AI credentials — see `src/llm/provider.py`)
- 4 GB RAM (for local FAISS index)

### Install
```bash
git clone https://github.com/<your-handle>/agentforge.git
cd agentforge
pip install -e ".[dev]"
```

### Build the demo knowledge base
```bash
python -m agentforge.rag.build_index --source data/sample_docs/ --out indexes/demo
```

### Run a query
```bash
python -m agentforge.cli ask "What's our recommended chunking strategy for technical documentation?"
```

### Run the eval suite
```bash
python -m agentforge.eval.run --suite eval/suites/baseline.yaml --judges llm,human-sample
```

Expected output:
```
[eval] suite=baseline cases=50
  factuality:    0.92  (LLM-Judge)
  tool_select:   0.96
  latency_p50:   1.4s
  latency_p95:   3.1s
  fallback_rate: 0.04
  hallucinations(human-sampled): 0/10
```

---

## Project Structure

```
agentforge/
├── README.md
├── pyproject.toml
├── data/
│   └── sample_docs/                # demo corpus (markdown technical docs)
├── eval/
│   ├── suites/
│   │   └── baseline.yaml           # eval set definition
│   └── cases/                      # individual gold-standard cases
├── indexes/                        # built vector indexes (gitignored)
├── src/agentforge/
│   ├── __init__.py
│   ├── cli.py                      # entry point
│   ├── config.py                   # env + thresholds
│   ├── tracing.py                  # OTel + conversation_id propagation
│   │
│   ├── graph/
│   │   ├── supervisor.py           # router + hierarchical delegate
│   │   ├── kb_agent.py             # RAG-backed Q&A
│   │   ├── code_agent.py           # tool-use sandbox
│   │   ├── ops_agent.py            # action / HTTP tools
│   │   └── critic.py               # self-reflection scorer
│   │
│   ├── rag/
│   │   ├── build_index.py          # chunking + embedding + persist
│   │   ├── retriever.py            # top-k + rerank + threshold gate
│   │   └── chunking.py             # semantic-boundary chunker
│   │
│   ├── llm/
│   │   ├── provider.py             # OpenAI / Vertex AI / Gemini swap
│   │   └── prompts/                # versioned prompt templates
│   │
│   ├── tools/
│   │   ├── registry.py             # tool catalog (descriptions + schemas)
│   │   └── ...
│   │
│   ├── eval/
│   │   ├── run.py                  # eval harness
│   │   ├── judges/
│   │   │   ├── llm_judge.py        # LLM-as-Judge
│   │   │   └── human_sample.py     # CLI for human review queue
│   │   └── metrics.py              # factuality / tool-select / latency
│   │
│   └── fallback/
│       └── template.py             # deterministic degraded path
└── tests/
    ├── test_supervisor_routing.py
    ├── test_rag_chunking.py
    ├── test_critic_threshold.py
    └── test_fallback.py
```

---

## Agent Topology

### Supervisor (Router + ReAct)
- Classifies user intent into one of `kb_query | code_task | ops_action | unclear`
- Maintains a ReAct scratchpad in graph state
- Delegates to a single sub-agent per turn (no fan-out by default — keeps cost predictable)
- Falls back to clarifying question when intent confidence < `SUPERVISOR_INTENT_THRESHOLD`

### KB Agent (RAG-backed)
- Retrieves top-k chunks, applies rerank, gates on similarity threshold
- Prompt enforces "answer only from context; if not in context, say I don't know"
- Returns `(answer, citations[], confidence)`

### Code Agent (Tool-Use)
- Executes a curated tool registry (file read, sandboxed Python, API call)
- Tool descriptions are first-class — bad tool descriptions cause bad tool selection
- Each tool returns a normalized `ToolResult` with `success | error | timeout`

### Ops Agent (Action)
- For side-effecting operations (HTTP POST, mutations)
- Requires explicit `confirm=true` flag in graph state for write actions (HITL gate)

### Critic (Self-Reflection)
- Receives the draft answer + retrieved evidence
- Scores: `factuality`, `groundedness`, `coverage`
- If `aggregate_score < CRITIC_THRESHOLD` → trigger fallback or escalate to HITL queue

---

## RAG Pipeline

### Five-Layer Hallucination Defense

| Layer | Mechanism | Knob |
|---|---|---|
| 1. Chunking | Semantic-boundary chunker (preserves headings, code blocks) | `chunking.max_tokens=512`, `chunking.overlap=64` |
| 2. Retrieval | top-k + cross-encoder rerank | `retriever.top_k=8`, `retriever.rerank_top_n=4` |
| 3. Threshold | Reject if best-match similarity below floor | `retriever.min_similarity=0.62` |
| 4. Prompt | Strict "answer-only-from-context" template | `prompts/kb_agent.v3.md` |
| 5. Critic | Self-reflection scoring before commit | `CRITIC_THRESHOLD=0.75` |

### Chunking Strategy
The default chunker splits on semantic boundaries (h1/h2/h3 headings for prose, function boundaries for code), with a fallback to fixed-size + overlap when no structural signal is present. Each chunk carries metadata: `doc_id`, `section_path`, `chunk_index`, `token_count`.

> **Empirical note**: in the source production system, switching from naive 1000-char fixed chunking to semantic-boundary chunking reduced human-edit-rate from ~22% to ~9% on technical KB content.

---

## Evaluation Framework

### Dual-Judge Pattern
Inspired by the Microsoft AI Wiki Agent in production: **never trust a single judge**.

- **LLM-as-Judge** — Fast, cheap, runs on every PR. Scores factuality + groundedness against gold answers.
- **Human Sample** — Slow, expensive, runs weekly. Engineer reviews random N% of production traffic with a structured rubric.

When LLM-Judge and Human-Sample disagree by > 0.15 on the same case, it's a signal the LLM-Judge prompt drifted — retune the judge.

### Eval Suite Format (`eval/suites/baseline.yaml`)
```yaml
suite: baseline
description: Smoke-level regression suite, runs on every prompt change.
judges: [llm, human_sample_5pct]
metrics:
  - factuality
  - tool_select_accuracy
  - latency_p50
  - latency_p95
  - fallback_rate
cases: ./cases/baseline/*.yaml
```

### A/B Testing Prompt Versions
The harness can run two prompt versions in parallel against the same eval set and report a paired-difference summary, so prompt changes ship with quantitative deltas, not vibes.

---

## Observability

### Conversation ID Propagation
Every request enters at the supervisor with a `conversation_id` UUID injected into the LangGraph state. The `tracing.py` module wraps every node and tool call as an OpenTelemetry span with the `conversation_id` set as a span attribute, so a single end-to-end trace can be reconstructed from supervisor → sub-agent → RAG retriever → vector store → LLM call → critic — across processes if you split into microservices.

### Metrics Emitted
- `agentforge.tool.calls{tool, status}` — counter
- `agentforge.tool.latency{tool}` — histogram
- `agentforge.critic.score{aspect}` — histogram
- `agentforge.fallback.triggered{reason}` — counter
- `agentforge.rag.retrieval_hit_rate` — gauge

### Live-Incident Playbook
When a user reports a bad answer:
1. Pull the trace by `conversation_id`
2. Inspect `supervisor.intent_classification` — wrong route?
3. Inspect `kb_agent.retrieval` — bad chunks?
4. Inspect `kb_agent.draft_answer` vs `critic.score` — should have triggered fallback but didn't?
5. Convert the case into an eval-suite addition before patching

---

## Production Hardening Checklist

Drawn from the seven gates I used to take the source production agent live:

- [x] **Eval baseline** — Suite passes with no regression vs main
- [x] **Security review** — API keys via secret manager; PII scrubbed before LLM call; per-tenant index isolation
- [x] **Observability** — Trace, log, metric, alert — all present, all tied to `conversation_id`
- [x] **Fallback path** — Confidence threshold + circuit breaker + template degradation
- [x] **HITL gate** — Critic-flagged answers route to human queue for write/sensitive actions
- [x] **Capacity planning** — Token budget per turn, QPS limit, timeout per node
- [x] **Rollback plan** — Versioned prompts, traffic-split deploy, instant revert via config flag

---

## Lessons from a Real Production Agent

This repo distills the architectural lessons from running an agent in production for 400+ engineers:

1. **The hardest part isn't the agent — it's the eval set.** Bad eval = false confidence. Invest in eval cases earlier than you think.
2. **Tool descriptions are part of the prompt.** A vague tool description silently degrades tool selection accuracy by 10-30%.
3. **Confidence thresholds beat clever prompts.** A `score < 0.75 → fallback` rule prevents more user-visible hallucinations than another round of prompt tweaking.
4. **HITL is leverage, not a crutch.** A well-placed reviewer step in the right 5% of cases keeps overall CSAT above 4.5 — the other 95% can ship fully automated.
5. **Conversation-ID tracing pays for itself the first time something breaks.** When a senior user reports a bad answer at 11pm, you have 15 minutes to find root cause. OTel + conversation_id is the only way.

---

## License

MIT — use freely. If this design helps you ship something to production, send a note: leonzhuinau@gmail.com.

## About the Author

Leon Zhu — 8+ years architecting AI and cloud platforms for Fortune 500 customers. Product Owner of the **AI Wiki Agent at Microsoft** (Azure OpenAI + RAG + MCP, 400+ engineers, >85% MAU, CSAT >4.5/5, zero quality incidents at rollout). Currently exploring frontier multi-agent infrastructure on Vertex AI, Gemini, and Google ADK.

🔗 LinkedIn: Leon Zhu · 📧 leonzhuinau@gmail.com
