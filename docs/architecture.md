# HA Multi-Agent Smart Home Assistant – Architecture

## Overview

```
User (chat / voice / UI)
         │
         ▼
   FastAPI Gateway
   ┌─────────────────────────────────────────┐
   │  POST /chat   WebSocket /ws/{session}   │
   └──────────────────┬──────────────────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │   LangGraph Orchestrator│
         │   (classify → route)   │
         └──┬────┬────┬────┬──────┘
            │    │    │    │
       ──────────────────────────────
       │         │         │         │         │
       ▼         ▼         ▼         ▼         ▼
  INSTANT     TOOL      GENERAL   RESEARCH  WORKFLOW
  < 100ms   < 800ms    < 3s       5-15s     async
     │          │          │         │         │
  Direct     nemotron   Qwen 2.5  SearXNG  n8n / Temporal
  HA call    4B nano    7-9B+RAG  +Online   webhooks /
  no LLM     tool call  Qdrant    LLM       durable wf
     │          │          │         │         │
     └──────────┴──────────┴─────────┴─────────┘
                           │
               ┌───────────▼───────────┐
               │    Home Assistant     │
               │  REST API + WebSocket │
               └───────────────────────┘
```

## Routing Logic

| Trigger | Route | Agent | Latency |
|---------|-------|-------|---------|
| "Licht an im Wohnzimmer" / "Turn on living room lights" | INSTANT | Pattern regex → HA REST | < 100ms |
| "Set brightness to 40%" (complex entity) | TOOL | nemotron-mini 4B | < 800ms |
| "Wie ist die Stimmung zuhause?" / "How's the house?" | GENERAL | Qwen 2.5 7B + RAG | 1-3s |
| "Was kostet aktuell Strom?" / "Search for X" | RESEARCH | SearXNG + local/online LLM | 5-15s |
| "Guten Morgen" / "Movie time" | WORKFLOW | n8n webhook (no LLM) | < 500ms |
| Multi-step complex automation | WORKFLOW | Temporal durable workflow | async |

## Stack

| Component | Purpose | Memory |
|-----------|---------|--------|
| Ollama | Local LLM inference (keep_alive=-1) | ~18GB |
| nemotron-mini 4B | Tool calling agent | ~4GB |
| Qwen 2.5 7B | General chat + RAG | ~8GB |
| Qdrant | Vector DB for RAG | ~500MB |
| Redis | State cache + history | ~3GB |
| SearXNG | Private web search | ~200MB |
| n8n | Non-AI workflow automation | ~500MB |
| Temporal | Durable workflow orchestration | ~500MB |
| fastembed | Local embeddings (BAAI/bge-small) | ~200MB |

**Total estimated RAM: ~35GB** (fits comfortably in 64GB)

## Languages

Intent patterns cover **English** and **German** in `config/intents.yaml`.
All LLM agents are instructed to reply in the user's language.

## Adding New Intents (Instant Path)

Edit `config/intents.yaml`, add a new entry under `instant_intents`:

```yaml
my_custom_intent:
  patterns:
    - "turn on the coffee machine"        # EN
    - "kaffeemaschine an"                 # DE
  service: "switch.turn_on"
  slot_map:
    name: entity_name
```

The intent engine reloads automatically on restart.

## Adding n8n Workflows

1. Create the workflow in the n8n UI at `http://localhost:5678`
2. Add a Webhook trigger node
3. Register the webhook path in `config/settings.yaml` under `n8n.workflow_webhooks`
4. Add a matching pattern in `config/intents.yaml` with `action: n8n_webhook`

## Adding Temporal Workflows

1. Define `@workflow.defn` and `@activity.defn` in `core/workflows/temporal_workflows.py`
2. Register them in `core/workflows/temporal_worker.py`
3. Add the workflow name to `temporal.durable_workflows` in `config/settings.yaml`
