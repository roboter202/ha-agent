"""
Tool Agent – nemotron-3-nano (4B) via Ollama.
Handles structured commands the instant agent can't match:
  - Multi-entity operations
  - Conditional device queries
  - Complex HA service calls
  - "Turn on X only if Y is off"-style logic

Target latency: < 1s for tool dispatch, < 2s for full round-trip.
"""

from __future__ import annotations

import json
import time
from typing import Any

import structlog

from core.agents.base_agent import AgentState
from core.config import get_settings
from core.integrations.home_assistant import HAClient
from core.integrations.ollama_client import get_ollama
from core.memory.cache import get_cache

log = structlog.get_logger(__name__)

# ── HA Tool Definitions (OpenAI function-calling format) ─────────────────────

HA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ha_get_state",
            "description": "Get the current state and attributes of a Home Assistant entity",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "Full entity ID, e.g. light.living_room",
                    }
                },
                "required": ["entity_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ha_call_service",
            "description": "Call a Home Assistant service to control a device or automation",
            "parameters": {
                "type": "object",
                "properties": {
                    "domain": {"type": "string", "description": "Service domain, e.g. light"},
                    "service": {
                        "type": "string",
                        "description": "Service name, e.g. turn_on",
                    },
                    "entity_id": {
                        "type": "string",
                        "description": "Target entity ID (optional)",
                    },
                    "service_data": {
                        "type": "object",
                        "description": "Additional service parameters",
                    },
                },
                "required": ["domain", "service"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ha_get_area_entities",
            "description": "List all entities in a specific area/room",
            "parameters": {
                "type": "object",
                "properties": {
                    "area_name": {
                        "type": "string",
                        "description": "Area name, e.g. living_room, kitchen",
                    }
                },
                "required": ["area_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ha_query_entities",
            "description": "Search for entities matching a name or domain across the entire home",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Entity name or partial match",
                    },
                    "domain": {
                        "type": "string",
                        "description": "Filter by domain: light, switch, sensor, etc.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


class ToolAgent:
    def __init__(self) -> None:
        cfg = get_settings()
        self._model = cfg.get("models", "tool", "name", default="nemotron-mini")
        self._temperature = cfg.get("models", "tool", "temperature", default=0.0)
        self._num_ctx = cfg.get("models", "tool", "num_ctx", default=4096)
        self._num_predict = cfg.get("models", "tool", "num_predict", default=512)

    async def run(self, state: AgentState, ha_client: HAClient) -> AgentState:
        t0 = time.monotonic()
        cfg = get_settings()
        ollama = get_ollama()
        cache = get_cache()

        # Build HA context snippet (current states of likely relevant entities)
        ha_summary = await self._build_ha_context(state.user_message, cache)

        system_prompt = (
            "You are a smart home assistant. Use the provided tools to control Home Assistant "
            "devices and answer questions about home state. "
            "Be concise and action-oriented. After calling tools, give a short confirmation.\n\n"
            f"Home state summary:\n{ha_summary}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            *state.history[-10:],
            {"role": "user", "content": state.user_message},
        ]

        response = await ollama.chat(
            model=self._model,
            messages=messages,
            tools=HA_TOOLS,
            temperature=self._temperature,
            num_ctx=self._num_ctx,
            num_predict=self._num_predict,
        )

        tool_calls = ollama.extract_tool_calls(response)
        state.tool_calls = tool_calls

        if tool_calls:
            # Execute tool calls
            tool_results = await self._execute_tools(tool_calls, ha_client, cache)

            # Follow-up message with tool results
            messages.append(response.get("message", {}))
            messages.append(
                {
                    "role": "tool",
                    "content": json.dumps(tool_results),
                }
            )
            final_response = await ollama.chat(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                num_ctx=self._num_ctx,
                num_predict=self._num_predict,
            )
            state.response = final_response.get("message", {}).get("content", "Done.")
        else:
            # Direct answer without tool call
            content = response.get("message", {}).get("content", "")
            if content:
                state.response = content
            else:
                # Model couldn't handle it – escalate
                state.response = ""
                state.route = None  # signals orchestrator to re-route to general

        state.agent_used = "tool"
        state.latency_ms = (time.monotonic() - t0) * 1000
        log.info(
            "tool_agent.done",
            latency_ms=round(state.latency_ms),
            tools_called=[c["name"] for c in tool_calls],
        )
        return state

    # ── Tool Execution ────────────────────────────────────────────────────

    async def _execute_tools(
        self,
        calls: list[dict],
        ha: HAClient,
        cache,
    ) -> list[dict]:
        results = []
        for call in calls:
            name = call["name"]
            args = call.get("arguments", {})
            try:
                if name == "ha_get_state":
                    state = await cache.get_ha_state(args["entity_id"])
                    if not state:
                        state = await ha.get_state(args["entity_id"])
                    results.append({"tool": name, "result": state or "not found"})

                elif name == "ha_call_service":
                    svc_data = args.get("service_data", {})
                    target = {}
                    if args.get("entity_id"):
                        target = {"entity_id": args["entity_id"]}
                    await ha.call_service(args["domain"], args["service"], svc_data, target)
                    results.append({"tool": name, "result": "service called successfully"})

                elif name == "ha_get_area_entities":
                    entities = await ha.get_entities_in_area(
                        args["area_name"].lower().replace(" ", "_")
                    )
                    results.append({"tool": name, "result": entities})

                elif name == "ha_query_entities":
                    all_states = await cache.get_all_ha_states()
                    q = args["query"].lower()
                    domain = args.get("domain", "")
                    matches = [
                        s["entity_id"]
                        for s in all_states
                        if q in s.get("entity_id", "").lower()
                        or q in s.get("attributes", {}).get("friendly_name", "").lower()
                        if not domain or s.get("entity_id", "").startswith(domain + ".")
                    ][:10]
                    results.append({"tool": name, "result": matches})

            except Exception as e:
                log.error("tool_agent.tool_error", tool=name, error=str(e))
                results.append({"tool": name, "error": str(e)})

        return results

    async def _build_ha_context(self, query: str, cache) -> str:
        """Build a short HA state summary relevant to the query."""
        states = await cache.get_all_ha_states()
        if not states:
            return "No HA states available."

        # Include a compact summary of all devices
        lines = []
        for s in states[:50]:  # cap to avoid context overflow
            eid = s.get("entity_id", "")
            friendly = s.get("attributes", {}).get("friendly_name", eid)
            lines.append(f"- {friendly} ({eid}): {s.get('state', '?')}")
        return "\n".join(lines)
