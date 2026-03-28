"""
Instant agent: pattern-matched intents → direct HA service calls.
No LLM involved. Target latency: < 100ms.

Flow:
  1. Compiled regex patterns from intents.yaml match the user message
  2. Slots are extracted from named groups
  3. Entity IDs are resolved from slot values using the HA state cache
  4. HA service is called or n8n webhook is triggered
  5. Plain-text response returned immediately
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import yaml
import structlog

from core.agents.base_agent import AgentState
from core.integrations.home_assistant import HAClient
from core.integrations.n8n_client import get_n8n
from core.memory.cache import get_cache

log = structlog.get_logger(__name__)

_INTENTS_PATH = Path(__file__).parent.parent.parent / "config" / "intents.yaml"


class CompiledIntent:
    def __init__(self, name: str, cfg: dict[str, Any]) -> None:
        self.name = name
        self.patterns: list[re.Pattern] = [
            re.compile(p, re.IGNORECASE) for p in cfg.get("patterns", [])
        ]
        self.service: str | None = cfg.get("service")
        self.action: str | None = cfg.get("action")
        self.webhook: str | None = cfg.get("webhook")
        self.target: str | None = cfg.get("target")
        self.slot_map: dict[str, str] = cfg.get("slot_map", {})

    def match(self, text: str) -> re.Match | None:
        for pat in self.patterns:
            m = pat.search(text)
            if m:
                return m
        return None


class InstantAgent:
    def __init__(self) -> None:
        self._intents: list[CompiledIntent] = []
        self._load_intents()

    def _load_intents(self) -> None:
        if not _INTENTS_PATH.exists():
            log.warning("instant_agent.intents_missing", path=str(_INTENTS_PATH))
            return
        data = yaml.safe_load(_INTENTS_PATH.read_text())
        for name, cfg in (data.get("instant_intents") or {}).items():
            self._intents.append(CompiledIntent(name, cfg))
        log.info("instant_agent.loaded", count=len(self._intents))

    def classify(self, text: str) -> tuple[CompiledIntent | None, dict[str, Any]]:
        """Return best matching intent and extracted slots, or (None, {})."""
        for intent in self._intents:
            m = intent.match(text)
            if m:
                slots = {k: v.strip() for k, v in m.groupdict().items() if v}
                return intent, slots
        return None, {}

    async def run(self, state: AgentState, ha_client: HAClient) -> AgentState:
        t0 = time.monotonic()
        intent, slots = self.classify(state.user_message)
        if not intent:
            return state  # no match – caller should re-route

        state.intent = intent.name
        state.slots = slots
        state.agent_used = "instant"
        log.info("instant_agent.matched", intent=intent.name, slots=slots)

        try:
            if intent.action == "n8n_webhook" and intent.webhook:
                response = await self._trigger_n8n(intent.webhook)
            elif intent.action == "query_entity_state":
                response = await self._query_state(slots, ha_client)
            elif intent.service:
                response = await self._call_service(intent, slots, ha_client)
            else:
                response = f"Intent '{intent.name}' matched but has no action configured."
        except Exception as e:
            log.error("instant_agent.error", intent=intent.name, error=str(e))
            response = f"Sorry, I couldn't complete that. Error: {e}"

        state.response = response
        state.latency_ms = (time.monotonic() - t0) * 1000
        return state

    # ── Action handlers ───────────────────────────────────────────────────

    async def _trigger_n8n(self, webhook: str) -> str:
        n8n = get_n8n()
        result = await n8n.trigger(webhook)
        if result.get("status") == "error":
            return f"Workflow '{webhook}' failed (code {result.get('code')})"
        return f"Done – {webhook.replace('_', ' ').title()} activated."

    async def _query_state(self, slots: dict, ha_client: HAClient) -> str:
        name = slots.get("entity_name") or slots.get("entity")
        if not name:
            return "Which device would you like to check?"
        cache = get_cache()
        # Try cache first
        all_states = await cache.get_all_ha_states()
        name_lower = name.lower().replace(" ", "_")
        matches = [
            s for s in all_states
            if name_lower in s.get("entity_id", "").lower()
            or name_lower in s.get("attributes", {}).get("friendly_name", "").lower()
        ]
        if not matches:
            return f"I couldn't find a device matching '{name}'."
        s = matches[0]
        friendly = s.get("attributes", {}).get("friendly_name", s["entity_id"])
        return f"{friendly} is currently {s['state']}."

    async def _call_service(
        self,
        intent: CompiledIntent,
        slots: dict,
        ha_client: HAClient,
    ) -> str:
        domain, service = intent.service.split(".", 1)
        service_data: dict[str, Any] = {}
        target: dict[str, Any] = {}

        # Resolve target entity / area
        if intent.target == "all":
            target = {"entity_id": "all"}
        elif "area_name" in intent.slot_map.values():
            area_raw = slots.get("area")
            if area_raw:
                # Resolve entities in area
                entities = await ha_client.get_entities_in_area(
                    area_raw.lower().replace(" ", "_")
                )
                if entities:
                    target = {"entity_id": entities}
                else:
                    # Try with area label directly
                    target = {"area_id": area_raw.lower().replace(" ", "_")}
        elif "entity_name" in intent.slot_map.values():
            eid = await ha_client.resolve_entity(
                slots.get("entity_name", ""), domain=domain
            )
            if eid:
                target = {"entity_id": eid}

        # Build service_data from remaining slots
        for slot_key, ha_key in intent.slot_map.items():
            if ha_key in ("area_name", "entity_name"):
                continue
            val = slots.get(slot_key)
            if val is not None:
                if ha_key == "brightness_pct":
                    service_data["brightness_pct"] = int(val)
                elif ha_key == "temperature":
                    service_data["temperature"] = float(val)
                elif ha_key == "volume_level":
                    service_data["volume_level"] = int(val) / 100.0
                else:
                    service_data[ha_key] = val

        await ha_client.call_service(domain, service, service_data, target)

        # Human-readable confirmation
        action_verb = {
            "turn_on": "turned on",
            "turn_off": "turned off",
            "lock": "locked",
            "unlock": "unlocked",
            "open_cover": "opened",
            "close_cover": "closed",
            "media_play": "playing",
            "media_pause": "paused",
        }.get(service, service.replace("_", " "))

        if slots.get("area"):
            location = slots["area"].title()
        elif slots.get("entity_name"):
            location = slots["entity_name"].title()
        elif intent.target == "all":
            location = "all devices"
        else:
            location = domain

        extra = ""
        if "brightness_pct" in service_data:
            extra = f" ({service_data['brightness_pct']}%)"
        elif "temperature" in service_data:
            extra = f" to {service_data['temperature']}°"

        return f"{location} {action_verb}{extra}."


_agent: InstantAgent | None = None


def get_instant_agent() -> InstantAgent:
    global _agent
    if _agent is None:
        _agent = InstantAgent()
    return _agent
