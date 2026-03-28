"""
LangGraph multi-agent orchestrator.

Graph topology:
  START
    │
    ▼
  [classify] ──instant──► [instant_node] ──► END
    │
    ├──tool──────────────► [tool_node] ──fallback──► [general_node] ──► END
    │                           │
    │                           └──► END
    │
    ├──general───────────► [general_node] ──► END
    │
    ├──research──────────► [research_node] ──► END
    │
    └──workflow──────────► [workflow_node] ──► END

All paths update AgentState.  The API layer reads state.response to reply.
"""

from __future__ import annotations

import re
import time
from typing import Any, Literal

import structlog
from langgraph.graph import StateGraph, END

from core.agents.base_agent import AgentState, RouteType
from core.agents.instant_agent import get_instant_agent
from core.agents.tool_agent import ToolAgent
from core.agents.general_agent import GeneralAgent
from core.agents.research_agent import ResearchAgent
from core.agents.workflow_agent import WorkflowAgent
from core.integrations.home_assistant import HAClient
from core.memory.cache import get_cache

log = structlog.get_logger(__name__)

# ── Routing keywords / patterns ───────────────────────────────────────────────
# Used for fast pre-classification before any LLM call.

_RESEARCH_PATTERNS = re.compile(
    r"(?:search|find|look up|what is|what are|how (?:does|do|can)|explain|"
    r"tell me about|compare|best|recommend|price|buy|review|latest|news|"
    # German
    r"suche|suchen|finde|was ist|was sind|wie (?:funktioniert|geht|kann)|"
    r"erkläre|erkläre mir|vergleiche|bestes|empfiehl|preis|kaufen|bewertung|neueste|nachrichten)",
    re.IGNORECASE,
)

_WORKFLOW_PATTERNS = re.compile(
    r"(?:start|run|activate|trigger|begin|enable|set|good morning|good night|good evening|"
    r"i'?m? (?:leaving|home|back|here)|bye|goodbye|going to sleep|movie (?:time|mode)|"
    # German
    r"starte|aktiviere|auslösen|beginne|guten morgen|gute nacht|guten abend|"
    r"ich (?:gehe|bin zuhause|bin da)|tschüss|auf wiedersehen|schlafenszeit|film (?:zeit|modus))"
    r".*(?:routine|mode|modus|workflow|automation|automatisierung)?",
    re.IGNORECASE,
)

_HA_CONTROL_PATTERNS = re.compile(
    r"(?:turn (?:on|off)|switch (?:on|off)|set|dim|lock|unlock|open|close|play|pause|"
    r"volume|brightness|temperature|thermostat|heat|cool|scene|"
    # German
    r"(?:ein|aus)schalten|anschalten|ausschalten|einstellen|dimmen|"
    r"absperren|aufschließen|öffnen|schließen|abspielen|pausieren|"
    r"lautstärke|helligkeit|temperatur|heizung|kühlen|szene)",
    re.IGNORECASE,
)


class Orchestrator:
    """Builds and runs the LangGraph agent graph."""

    def __init__(self, ha_client: HAClient) -> None:
        self._ha = ha_client
        self._instant = get_instant_agent()
        self._tool = ToolAgent()
        self._general = GeneralAgent()
        self._research = ResearchAgent()
        self._workflow = WorkflowAgent()
        self._graph = self._build_graph()

    # ── Graph construction ────────────────────────────────────────────────

    def _build_graph(self) -> Any:
        builder = StateGraph(AgentState)

        builder.add_node("classify", self._classify_node)
        builder.add_node("instant", self._instant_node)
        builder.add_node("tool", self._tool_node)
        builder.add_node("general", self._general_node)
        builder.add_node("research", self._research_node)
        builder.add_node("workflow", self._workflow_node)

        builder.set_entry_point("classify")

        builder.add_conditional_edges(
            "classify",
            self._route,
            {
                RouteType.INSTANT: "instant",
                RouteType.TOOL: "tool",
                RouteType.GENERAL: "general",
                RouteType.RESEARCH: "research",
                RouteType.WORKFLOW: "workflow",
            },
        )

        # Tool agent can fall back to general if it can't handle the request
        builder.add_conditional_edges(
            "tool",
            lambda s: "general" if not s.response else END,
            {"general": "general", END: END},
        )

        for node in ("instant", "general", "research", "workflow"):
            builder.add_edge(node, END)

        return builder.compile()

    # ── Nodes ──────────────────────────────────────────────────────────────

    async def _classify_node(self, state: AgentState) -> AgentState:
        """Fast rule-based pre-classifier (< 5ms). Loads conversation history."""
        cache = get_cache()
        state.history = await cache.history.get(state.session_id)

        text = state.user_message.strip()

        # 1. Try instant pattern match first (fastest path)
        instant = get_instant_agent()
        intent, slots = instant.classify(text)
        if intent:
            state.route = RouteType.INSTANT
            state.intent = intent.name
            state.slots = slots
            log.info("classify.instant", intent=intent.name)
            return state

        # 2. Workflow keywords
        if _WORKFLOW_PATTERNS.search(text):
            # Check if instant already handles it (routine patterns)
            state.route = RouteType.WORKFLOW
            log.info("classify.workflow")
            return state

        # 3. Research keywords
        if _RESEARCH_PATTERNS.search(text) and not _HA_CONTROL_PATTERNS.search(text):
            state.route = RouteType.RESEARCH
            log.info("classify.research")
            return state

        # 4. HA control keywords → tool agent
        if _HA_CONTROL_PATTERNS.search(text):
            state.route = RouteType.TOOL
            log.info("classify.tool")
            return state

        # 5. Default to general conversation
        state.route = RouteType.GENERAL
        log.info("classify.general")
        return state

    async def _instant_node(self, state: AgentState) -> AgentState:
        state = await self._instant.run(state, self._ha)
        await self._save_to_history(state)
        return state

    async def _tool_node(self, state: AgentState) -> AgentState:
        state = await self._tool.run(state, self._ha)
        if state.response:
            await self._save_to_history(state)
        return state

    async def _general_node(self, state: AgentState) -> AgentState:
        state = await self._general.run(state, self._ha)
        await self._save_to_history(state)
        return state

    async def _research_node(self, state: AgentState) -> AgentState:
        state = await self._research.run(state)
        await self._save_to_history(state)
        return state

    async def _workflow_node(self, state: AgentState) -> AgentState:
        state = await self._workflow.run(state)
        await self._save_to_history(state)
        return state

    # ── Router edge ────────────────────────────────────────────────────────

    @staticmethod
    def _route(state: AgentState) -> RouteType:
        return state.route or RouteType.GENERAL

    # ── History persistence ────────────────────────────────────────────────

    async def _save_to_history(self, state: AgentState) -> None:
        if not state.session_id:
            return
        cache = get_cache()
        await cache.history.add(state.session_id, "user", state.user_message)
        if state.response:
            await cache.history.add(state.session_id, "assistant", state.response)

    # ── Public API ─────────────────────────────────────────────────────────

    async def run(self, session_id: str, message: str) -> AgentState:
        """Run the full graph and return the final state."""
        t0 = time.monotonic()
        initial = AgentState(session_id=session_id, user_message=message)
        final: AgentState = await self._graph.ainvoke(initial)
        final.latency_ms = (time.monotonic() - t0) * 1000
        log.info(
            "orchestrator.done",
            session=session_id,
            route=final.route,
            agent=final.agent_used,
            latency_ms=round(final.latency_ms),
        )
        return final
