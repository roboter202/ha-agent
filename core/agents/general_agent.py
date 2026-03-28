"""
General Agent – Qwen 3.5 9B (or configured general model) + RAG.
Handles:
  - Conversational home management
  - Status summaries ("how's the house?")
  - Recommendations ("should I turn on the AC?")
  - HA knowledge questions

Target latency: < 3s first token (streaming), ~20 TPS.
"""

from __future__ import annotations

import time
from typing import AsyncIterator

import structlog

from core.agents.base_agent import AgentState
from core.config import get_settings
from core.integrations.home_assistant import HAClient
from core.integrations.ollama_client import get_ollama
from core.memory.cache import get_cache
from core.memory.rag import get_rag

log = structlog.get_logger(__name__)


_SYSTEM_PROMPT = """\
You are a helpful, knowledgeable smart home assistant integrated with Home Assistant.
You can answer questions about the home, explain device states, make recommendations,
and help the user understand their home's systems.

Language: Always respond in the same language the user wrote in.
Sprache: Antworte immer in der Sprache, in der der Nutzer geschrieben hat.
Supported languages: English and German (Englisch und Deutsch).

Guidelines / Richtlinien:
- Be concise and conversational / Sei präzise und gesprächig
- Use actual device names and states from the context provided
- If you're unsure about something, say so rather than guessing
- For actions, prefer short confirmations / Für Aktionen kurze Bestätigungen
- For questions, give helpful and specific answers using the home context
"""


class GeneralAgent:
    def __init__(self) -> None:
        cfg = get_settings()
        self._model = cfg.get("models", "general", "name", default="qwen2.5:7b")
        self._temperature = cfg.get("models", "general", "temperature", default=0.3)
        self._num_ctx = cfg.get("models", "general", "num_ctx", default=8192)
        self._num_predict = cfg.get("models", "general", "num_predict", default=1024)

    async def run(self, state: AgentState, ha_client: HAClient) -> AgentState:
        t0 = time.monotonic()
        ollama = get_ollama()
        cache = get_cache()
        rag = get_rag()

        # ── Gather context ──────────────────────────────────────────────
        ha_context = await self._build_ha_context(state.user_message, cache)
        rag_hits = await rag.search_home_context(state.user_message)
        rag_context = "\n".join(h["text"] for h in rag_hits)

        state.ha_context = ha_context
        state.rag_context = rag_context

        system = _SYSTEM_PROMPT
        if ha_context:
            system += f"\n\n## Current Home State\n{ha_context}"
        if rag_context:
            system += f"\n\n## Home Knowledge\n{rag_context}"

        messages = [
            {"role": "system", "content": system},
            *state.history[-12:],
            {"role": "user", "content": state.user_message},
        ]

        # Non-streaming (used by REST endpoint or when caller wants full response)
        response = await ollama.chat(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            num_ctx=self._num_ctx,
            num_predict=self._num_predict,
        )
        state.response = response.get("message", {}).get("content", "")
        state.agent_used = "general"
        state.latency_ms = (time.monotonic() - t0) * 1000
        log.info("general_agent.done", latency_ms=round(state.latency_ms))
        return state

    async def stream(
        self,
        state: AgentState,
        ha_client: HAClient,
    ) -> AsyncIterator[str]:
        """Streaming version – yields text chunks as they arrive from Ollama."""
        ollama = get_ollama()
        cache = get_cache()
        rag = get_rag()

        ha_context = await self._build_ha_context(state.user_message, cache)
        rag_hits = await rag.search_home_context(state.user_message)
        rag_context = "\n".join(h["text"] for h in rag_hits)

        system = _SYSTEM_PROMPT
        if ha_context:
            system += f"\n\n## Current Home State\n{ha_context}"
        if rag_context:
            system += f"\n\n## Home Knowledge\n{rag_context}"

        messages = [
            {"role": "system", "content": system},
            *state.history[-12:],
            {"role": "user", "content": state.user_message},
        ]

        async for chunk in ollama.chat_stream(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            num_ctx=self._num_ctx,
            num_predict=self._num_predict,
        ):
            yield chunk

    # ── Context helpers ───────────────────────────────────────────────────

    async def _build_ha_context(self, query: str, cache) -> str:
        states = await cache.get_all_ha_states()
        if not states:
            return ""

        # Group by domain for readability
        by_domain: dict[str, list[str]] = {}
        for s in states:
            eid = s.get("entity_id", "")
            domain = eid.split(".")[0] if "." in eid else "other"
            friendly = s.get("attributes", {}).get("friendly_name", eid)
            line = f"  {friendly}: {s.get('state', '?')}"
            by_domain.setdefault(domain, []).append(line)

        # Prioritize relevant domains
        priority = ["light", "climate", "sensor", "binary_sensor", "lock", "cover", "switch"]
        lines = []
        for d in priority:
            if d in by_domain:
                lines.append(f"[{d.upper()}]")
                lines.extend(by_domain[d][:10])
        return "\n".join(lines)
