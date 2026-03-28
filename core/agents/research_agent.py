"""
Research Agent – SearXNG search + optional escalation to online LLM.
Triggered when the query needs external knowledge:
  - Product comparisons, recommendations
  - "How does X work?"
  - News, prices, weather (beyond HA sensors)
  - Any query the local models flag as needing more knowledge

Flow:
  1. SearXNG search for the query
  2. Summarize results with Qwen 3.5 9B locally
  3. If local confidence is low → escalate to online LLM (Claude Haiku / GPT-4o-mini)
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from core.agents.base_agent import AgentState
from core.config import get_settings
from core.integrations.searxng_client import get_searxng
from core.integrations.ollama_client import get_ollama

log = structlog.get_logger(__name__)


class ResearchAgent:
    def __init__(self) -> None:
        cfg = get_settings()
        self._general_model = cfg.get("models", "general", "name", default="qwen2.5:7b")
        self._search_count = cfg.get("routing", "search_result_count", default=8)
        self._escalate_threshold = cfg.get(
            "routing", "escalate_to_online_threshold", default=0.4
        )
        self._online_provider = cfg.get("models", "online", "provider", default="anthropic")
        self._online_anthropic_model = cfg.get(
            "models", "online", "anthropic_model", default="claude-3-5-haiku-20241022"
        )
        self._online_openai_model = cfg.get(
            "models", "online", "openai_model", default="gpt-4o-mini"
        )

    async def run(self, state: AgentState) -> AgentState:
        t0 = time.monotonic()
        cfg = get_settings()
        searxng = get_searxng()
        ollama = get_ollama()

        # ── Search ────────────────────────────────────────────────────────
        results = await searxng.search_general(
            state.user_message, num_results=self._search_count
        )
        log.info("research_agent.searched", query=state.user_message, hits=len(results))

        search_context = self._format_results(results)

        # ── Local synthesis attempt ───────────────────────────────────────
        synth_prompt = (
            f"Answer the following question based on the search results provided.\n"
            f"Be concise and factual. If the search results don't contain enough information, "
            f"say so and provide what you know.\n\n"
            f"Question: {state.user_message}\n\n"
            f"Search Results:\n{search_context}"
        )

        messages = [
            {
                "role": "system",
                "content": "You are a helpful research assistant. Synthesize search results accurately.",
            },
            *state.history[-6:],
            {"role": "user", "content": synth_prompt},
        ]

        local_resp = await ollama.chat(
            model=self._general_model,
            messages=messages,
            temperature=0.2,
            num_ctx=8192,
            num_predict=1024,
        )
        local_answer = local_resp.get("message", {}).get("content", "")

        # ── Check if escalation is needed ─────────────────────────────────
        needs_escalation = (
            state.needs_online_llm
            or self._needs_escalation(local_answer, results)
        )

        if needs_escalation and (cfg.anthropic_api_key or cfg.openai_api_key):
            log.info("research_agent.escalating_to_online")
            online_answer = await self._call_online_llm(
                state, search_context, cfg
            )
            state.response = online_answer
            state.agent_used = f"research+online({self._online_provider})"
        else:
            state.response = local_answer
            state.agent_used = "research+local"

        state.latency_ms = (time.monotonic() - t0) * 1000
        log.info("research_agent.done", latency_ms=round(state.latency_ms))
        return state

    # ── Helpers ───────────────────────────────────────────────────────────

    def _format_results(self, results: list[dict]) -> str:
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] {r['title']}")
            if r.get("content"):
                lines.append(f"    {r['content'][:300]}")
            lines.append(f"    Source: {r['url']}")
        return "\n".join(lines)

    def _needs_escalation(self, answer: str, results: list[dict]) -> bool:
        """Heuristic: escalate if answer is vague or results are sparse."""
        if not results:
            return True
        low_confidence_phrases = [
            "i don't know",
            "i'm not sure",
            "i cannot",
            "insufficient information",
            "not enough information",
            "unable to find",
        ]
        answer_lower = answer.lower()
        return any(phrase in answer_lower for phrase in low_confidence_phrases)

    async def _call_online_llm(
        self,
        state: AgentState,
        search_context: str,
        cfg,
    ) -> str:
        prompt = (
            f"Answer concisely based on the search results and your knowledge.\n\n"
            f"Question: {state.user_message}\n\n"
            f"Search Results:\n{search_context}"
        )

        if self._online_provider == "anthropic" and cfg.anthropic_api_key:
            return await self._call_anthropic(prompt, cfg)
        elif cfg.openai_api_key:
            return await self._call_openai(prompt, cfg)
        return state.response  # fallback to local answer

    async def _call_anthropic(self, prompt: str, cfg) -> str:
        try:
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=cfg.anthropic_api_key)
            message = await client.messages.create(
                model=self._online_anthropic_model,
                max_tokens=cfg.get("models", "online", "max_tokens", default=2048),
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except Exception as e:
            log.error("research_agent.anthropic_error", error=str(e))
            return ""

    async def _call_openai(self, prompt: str, cfg) -> str:
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=cfg.openai_api_key)
            resp = await client.chat.completions.create(
                model=self._online_openai_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=cfg.get("models", "online", "max_tokens", default=2048),
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            log.error("research_agent.openai_error", error=str(e))
            return ""
