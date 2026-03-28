"""
Personal Data Agent – uses nemotron-mini 4B for tool dispatch against
Email, Finance (Firefly III), and Paperless-ngx.

All three sources are modelled as tools in the same tool-calling loop,
so a single query like "how much did I spend at Rewe this week, and are
there any emails from them?" hits all sources in one pass.

Design goals:
  - Fast: tool agent (4B) dispatches in < 1s; data fetches run concurrently
  - Private: all data stays local – no user data ever sent to online LLMs
  - Bilingual: system prompt + tool descriptions cover EN + DE
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import date, timedelta
from typing import Any

import structlog

from core.agents.base_agent import AgentState
from core.config import get_settings
from core.integrations.email_client import get_email
from core.integrations.finance_client import get_finance
from core.integrations.paperless_client import get_paperless
from core.integrations.ollama_client import get_ollama

log = structlog.get_logger(__name__)

# ── Tool definitions ──────────────────────────────────────────────────────────

PERSONAL_TOOLS = [
    # ── Email ────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "email_inbox_summary",
            "description": (
                "Get inbox overview: unread count and recent unread subjects. "
                "Use for: 'check my emails', 'any new emails?', 'E-Mails checken', 'neue Mails?'"
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "email_search",
            "description": (
                "Search emails by sender, subject keyword, or date. "
                "Use for: 'emails from X', 'find email about Y', 'Mails von X'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sender": {"type": "string", "description": "Sender email or name filter"},
                    "subject_contains": {"type": "string", "description": "Keyword in subject"},
                    "days_back": {
                        "type": "integer",
                        "description": "Look back N days (default 7)",
                        "default": 7,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "email_fetch",
            "description": "Fetch full body of a specific email by UID for summarisation",
            "parameters": {
                "type": "object",
                "properties": {
                    "uid": {"type": "string", "description": "Email UID from search results"}
                },
                "required": ["uid"],
            },
        },
    },
    # ── Finance ──────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "finance_spending_summary",
            "description": (
                "Get spending summary for a time period: total, by category, top merchants. "
                "Use for: 'how much did I spend', 'was habe ich ausgegeben', "
                "'spending this week/month', 'Ausgaben diese Woche'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "Number of days to look back (7=week, 30=month)",
                        "default": 7,
                    },
                    "category": {
                        "type": "string",
                        "description": "Filter by category name (optional)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finance_budget_overview",
            "description": (
                "Show budget limits vs actual spending. "
                "Use for: 'budget status', 'am I over budget', 'Budgetübersicht'"
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finance_account_balances",
            "description": (
                "Show current balances across all accounts. "
                "Use for: 'account balance', 'how much money', 'Kontostand'"
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    # ── Paperless-ngx ────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "paperless_search_documents",
            "description": (
                "Search digitised documents by text, correspondent, or tags. "
                "Use for: 'find invoice from X', 'show my documents', 'Dokumente suchen', 'Rechnung von X'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Full-text search query"},
                    "correspondent": {
                        "type": "string",
                        "description": "Correspondent/sender name (e.g. 'Rewe', 'Telekom')",
                    },
                    "days_back": {
                        "type": "integer",
                        "description": "Limit to documents from last N days",
                        "default": 30,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "paperless_receipt_summary",
            "description": (
                "Summarise receipts from a specific shop/correspondent with totals. "
                "Use for: 'what did I buy at Rewe', 'Rewe Einkäufe', "
                "'how much did I spend at Lidl', 'Einkäufe diese Woche'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "correspondent": {
                        "type": "string",
                        "description": "Shop name (e.g. 'Rewe', 'Lidl', 'Aldi')",
                    },
                    "days_back": {
                        "type": "integer",
                        "description": "Look back N days",
                        "default": 7,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "paperless_get_document",
            "description": "Fetch and read the content of a specific document by ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {"type": "integer", "description": "Paperless document ID"}
                },
                "required": ["document_id"],
            },
        },
    },
]

_SYSTEM_PROMPT = """\
You are a personal assistant with access to the user's private data:
emails, financial transactions, and digitised documents (receipts, invoices).

IMPORTANT: This data is private. Never suggest sending it anywhere.
Respond concisely. Use the tools to fetch live data, then summarise clearly.

Language: Always respond in the same language the user wrote in.
Sprache: Antworte immer in der Sprache des Nutzers (Englisch oder Deutsch).
"""


class PersonalAgent:
    def __init__(self) -> None:
        cfg = get_settings()
        # Use the tool model (4B) for dispatch; it's fast enough
        self._model = cfg.get("models", "tool", "name", default="nemotron-mini")
        self._temperature = 0.0
        self._num_ctx = cfg.get("models", "tool", "num_ctx", default=4096)
        self._num_predict = 768

    async def run(self, state: AgentState) -> AgentState:
        t0 = time.monotonic()
        ollama = get_ollama()

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            *state.history[-8:],
            {"role": "user", "content": state.user_message},
        ]

        # ── First call: let model choose tools ────────────────────────────
        response = await ollama.chat(
            model=self._model,
            messages=messages,
            tools=PERSONAL_TOOLS,
            temperature=self._temperature,
            num_ctx=self._num_ctx,
            num_predict=self._num_predict,
        )

        tool_calls = ollama.extract_tool_calls(response)
        state.tool_calls = tool_calls

        if not tool_calls:
            # Model answered directly (e.g. clarification question)
            state.response = response.get("message", {}).get("content", "")
            state.agent_used = "personal"
            state.latency_ms = (time.monotonic() - t0) * 1000
            return state

        # ── Execute all tool calls concurrently ───────────────────────────
        tool_results = await self._execute_tools_concurrently(tool_calls)

        # ── Second call: synthesise results into a natural response ───────
        messages.append(response.get("message", {}))
        messages.append({"role": "tool", "content": json.dumps(tool_results, ensure_ascii=False)})

        final = await ollama.chat(
            model=self._model,
            messages=messages,
            temperature=0.2,
            num_ctx=self._num_ctx,
            num_predict=self._num_predict,
        )
        state.response = final.get("message", {}).get("content", "")
        state.agent_used = "personal"
        state.latency_ms = (time.monotonic() - t0) * 1000
        log.info(
            "personal_agent.done",
            tools=[c["name"] for c in tool_calls],
            latency_ms=round(state.latency_ms),
        )
        return state

    # ── Concurrent tool execution ─────────────────────────────────────────

    async def _execute_tools_concurrently(
        self, calls: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        tasks = [self._execute_tool(c["name"], c.get("arguments", {})) for c in calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        output = []
        for call, result in zip(calls, results):
            if isinstance(result, Exception):
                log.error("personal_agent.tool_error", tool=call["name"], error=str(result))
                output.append({"tool": call["name"], "error": str(result)})
            else:
                output.append({"tool": call["name"], "result": result})
        return output

    async def _execute_tool(self, name: str, args: dict[str, Any]) -> Any:
        today = date.today()
        days = int(args.get("days_back", 7))
        since = today - timedelta(days=days)

        # ── Email tools ───────────────────────────────────────────────────
        if name == "email_inbox_summary":
            return await get_email().get_inbox_summary()

        if name == "email_search":
            return await get_email().search_messages(
                sender=args.get("sender"),
                subject_contains=args.get("subject_contains"),
                since=since,
            )

        if name == "email_fetch":
            return await get_email().fetch_message(args["uid"])

        # ── Finance tools ─────────────────────────────────────────────────
        if name == "finance_spending_summary":
            return await get_finance().spending_summary(
                days=days,
                category=args.get("category"),
            )

        if name == "finance_budget_overview":
            start = today.replace(day=1)
            budgets = await get_finance().list_budgets(start=start, end=today)
            return {"budgets": budgets}

        if name == "finance_account_balances":
            return await get_finance().get_balance()

        # ── Paperless tools ───────────────────────────────────────────────
        if name == "paperless_search_documents":
            date_after = today - timedelta(days=int(args.get("days_back", 30)))
            docs = await get_paperless().search(
                query=args.get("query"),
                correspondent=args.get("correspondent"),
                date_after=date_after,
            )
            # Return lightweight summaries only (avoid overflowing context)
            return [
                {
                    "id": d["id"],
                    "title": d.get("title", ""),
                    "correspondent": d.get("correspondent", ""),
                    "created": d.get("created", ""),
                    "tags": d.get("tags", []),
                }
                for d in docs[:10]
            ]

        if name == "paperless_receipt_summary":
            date_after = today - timedelta(days=days)
            return await get_paperless().summarise_receipts(
                correspondent=args.get("correspondent"),
                days=days,
            )

        if name == "paperless_get_document":
            content = await get_paperless().get_document_content(args["document_id"])
            return {"document_id": args["document_id"], "content": content[:2000]}

        raise ValueError(f"Unknown tool: {name!r}")
