"""Shared state schema for the LangGraph agent graph."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RouteType(str, Enum):
    INSTANT = "instant"      # < 50ms: pattern match → direct HA call
    TOOL = "tool"            # < 800ms: nemotron-3-nano tool call
    GENERAL = "general"      # < 3s: Qwen 3.5 9B + RAG
    RESEARCH = "research"    # 5-15s: SearXNG + optional online LLM
    WORKFLOW = "workflow"    # async: n8n webhook or Temporal durable workflow
    PERSONAL = "personal"    # < 2s: email / finance / Paperless-ngx (private data, no online LLM)


class AgentState(BaseModel):
    """Shared state flowing through the LangGraph graph."""

    # Input
    session_id: str = ""
    user_message: str = ""

    # Routing
    route: RouteType | None = None
    intent: str | None = None          # matched intent name (instant path)
    slots: dict[str, Any] = Field(default_factory=dict)

    # Context
    history: list[dict[str, str]] = Field(default_factory=list)
    ha_context: str = ""               # relevant HA states as text
    rag_context: str = ""              # RAG search results as text

    # Output
    response: str = ""
    streaming: bool = False
    error: str | None = None

    # Metadata
    latency_ms: float = 0.0
    agent_used: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    needs_online_llm: bool = False
