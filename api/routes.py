"""HTTP routes (REST API)."""

from __future__ import annotations

import uuid
from typing import Callable

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.agents.base_agent import AgentState
from core.memory.cache import get_cache


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    stream: bool = False


class ChatResponse(BaseModel):
    response: str
    session_id: str
    agent: str
    latency_ms: float
    route: str | None = None


class WorkflowRequest(BaseModel):
    workflow_name: str
    payload: dict = {}


def build_router(
    get_orchestrator: Callable,
    get_ha: Callable,
) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    async def health():
        cache = get_cache()
        redis_ok = await cache.ping()
        return {
            "status": "ok" if redis_ok else "degraded",
            "redis": redis_ok,
        }

    @router.get("/info")
    async def info():
        from core.config import get_settings
        cfg = get_settings()
        return {
            "tool_model": cfg.get("models", "tool", "name"),
            "general_model": cfg.get("models", "general", "name"),
            "ha_url": cfg.ha_url,
        }

    @router.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest):
        orch = get_orchestrator()
        if orch is None:
            raise HTTPException(503, "Service not ready")

        session_id = req.session_id or str(uuid.uuid4())
        state: AgentState = await orch.run(session_id, req.message)

        return ChatResponse(
            response=state.response or "(no response)",
            session_id=session_id,
            agent=state.agent_used,
            latency_ms=round(state.latency_ms, 1),
            route=state.route,
        )

    @router.post("/workflow")
    async def trigger_workflow(req: WorkflowRequest):
        from core.integrations.n8n_client import get_n8n
        n8n = get_n8n()
        result = await n8n.trigger(req.workflow_name, req.payload)
        return result

    @router.get("/ha/states")
    async def get_ha_states(domain: str | None = None):
        cache = get_cache()
        states = await cache.get_all_ha_states()
        if domain:
            states = [s for s in states if s.get("entity_id", "").startswith(domain + ".")]
        return states

    @router.get("/ha/state/{entity_id:path}")
    async def get_entity_state(entity_id: str):
        cache = get_cache()
        state = await cache.get_ha_state(entity_id)
        if not state:
            ha = get_ha()
            state = await ha.get_state(entity_id)
        if not state:
            raise HTTPException(404, f"Entity {entity_id!r} not found")
        return state

    @router.delete("/session/{session_id}")
    async def clear_session(session_id: str):
        cache = get_cache()
        await cache.history.clear(session_id)
        return {"cleared": session_id}

    return router
