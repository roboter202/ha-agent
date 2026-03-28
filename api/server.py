"""
FastAPI application entry point.
- REST endpoint  POST /chat
- WebSocket      ws://host/ws/{session_id}
- Health/info    GET /health, GET /info
- Startup: seed HA state cache, warm up models, start WS subscription
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from core.config import get_settings
from core.integrations.home_assistant import HAClient
from core.integrations.ollama_client import get_ollama
from core.memory.cache import get_cache
from core.memory.rag import get_rag
from core.orchestrator import Orchestrator
from api.routes import build_router

log = structlog.get_logger(__name__)

# Module-level singletons shared across requests
_ha_client: HAClient | None = None
_orchestrator: Orchestrator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ha_client, _orchestrator
    cfg = get_settings()

    # ── Start services ────────────────────────────────────────────────────
    log.info("startup.begin")

    cache = get_cache()
    assert await cache.ping(), "Redis not reachable"

    rag = get_rag()
    await rag.ensure_collections()

    _ha_client = HAClient()
    await _ha_client.start()

    # Seed HA state cache from REST API
    try:
        all_states = await _ha_client.get_all_states()
        await cache.states.seed(all_states)
        await rag.ingest_ha_states(all_states)
        log.info("startup.ha_states_seeded", count=len(all_states))
    except Exception as e:
        log.warning("startup.ha_seed_failed", error=str(e))

    # Register WS state update callback
    _ha_client.on_state_changed(cache.states.on_state_changed)

    # Start HA WebSocket subscription (background task)
    asyncio.create_task(_ha_client.subscribe_events())

    # Warm up Ollama models (non-blocking)
    ollama = get_ollama()
    tool_model = cfg.get("models", "tool", "name", default="nemotron-mini")
    general_model = cfg.get("models", "general", "name", default="qwen2.5:7b")
    asyncio.create_task(ollama.warmup(tool_model))
    asyncio.create_task(ollama.warmup(general_model))

    _orchestrator = Orchestrator(_ha_client)
    log.info("startup.complete")

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────
    log.info("shutdown.begin")
    if _ha_client:
        await _ha_client.stop()
    ollama = get_ollama()
    await ollama.close()
    rag = get_rag()
    await rag.close()
    await cache.close()
    log.info("shutdown.complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="HA Multi-Agent Assistant",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(build_router(lambda: _orchestrator, lambda: _ha_client))

    @app.websocket("/ws/{session_id}")
    async def websocket_endpoint(ws: WebSocket, session_id: str):
        await ws.accept()
        log.info("ws.connected", session=session_id)
        try:
            while True:
                text = await ws.receive_text()
                if not text.strip():
                    continue

                orch = _orchestrator
                if orch is None:
                    await ws.send_json({"error": "Service not ready"})
                    continue

                # Check if client wants streaming
                use_stream = text.startswith("stream:")
                message = text[7:].strip() if use_stream else text

                if use_stream:
                    from core.agents.general_agent import GeneralAgent
                    from core.agents.base_agent import AgentState, RouteType
                    from core.memory.cache import get_cache

                    cache = get_cache()
                    history = await cache.history.get(session_id)
                    state = AgentState(
                        session_id=session_id,
                        user_message=message,
                        history=history,
                        route=RouteType.GENERAL,
                        streaming=True,
                    )
                    agent = GeneralAgent()
                    full_response = ""
                    async for chunk in agent.stream(state, _ha_client):
                        await ws.send_json({"chunk": chunk})
                        full_response += chunk
                    await ws.send_json({"done": True, "response": full_response})
                    await cache.history.add(session_id, "user", message)
                    await cache.history.add(session_id, "assistant", full_response)
                else:
                    state = await orch.run(session_id, message)
                    await ws.send_json(
                        {
                            "response": state.response,
                            "agent": state.agent_used,
                            "latency_ms": round(state.latency_ms, 1),
                            "route": state.route,
                        }
                    )
        except WebSocketDisconnect:
            log.info("ws.disconnected", session=session_id)
        except Exception as e:
            log.error("ws.error", session=session_id, error=str(e))
            try:
                await ws.send_json({"error": str(e)})
            except Exception:
                pass

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, loop="uvloop")
