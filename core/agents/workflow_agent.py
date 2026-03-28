"""
Workflow Agent – bridges AI-side decisions to n8n and Temporal.

n8n handles:  Predefined non-AI workflows (routines, scenes, notifications)
Temporal handles: Complex durable multi-step AI workflows that need retries,
                  scheduling, long-running coordination.
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from core.agents.base_agent import AgentState
from core.config import get_settings
from core.integrations.n8n_client import get_n8n

log = structlog.get_logger(__name__)


class WorkflowAgent:
    def __init__(self) -> None:
        cfg = get_settings()
        self._temporal_host = cfg.temporal_host
        self._temporal_namespace = cfg.temporal_namespace
        self._task_queue = cfg.get("temporal", "task_queue", default="ha-agent-tasks")

    async def run(self, state: AgentState) -> AgentState:
        t0 = time.monotonic()
        cfg = get_settings()

        workflow_name = state.slots.get("workflow_name") or state.intent
        if not workflow_name:
            state.response = "I'm not sure which workflow to run."
            return state

        # Check if it's a Temporal durable workflow
        durable = cfg.get("temporal", "durable_workflows") or []
        if workflow_name in durable:
            state.response = await self._run_temporal(workflow_name, state)
            state.agent_used = "workflow+temporal"
        else:
            # n8n webhook
            state.response = await self._run_n8n(workflow_name, state)
            state.agent_used = "workflow+n8n"

        state.latency_ms = (time.monotonic() - t0) * 1000
        return state

    async def _run_n8n(self, workflow_name: str, state: AgentState) -> str:
        n8n = get_n8n()
        payload = {
            "session_id": state.session_id,
            "user_message": state.user_message,
            **state.slots,
        }
        result = await n8n.trigger(workflow_name, payload)
        if result.get("status") == "error":
            return f"Workflow '{workflow_name}' failed."
        msg = result.get("message", "")
        return msg or f"{workflow_name.replace('_', ' ').title()} started."

    async def _run_temporal(self, workflow_name: str, state: AgentState) -> str:
        try:
            from temporalio.client import Client

            client = await Client.connect(
                self._temporal_host, namespace=self._temporal_namespace
            )
            handle = await client.start_workflow(
                workflow_name,
                args=[
                    {
                        "session_id": state.session_id,
                        "user_message": state.user_message,
                        "slots": state.slots,
                    }
                ],
                id=f"{workflow_name}-{state.session_id}-{int(time.time())}",
                task_queue=self._task_queue,
            )
            log.info("temporal.workflow_started", id=handle.id, name=workflow_name)
            return f"Started '{workflow_name.replace('_', ' ')}' workflow (id: {handle.id})."
        except Exception as e:
            log.error("temporal.start_failed", workflow=workflow_name, error=str(e))
            return f"Could not start workflow: {e}"
