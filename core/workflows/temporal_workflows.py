"""
Temporal durable workflow definitions.
These handle complex, long-running, or retry-critical home automation scenarios.

Workflows:
  - MultiStepAutomation : execute ordered HA service calls with conditions + delays
  - ScheduledRoutine    : time-based recurring automation
  - DeviceMonitor       : watch a device state and react when threshold crossed
  - EnergyOptimization  : adjust climate/appliances based on energy data
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

import structlog
from temporalio import activity, workflow
from temporalio.common import RetryPolicy

log = structlog.get_logger(__name__)

# ── Activities (the actual work) ──────────────────────────────────────────────


@activity.defn
async def call_ha_service_activity(
    domain: str,
    service: str,
    entity_id: str | None,
    service_data: dict[str, Any],
) -> dict[str, Any]:
    """Activity: call a single HA service. Retried automatically by Temporal."""
    from core.integrations.home_assistant import HAClient
    from core.config import get_settings

    cfg = get_settings()
    ha = HAClient()
    await ha.start()
    try:
        target = {"entity_id": entity_id} if entity_id else {}
        result = await ha.call_service(domain, service, service_data, target)
        return {"status": "ok", "result": result}
    finally:
        await ha.stop()


@activity.defn
async def get_ha_state_activity(entity_id: str) -> dict[str, Any]:
    from core.integrations.home_assistant import HAClient

    ha = HAClient()
    await ha.start()
    try:
        state = await ha.get_state(entity_id)
        return state or {}
    finally:
        await ha.stop()


@activity.defn
async def trigger_n8n_activity(workflow_name: str, payload: dict[str, Any]) -> dict:
    from core.integrations.n8n_client import get_n8n

    n8n = get_n8n()
    return await n8n.trigger(workflow_name, payload)


@activity.defn
async def wait_for_state_activity(
    entity_id: str, target_state: str, timeout_seconds: int = 300
) -> bool:
    """Poll entity state until it matches target or timeout."""
    from core.memory.cache import get_cache

    cache = get_cache()
    deadline = asyncio.get_event_loop().time() + timeout_seconds
    while asyncio.get_event_loop().time() < deadline:
        state = await cache.get_ha_state(entity_id)
        if state and state.get("state") == target_state:
            return True
        await asyncio.sleep(5)
    return False


# ── Workflow definitions ──────────────────────────────────────────────────────


@workflow.defn(name="multi_step_automation")
class MultiStepAutomation:
    """
    Execute a sequence of HA service calls with optional delays and conditions.
    Input: {"steps": [...], "session_id": "..."}
    Step format:
      {"domain": "light", "service": "turn_on", "entity_id": "light.kitchen",
       "service_data": {}, "delay_after": 2.0,
       "condition": {"entity_id": "sensor.motion", "state": "on"}}
    """

    @workflow.run
    async def run(self, args: dict[str, Any]) -> dict[str, Any]:
        steps: list[dict] = args.get("steps", [])
        results = []

        for i, step in enumerate(steps):
            # Optional condition check
            if cond := step.get("condition"):
                state = await workflow.execute_activity(
                    get_ha_state_activity,
                    args=[cond["entity_id"]],
                    start_to_close_timeout=timedelta(seconds=10),
                    retry_policy=RetryPolicy(maximum_attempts=3),
                )
                if state.get("state") != cond.get("state"):
                    results.append({"step": i, "skipped": True, "reason": "condition not met"})
                    continue

            result = await workflow.execute_activity(
                call_ha_service_activity,
                args=[
                    step["domain"],
                    step["service"],
                    step.get("entity_id"),
                    step.get("service_data", {}),
                ],
                start_to_close_timeout=timedelta(seconds=15),
                retry_policy=RetryPolicy(maximum_attempts=3, backoff_coefficient=2.0),
            )
            results.append({"step": i, "result": result})

            delay = step.get("delay_after", 0)
            if delay > 0:
                await asyncio.sleep(delay)

        return {"status": "completed", "results": results}


@workflow.defn(name="scheduled_routine")
class ScheduledRoutine:
    """
    Trigger an n8n webhook on a schedule.
    Input: {"workflow_name": "morning_routine", "interval_seconds": 86400}
    """

    @workflow.run
    async def run(self, args: dict[str, Any]) -> dict[str, Any]:
        workflow_name = args["workflow_name"]
        payload = args.get("payload", {})

        result = await workflow.execute_activity(
            trigger_n8n_activity,
            args=[workflow_name, payload],
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(maximum_attempts=5, backoff_coefficient=2.0),
        )
        return {"status": "triggered", "result": result}


@workflow.defn(name="device_monitor")
class DeviceMonitor:
    """
    Watch a device state and execute actions when a threshold is crossed.
    Input: {"entity_id": "sensor.temperature", "threshold": 25, "operator": "gt",
            "action_steps": [...]}
    """

    @workflow.run
    async def run(self, args: dict[str, Any]) -> dict[str, Any]:
        entity_id = args["entity_id"]
        threshold = float(args.get("threshold", 0))
        operator = args.get("operator", "gt")
        action_steps = args.get("action_steps", [])

        state = await workflow.execute_activity(
            get_ha_state_activity,
            args=[entity_id],
            start_to_close_timeout=timedelta(seconds=10),
        )

        try:
            current_value = float(state.get("state", 0))
        except ValueError:
            return {"status": "error", "reason": "non-numeric state"}

        triggered = (
            (operator == "gt" and current_value > threshold)
            or (operator == "lt" and current_value < threshold)
            or (operator == "eq" and current_value == threshold)
        )

        if triggered and action_steps:
            return await workflow.execute_child_workflow(
                MultiStepAutomation.run,
                args=[{"steps": action_steps}],
                id=f"device-monitor-action-{entity_id}",
            )
        return {"status": "ok", "triggered": triggered, "value": current_value}


@workflow.defn(name="energy_optimization")
class EnergyOptimization:
    """
    Adjust thermostat and switchable loads based on current energy/tariff data.
    Input: {"high_tariff": bool, "outdoor_temp": float}
    """

    @workflow.run
    async def run(self, args: dict[str, Any]) -> dict[str, Any]:
        high_tariff = args.get("high_tariff", False)
        outdoor_temp = float(args.get("outdoor_temp", 20))

        steps = []
        if high_tariff:
            # Reduce non-essential loads
            steps.append(
                {
                    "domain": "climate",
                    "service": "set_temperature",
                    "entity_id": None,
                    "service_data": {"temperature": 19 if outdoor_temp < 10 else 22},
                    "delay_after": 0,
                }
            )

        if steps:
            return await workflow.execute_child_workflow(
                MultiStepAutomation.run,
                args=[{"steps": steps}],
                id="energy-optimization-action",
            )
        return {"status": "no_action"}
