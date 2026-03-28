"""
Temporal worker entry point.
Run with: python -m core.workflows.temporal_worker
"""

from __future__ import annotations

import asyncio

import structlog
from temporalio.client import Client
from temporalio.worker import Worker

from core.config import get_settings
from core.workflows.temporal_workflows import (
    MultiStepAutomation,
    ScheduledRoutine,
    DeviceMonitor,
    EnergyOptimization,
    call_ha_service_activity,
    get_ha_state_activity,
    trigger_n8n_activity,
    wait_for_state_activity,
)

log = structlog.get_logger(__name__)


async def main() -> None:
    cfg = get_settings()
    client = await Client.connect(
        cfg.temporal_host,
        namespace=cfg.temporal_namespace,
    )
    log.info("temporal_worker.connected", host=cfg.temporal_host)

    worker = Worker(
        client,
        task_queue=cfg.get("temporal", "task_queue", default="ha-agent-tasks"),
        workflows=[
            MultiStepAutomation,
            ScheduledRoutine,
            DeviceMonitor,
            EnergyOptimization,
        ],
        activities=[
            call_ha_service_activity,
            get_ha_state_activity,
            trigger_n8n_activity,
            wait_for_state_activity,
        ],
    )
    log.info("temporal_worker.starting")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
