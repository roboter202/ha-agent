"""n8n webhook client – triggers predefined non-AI workflows."""

from __future__ import annotations

from typing import Any

import httpx
import structlog

from core.config import get_settings

log = structlog.get_logger(__name__)


class N8NClient:
    def __init__(self) -> None:
        cfg = get_settings()
        self._base = cfg.n8n_url.rstrip("/")
        self._http = httpx.AsyncClient(
            auth=(cfg.n8n_user, cfg.n8n_password),
            timeout=httpx.Timeout(30.0, connect=5.0),
        )
        # webhook_name → path, loaded from settings
        cfg_yaml = cfg.get("n8n", "workflow_webhooks") or {}
        self._webhooks: dict[str, str] = cfg_yaml

    async def close(self) -> None:
        await self._http.aclose()

    async def trigger(
        self,
        workflow_name: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Trigger an n8n webhook by workflow name and return the response."""
        path = self._webhooks.get(workflow_name)
        if not path:
            raise ValueError(f"Unknown n8n workflow: {workflow_name!r}")

        url = f"{self._base}{path}"
        log.info("n8n.trigger", workflow=workflow_name, url=url)
        resp = await self._http.post(url, json=payload or {})
        if resp.status_code in (200, 201):
            try:
                return resp.json()
            except Exception:
                return {"status": "ok"}
        else:
            log.error("n8n.trigger_failed", workflow=workflow_name, status=resp.status_code)
            return {"status": "error", "code": resp.status_code}

    def list_workflows(self) -> list[str]:
        return list(self._webhooks.keys())


_client: N8NClient | None = None


def get_n8n() -> N8NClient:
    global _client
    if _client is None:
        _client = N8NClient()
    return _client
