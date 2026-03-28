"""
Home Assistant client.
- REST API for one-shot calls (service calls, entity queries)
- WebSocket for real-time state subscriptions (feeds Redis state cache)
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator, Callable

import aiohttp
import structlog

from core.config import get_settings

log = structlog.get_logger(__name__)


class HAClient:
    """Async Home Assistant REST + WebSocket client."""

    def __init__(self) -> None:
        cfg = get_settings()
        self._base_url = cfg.ha_url.rstrip("/")
        self._token = cfg.ha_token
        self._headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }
        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._ws_id = 1
        self._state_callbacks: list[Callable[[dict], None]] = []

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._session = aiohttp.ClientSession(
            headers=self._headers,
            connector=aiohttp.TCPConnector(limit=20),
            timeout=aiohttp.ClientTimeout(total=15),
        )
        log.info("ha_client.started", url=self._base_url)

    async def stop(self) -> None:
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()

    # ── REST helpers ──────────────────────────────────────────────────────

    async def get_state(self, entity_id: str) -> dict[str, Any] | None:
        """Fetch current state of a single entity."""
        async with self._session.get(
            f"{self._base_url}/api/states/{entity_id}"
        ) as r:
            if r.status == 200:
                return await r.json()
            log.warning("ha.get_state.miss", entity_id=entity_id, status=r.status)
            return None

    async def get_all_states(self) -> list[dict[str, Any]]:
        """Fetch all entity states."""
        async with self._session.get(f"{self._base_url}/api/states") as r:
            r.raise_for_status()
            return await r.json()

    async def call_service(
        self,
        domain: str,
        service: str,
        service_data: dict[str, Any] | None = None,
        target: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Call a HA service and return the resulting states."""
        payload: dict[str, Any] = {}
        if service_data:
            payload.update(service_data)
        if target:
            payload["target"] = target

        async with self._session.post(
            f"{self._base_url}/api/services/{domain}/{service}",
            json=payload,
        ) as r:
            r.raise_for_status()
            try:
                return await r.json()
            except Exception:
                return []

    async def get_areas(self) -> list[dict[str, Any]]:
        async with self._session.post(
            f"{self._base_url}/api/template",
            json={"template": "{{ areas() | list | tojson }}"},
        ) as r:
            if r.status == 200:
                text = await r.text()
                return json.loads(text)
            return []

    async def get_entities_in_area(self, area_id: str) -> list[str]:
        tmpl = "{{ area_entities('" + area_id + "') | list | tojson }}"
        async with self._session.post(
            f"{self._base_url}/api/template", json={"template": tmpl}
        ) as r:
            if r.status == 200:
                return json.loads(await r.text())
            return []

    async def fire_event(self, event_type: str, event_data: dict | None = None) -> None:
        async with self._session.post(
            f"{self._base_url}/api/events/{event_type}",
            json=event_data or {},
        ) as r:
            r.raise_for_status()

    # ── WebSocket subscription ────────────────────────────────────────────

    def on_state_changed(self, callback: Callable[[dict], None]) -> None:
        """Register a callback for state_changed events."""
        self._state_callbacks.append(callback)

    async def subscribe_events(self) -> None:
        """Connect to HA WebSocket and stream state_changed events to callbacks.
        Runs forever; reconnects on disconnect."""
        ws_url = self._base_url.replace("http", "ws") + "/api/websocket"
        while True:
            try:
                await self._ws_connect(ws_url)
            except Exception as e:
                log.warning("ha.ws.disconnected", error=str(e))
                await asyncio.sleep(5)

    async def _ws_connect(self, ws_url: str) -> None:
        async with self._session.ws_connect(ws_url) as ws:
            self._ws = ws
            self._ws_id = 1

            # Authenticate
            auth_req = await ws.receive_json()
            assert auth_req["type"] == "auth_required"
            await ws.send_json({"type": "auth", "access_token": self._token})
            auth_ok = await ws.receive_json()
            assert auth_ok["type"] == "auth_ok", f"Auth failed: {auth_ok}"
            log.info("ha.ws.authenticated")

            # Subscribe to state_changed
            await ws.send_json(
                {
                    "id": self._ws_id,
                    "type": "subscribe_events",
                    "event_type": "state_changed",
                }
            )
            self._ws_id += 1

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if data.get("type") == "event":
                        event = data["event"]
                        for cb in self._state_callbacks:
                            try:
                                cb(event)
                            except Exception as e:
                                log.error("ha.ws.callback_error", error=str(e))
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break

    # ── Entity resolution helpers ─────────────────────────────────────────

    async def resolve_entity(
        self, name: str, domain: str | None = None
    ) -> str | None:
        """Fuzzy-resolve a human name to an entity_id using cached states.
        Falls back to direct API if cache misses."""
        from core.memory.cache import get_cache

        cache = get_cache()
        states = await cache.get_all_ha_states()
        if not states:
            states = await self.get_all_states()

        name_lower = name.lower().replace(" ", "_")
        candidates = []
        for s in states:
            eid = s.get("entity_id", "")
            if domain and not eid.startswith(domain + "."):
                continue
            friendly = (
                s.get("attributes", {}).get("friendly_name", "").lower().replace(" ", "_")
            )
            entity_slug = eid.split(".")[1] if "." in eid else eid
            if name_lower in entity_slug or name_lower in friendly:
                candidates.append(eid)
            elif entity_slug in name_lower or friendly in name_lower:
                candidates.append(eid)

        if not candidates:
            return None
        # prefer exact match, else first
        for c in candidates:
            if name_lower == c.split(".")[1]:
                return c
        return candidates[0]
