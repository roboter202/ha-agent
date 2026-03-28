"""
Redis cache layer.
- HA state mirror (updated live by WS subscription)
- Conversation history per session
- Response deduplication / memoization
"""

from __future__ import annotations

import json
import time
from typing import Any

import redis.asyncio as aioredis
import structlog

from core.config import get_settings

log = structlog.get_logger(__name__)

_STATES_KEY = "ha:states"          # Hash: entity_id → JSON state
_HISTORY_PREFIX = "session:hist:"  # List per session_id
_MEMO_PREFIX = "memo:"             # Response cache


class StateCache:
    """Manages HA entity state in Redis."""

    def __init__(self, redis: aioredis.Redis) -> None:
        self._r = redis

    # Called by HA WebSocket listener for every state_changed event
    def on_state_changed(self, event: dict) -> None:
        data = event.get("data", {})
        entity_id = data.get("entity_id")
        new_state = data.get("new_state")
        if entity_id and new_state:
            import asyncio
            asyncio.create_task(self._update(entity_id, new_state))

    async def _update(self, entity_id: str, state: dict) -> None:
        cfg = get_settings()
        ttl = cfg.get("home_assistant", "state_ttl", default=300)
        await self._r.hset(_STATES_KEY, entity_id, json.dumps(state))
        await self._r.expire(_STATES_KEY, ttl * 10)  # whole hash refreshed TTL

    async def get(self, entity_id: str) -> dict | None:
        raw = await self._r.hget(_STATES_KEY, entity_id)
        return json.loads(raw) if raw else None

    async def get_all(self) -> list[dict]:
        raw = await self._r.hgetall(_STATES_KEY)
        return [json.loads(v) for v in raw.values()]

    async def seed(self, states: list[dict]) -> None:
        """Bulk-load all HA states into Redis on startup."""
        if not states:
            return
        pipe = self._r.pipeline()
        for s in states:
            eid = s.get("entity_id")
            if eid:
                pipe.hset(_STATES_KEY, eid, json.dumps(s))
        await pipe.execute()
        log.info("state_cache.seeded", count=len(states))


class ConversationCache:
    """Per-session rolling conversation history."""

    def __init__(self, redis: aioredis.Redis, window: int = 20) -> None:
        self._r = redis
        self._window = window

    async def add(self, session_id: str, role: str, content: str) -> None:
        key = _HISTORY_PREFIX + session_id
        entry = json.dumps({"role": role, "content": content, "ts": time.time()})
        await self._r.rpush(key, entry)
        await self._r.ltrim(key, -self._window * 2, -1)
        await self._r.expire(key, 3600 * 24)

    async def get(self, session_id: str) -> list[dict]:
        key = _HISTORY_PREFIX + session_id
        raw = await self._r.lrange(key, 0, -1)
        return [json.loads(r) for r in raw]

    async def clear(self, session_id: str) -> None:
        await self._r.delete(_HISTORY_PREFIX + session_id)


class ResponseMemo:
    """Short-lived response memoization for identical quick queries."""

    def __init__(self, redis: aioredis.Redis) -> None:
        self._r = redis

    async def get(self, key: str) -> str | None:
        val = await self._r.get(_MEMO_PREFIX + key)
        return val.decode() if val else None

    async def set(self, key: str, value: str, ttl: int = 30) -> None:
        await self._r.set(_MEMO_PREFIX + key, value, ex=ttl)


class Cache:
    """Unified cache facade."""

    def __init__(self) -> None:
        cfg = get_settings()
        self._r: aioredis.Redis = aioredis.from_url(
            cfg.redis_url, decode_responses=True
        )
        self.states = StateCache(self._r)
        cfg_window = cfg.get("agent", "history_window", default=20)
        self.history = ConversationCache(self._r, window=cfg_window)
        self.memo = ResponseMemo(self._r)

    async def ping(self) -> bool:
        try:
            return await self._r.ping()
        except Exception:
            return False

    async def close(self) -> None:
        await self._r.aclose()

    # Convenience re-exports
    async def get_all_ha_states(self) -> list[dict]:
        return await self.states.get_all()

    async def get_ha_state(self, entity_id: str) -> dict | None:
        return await self.states.get(entity_id)


_cache: Cache | None = None


def get_cache() -> Cache:
    global _cache
    if _cache is None:
        _cache = Cache()
    return _cache
