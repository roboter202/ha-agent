"""Central configuration loaded from settings.yaml + env vars."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


class Settings(BaseSettings):
    # ── Service URLs ────────────────────────────────────────────────────────
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")
    qdrant_url: str = Field("http://localhost:6333", env="QDRANT_URL")
    ollama_url: str = Field("http://localhost:11434", env="OLLAMA_URL")
    searxng_url: str = Field("http://localhost:8888", env="SEARXNG_URL")
    n8n_url: str = Field("http://localhost:5678", env="N8N_URL")
    n8n_user: str = Field("admin", env="N8N_BASIC_AUTH_USER")
    n8n_password: str = Field("changeme", env="N8N_BASIC_AUTH_PASSWORD")
    temporal_host: str = Field("localhost:7233", env="TEMPORAL_HOST")
    temporal_namespace: str = Field("ha-agent", env="TEMPORAL_NAMESPACE")

    # ── Home Assistant ──────────────────────────────────────────────────────
    ha_url: str = Field("http://homeassistant.local:8123", env="HA_URL")
    ha_token: str = Field("", env="HA_TOKEN")

    # ── API Keys ────────────────────────────────────────────────────────────
    anthropic_api_key: str = Field("", env="ANTHROPIC_API_KEY")
    openai_api_key: str = Field("", env="OPENAI_API_KEY")

    # ── Logging ─────────────────────────────────────────────────────────────
    log_level: str = Field("INFO", env="LOG_LEVEL")

    # ── YAML config (loaded separately) ─────────────────────────────────────
    _yaml: dict[str, Any] = {}

    model_config = {"env_file": ".env", "extra": "ignore"}

    def __init__(self, **data: Any):
        super().__init__(**data)
        config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
        if config_path.exists():
            self._yaml = _load_yaml(config_path)

    def get(self, *keys: str, default: Any = None) -> Any:
        """Dot-path access into YAML config, e.g. get('models', 'tool', 'name')."""
        obj = self._yaml
        for k in keys:
            if not isinstance(obj, dict):
                return default
            obj = obj.get(k, default)
        return obj


@lru_cache
def get_settings() -> Settings:
    return Settings()
