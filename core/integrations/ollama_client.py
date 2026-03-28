"""
Ollama client wrapper.
Provides streaming and non-streaming completions + tool/function calling.
Models are kept hot (OLLAMA_KEEP_ALIVE=-1) so first-token latency stays low.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx
import structlog

from core.config import get_settings

log = structlog.get_logger(__name__)


class OllamaClient:
    """Thin async Ollama client with tool-call support."""

    def __init__(self) -> None:
        cfg = get_settings()
        self._base = cfg.ollama_url.rstrip("/")
        self._http = httpx.AsyncClient(
            base_url=self._base,
            timeout=httpx.Timeout(120.0, connect=5.0),
        )

    async def close(self) -> None:
        await self._http.aclose()

    # ── Model management ──────────────────────────────────────────────────

    async def ensure_model(self, model: str) -> None:
        """Pull model if not present (idempotent)."""
        resp = await self._http.get("/api/tags")
        tags = {m["name"] for m in resp.json().get("models", [])}
        if model not in tags and model.split(":")[0] not in {t.split(":")[0] for t in tags}:
            log.info("ollama.pulling", model=model)
            async with self._http.stream("POST", "/api/pull", json={"name": model}) as r:
                async for line in r.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if data.get("status") == "success":
                            log.info("ollama.pulled", model=model)
                            break

    async def warmup(self, model: str) -> None:
        """Load model into memory with an empty prompt."""
        try:
            await self._http.post(
                "/api/generate",
                json={"model": model, "prompt": "", "keep_alive": -1},
                timeout=60.0,
            )
            log.info("ollama.warmed_up", model=model)
        except Exception as e:
            log.warning("ollama.warmup_failed", model=model, error=str(e))

    # ── Chat completion (non-streaming) ───────────────────────────────────

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        num_ctx: int = 4096,
        num_predict: int = 512,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
            },
        }
        if tools:
            payload["tools"] = tools

        resp = await self._http.post("/api/chat", json=payload, timeout=120.0)
        resp.raise_for_status()
        return resp.json()

    # ── Streaming chat ────────────────────────────────────────────────────

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.3,
        num_ctx: int = 8192,
        num_predict: int = 1024,
    ) -> AsyncIterator[str]:
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
            },
        }
        async with self._http.stream(
            "POST", "/api/chat", json=payload, timeout=120.0
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                content = data.get("message", {}).get("content", "")
                if content:
                    yield content
                if data.get("done"):
                    break

    # ── Tool call extraction ───────────────────────────────────────────────

    @staticmethod
    def extract_tool_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
        """Return list of {name, arguments} from an Ollama chat response."""
        msg = response.get("message", {})
        calls = msg.get("tool_calls", [])
        result = []
        for call in calls:
            fn = call.get("function", {})
            result.append(
                {
                    "name": fn.get("name", ""),
                    "arguments": fn.get("arguments", {}),
                }
            )
        return result


_client: OllamaClient | None = None


def get_ollama() -> OllamaClient:
    global _client
    if _client is None:
        _client = OllamaClient()
    return _client
