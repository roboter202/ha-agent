"""SearXNG search client for the research agent."""

from __future__ import annotations

from typing import Any

import httpx
import structlog

from core.config import get_settings

log = structlog.get_logger(__name__)


class SearXNGClient:
    def __init__(self) -> None:
        cfg = get_settings()
        self._base = cfg.searxng_url.rstrip("/")
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=5.0)
        )

    async def close(self) -> None:
        await self._http.aclose()

    async def search(
        self,
        query: str,
        num_results: int = 8,
        categories: list[str] | None = None,
        engines: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Run a search and return list of {title, url, content} dicts."""
        params: dict[str, Any] = {
            "q": query,
            "format": "json",
            "pageno": 1,
        }
        if categories:
            params["categories"] = ",".join(categories)
        if engines:
            params["engines"] = ",".join(engines)

        try:
            resp = await self._http.get(f"{self._base}/search", params=params)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                    "score": r.get("score", 0),
                }
                for r in results[:num_results]
            ]
        except Exception as e:
            log.error("searxng.search_failed", query=query, error=str(e))
            return []

    async def search_news(self, query: str, num_results: int = 5) -> list[dict[str, Any]]:
        return await self.search(query, num_results=num_results, categories=["news"])

    async def search_general(self, query: str, num_results: int = 8) -> list[dict[str, Any]]:
        return await self.search(query, num_results=num_results, categories=["general"])


_client: SearXNGClient | None = None


def get_searxng() -> SearXNGClient:
    global _client
    if _client is None:
        _client = SearXNGClient()
    return _client
