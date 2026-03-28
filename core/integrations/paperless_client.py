"""
Paperless-ngx integration.
Provides document search, receipt parsing, and correspondent/tag filtering.
API reference: https://docs.paperless-ngx.com/api/

Key capabilities used here:
  - Full-text document search
  - Filter by correspondent (e.g. "Rewe"), tag, document type, date range
  - Retrieve document content for LLM summarisation
  - Extract structured data from receipts (total, date, line items)
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any

import httpx
import structlog

from core.config import get_settings

log = structlog.get_logger(__name__)


class PaperlessClient:
    def __init__(self) -> None:
        cfg = get_settings()
        base = cfg.get("paperless", "url", default="http://localhost:8010")
        token = cfg.get("paperless", "token", default="")
        self._base = base.rstrip("/")
        self._http = httpx.AsyncClient(
            base_url=self._base,
            headers={"Authorization": f"Token {token}", "Accept": "application/json"},
            timeout=httpx.Timeout(30.0, connect=5.0),
        )

    async def close(self) -> None:
        await self._http.aclose()

    # ── Documents ─────────────────────────────────────────────────────────

    async def search(
        self,
        query: str | None = None,
        correspondent: str | None = None,
        tags: list[str] | None = None,
        document_type: str | None = None,
        date_after: date | None = None,
        date_before: date | None = None,
        page_size: int = 20,
    ) -> list[dict[str, Any]]:
        """Search documents with optional filters."""
        params: dict[str, Any] = {"page_size": page_size, "ordering": "-created"}
        if query:
            params["query"] = query
        if correspondent:
            # Resolve correspondent name → ID first
            cid = await self._resolve_correspondent(correspondent)
            if cid:
                params["correspondent__id"] = cid
        if tags:
            tag_ids = []
            for tag in tags:
                tid = await self._resolve_tag(tag)
                if tid:
                    tag_ids.append(tid)
            if tag_ids:
                params["tags__id__all"] = ",".join(str(t) for t in tag_ids)
        if date_after:
            params["created__date__gt"] = date_after.isoformat()
        if date_before:
            params["created__date__lt"] = date_before.isoformat()

        resp = await self._http.get("/api/documents/", params=params)
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])

    async def get_document(self, doc_id: int) -> dict[str, Any]:
        resp = await self._http.get(f"/api/documents/{doc_id}/")
        resp.raise_for_status()
        return resp.json()

    async def get_document_content(self, doc_id: int) -> str:
        """Return the full OCR-extracted text of a document."""
        doc = await self.get_document(doc_id)
        return doc.get("content", "")

    # ── Receipt-specific helpers ──────────────────────────────────────────

    async def get_receipts(
        self,
        correspondent: str | None = None,
        date_after: date | None = None,
        date_before: date | None = None,
        page_size: int = 20,
    ) -> list[dict[str, Any]]:
        """Return documents tagged as receipts."""
        return await self.search(
            correspondent=correspondent,
            tags=["receipt", "kassenbon", "quittung"],  # EN + DE tags
            date_after=date_after,
            date_before=date_before,
            page_size=page_size,
        )

    async def parse_receipt_total(self, doc_id: int) -> float | None:
        """Extract total amount from OCR text using regex."""
        content = await self.get_document_content(doc_id)
        # Patterns for German/English receipt totals
        patterns = [
            r"(?:gesamt|total|summe|betrag|endbetrag|zu zahlen)[^\d]*(\d+[.,]\d{2})",
            r"(?:EUR|€)\s*(\d+[.,]\d{2})",
            r"(\d+[.,]\d{2})\s*(?:EUR|€)\s*$",
        ]
        for pattern in patterns:
            m = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
            if m:
                raw = m.group(1).replace(",", ".")
                try:
                    return float(raw)
                except ValueError:
                    continue
        return None

    async def summarise_receipts(
        self,
        correspondent: str | None = None,
        days: int = 7,
    ) -> dict[str, Any]:
        """
        Return a summary dict for receipts in the last N days:
          {correspondent, period_days, count, total_eur, documents: [...]}
        """
        date_after = date.today() - timedelta(days=days)
        docs = await self.get_receipts(
            correspondent=correspondent, date_after=date_after
        )
        total = 0.0
        parsed = []
        for doc in docs:
            amount = await self.parse_receipt_total(doc["id"])
            parsed.append(
                {
                    "id": doc["id"],
                    "title": doc.get("title", ""),
                    "created": doc.get("created", ""),
                    "correspondent": doc.get("correspondent", ""),
                    "total_eur": amount,
                }
            )
            if amount:
                total += amount

        return {
            "correspondent": correspondent or "all",
            "period_days": days,
            "count": len(parsed),
            "total_eur": round(total, 2),
            "documents": parsed,
        }

    # ── Correspondents & Tags ─────────────────────────────────────────────

    async def list_correspondents(self) -> list[dict[str, Any]]:
        resp = await self._http.get("/api/correspondents/", params={"page_size": 100})
        resp.raise_for_status()
        return resp.json().get("results", [])

    async def list_tags(self) -> list[dict[str, Any]]:
        resp = await self._http.get("/api/tags/", params={"page_size": 100})
        resp.raise_for_status()
        return resp.json().get("results", [])

    async def _resolve_correspondent(self, name: str) -> int | None:
        correspondents = await self.list_correspondents()
        name_lower = name.lower()
        for c in correspondents:
            if name_lower in c.get("name", "").lower():
                return c["id"]
        return None

    async def _resolve_tag(self, name: str) -> int | None:
        tags = await self.list_tags()
        name_lower = name.lower()
        for t in tags:
            if name_lower in t.get("name", "").lower():
                return t["id"]
        return None

    # ── Generic document query ────────────────────────────────────────────

    async def find_document(self, description: str) -> list[dict[str, Any]]:
        """Best-effort semantic search via full-text query."""
        return await self.search(query=description, page_size=5)


_client: PaperlessClient | None = None


def get_paperless() -> PaperlessClient:
    global _client
    if _client is None:
        _client = PaperlessClient()
    return _client
