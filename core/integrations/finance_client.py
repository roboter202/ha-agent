"""
Financial planning service client.
Designed against the Firefly III REST API (v1) – a popular self-hosted
personal finance manager. All calls are read-only (no mutations via this agent).

Firefly III API docs: https://api-docs.firefly-iii.org/

If you use a different service, implement the same interface by subclassing
FinanceClientBase and pointing FINANCE_BACKEND in settings to your class.

Provides:
  - Spending summary for a date range (total, by category, by account)
  - Transaction list (filterable by account, category, budget, merchant)
  - Budget overview (spent vs limit)
  - Account balances
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import httpx
import structlog

from core.config import get_settings

log = structlog.get_logger(__name__)


class FireflyClient:
    """Firefly III REST API client (read-only)."""

    def __init__(self) -> None:
        cfg = get_settings()
        base = cfg.get("finance", "url", default="http://localhost:8011")
        token = cfg.get("finance", "token", default="")
        self._base = base.rstrip("/")
        self._http = httpx.AsyncClient(
            base_url=self._base,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.api+json",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(20.0, connect=5.0),
        )

    async def close(self) -> None:
        await self._http.aclose()

    # ── Transactions ──────────────────────────────────────────────────────

    async def list_transactions(
        self,
        start: date | None = None,
        end: date | None = None,
        account_id: int | None = None,
        category_name: str | None = None,
        budget_name: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List transactions with optional filters."""
        params: dict[str, Any] = {"limit": limit, "page": 1}
        if start:
            params["start"] = start.isoformat()
        if end:
            params["end"] = end.isoformat()

        endpoint = "/api/v1/transactions"
        if account_id:
            endpoint = f"/api/v1/accounts/{account_id}/transactions"

        resp = await self._http.get(endpoint, params=params)
        resp.raise_for_status()
        raw = resp.json().get("data", [])

        txns = []
        for item in raw:
            attrs = item.get("attributes", {})
            for split in attrs.get("transactions", []):
                # Filter by category
                if category_name:
                    cat = split.get("category_name", "").lower()
                    if category_name.lower() not in cat:
                        continue
                # Filter by budget
                if budget_name:
                    bgt = split.get("budget_name", "").lower()
                    if budget_name.lower() not in bgt:
                        continue
                txns.append(
                    {
                        "id": item.get("id"),
                        "date": split.get("date", ""),
                        "description": split.get("description", ""),
                        "amount": float(split.get("amount", 0)),
                        "currency": split.get("currency_symbol", "€"),
                        "category": split.get("category_name", ""),
                        "budget": split.get("budget_name", ""),
                        "account": split.get("source_name", ""),
                        "destination": split.get("destination_name", ""),
                        "type": attrs.get("transactions", [{}])[0].get("type", ""),
                    }
                )
        return txns

    async def spending_summary(
        self,
        days: int = 7,
        start: date | None = None,
        end: date | None = None,
    ) -> dict[str, Any]:
        """Aggregate spending for the period, grouped by category."""
        end = end or date.today()
        start = start or (end - timedelta(days=days))

        txns = await self.list_transactions(start=start, end=end)
        # Only withdrawals (expenses)
        expenses = [t for t in txns if t["type"] in ("withdrawal", "Withdrawal", "expense")]

        total = sum(t["amount"] for t in expenses)
        by_category: dict[str, float] = {}
        by_merchant: dict[str, float] = {}

        for t in expenses:
            cat = t["category"] or "Uncategorized"
            by_category[cat] = round(by_category.get(cat, 0.0) + t["amount"], 2)
            dest = t["destination"] or "Unknown"
            by_merchant[dest] = round(by_merchant.get(dest, 0.0) + t["amount"], 2)

        return {
            "period": {"start": start.isoformat(), "end": end.isoformat(), "days": days},
            "total_spent": round(total, 2),
            "currency": "€",
            "transaction_count": len(expenses),
            "by_category": dict(
                sorted(by_category.items(), key=lambda x: x[1], reverse=True)
            ),
            "top_merchants": dict(
                sorted(by_merchant.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
        }

    async def monthly_summary(self, month: date | None = None) -> dict[str, Any]:
        """Spending summary for a full calendar month."""
        ref = month or date.today()
        start = ref.replace(day=1)
        if ref.month == 12:
            end = ref.replace(year=ref.year + 1, month=1, day=1) - timedelta(days=1)
        else:
            end = ref.replace(month=ref.month + 1, day=1) - timedelta(days=1)
        days = (end - start).days + 1
        return await self.spending_summary(days=days, start=start, end=end)

    # ── Budgets ───────────────────────────────────────────────────────────

    async def list_budgets(self, start: date | None = None, end: date | None = None) -> list[dict]:
        params: dict[str, Any] = {}
        if start:
            params["start"] = start.isoformat()
        if end:
            params["end"] = end.isoformat()
        resp = await self._http.get("/api/v1/budgets", params=params)
        resp.raise_for_status()
        raw = resp.json().get("data", [])
        budgets = []
        for item in raw:
            attrs = item.get("attributes", {})
            spent = 0.0
            limit = 0.0
            for bl in attrs.get("spent", []):
                spent += abs(float(bl.get("sum", 0)))
            for bl in attrs.get("auto_budget_amount", []):
                limit = float(bl) if bl else 0.0
            budgets.append(
                {
                    "id": item.get("id"),
                    "name": attrs.get("name", ""),
                    "limit": attrs.get("auto_budget_amount") or limit,
                    "spent": round(spent, 2),
                    "remaining": round((limit or 0) - spent, 2),
                }
            )
        return budgets

    # ── Accounts ──────────────────────────────────────────────────────────

    async def list_accounts(self, account_type: str = "asset") -> list[dict]:
        resp = await self._http.get(
            "/api/v1/accounts", params={"type": account_type}
        )
        resp.raise_for_status()
        raw = resp.json().get("data", [])
        return [
            {
                "id": a.get("id"),
                "name": a.get("attributes", {}).get("name", ""),
                "balance": float(a.get("attributes", {}).get("current_balance", 0)),
                "currency": a.get("attributes", {}).get("currency_symbol", "€"),
                "type": a.get("attributes", {}).get("type", ""),
            }
            for a in raw
        ]

    async def get_balance(self) -> dict[str, Any]:
        """Return total balance across all asset accounts."""
        accounts = await self.list_accounts()
        total = sum(a["balance"] for a in accounts)
        return {
            "total_balance": round(total, 2),
            "currency": "€",
            "accounts": accounts,
        }


_client: FireflyClient | None = None


def get_finance() -> FireflyClient:
    global _client
    if _client is None:
        _client = FireflyClient()
    return _client
