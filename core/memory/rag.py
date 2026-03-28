"""
RAG system using Qdrant + fastembed.
Collections:
  - ha_documentation : HA docs / device manuals
  - home_context     : rooms, devices, routines, user preferences
  - conversation_memory : long-term user preferences derived from chats
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

import structlog
from fastembed import TextEmbedding
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from core.config import get_settings

log = structlog.get_logger(__name__)

VECTOR_SIZE = 384   # BAAI/bge-small-en-v1.5


class RAG:
    def __init__(self) -> None:
        cfg = get_settings()
        self._client = AsyncQdrantClient(url=cfg.qdrant_url)
        self._embed_model = TextEmbedding(
            model_name=cfg.get("models", "embedding", "name",
                               default="BAAI/bge-small-en-v1.5")
        )
        self._collections: dict[str, str] = cfg.get("rag", "collections") or {
            "ha_docs": "ha_documentation",
            "home_context": "home_context",
            "conversation": "conversation_memory",
        }
        self._top_k: int = cfg.get("rag", "top_k", default=5)
        self._min_score: float = cfg.get("rag", "min_score", default=0.6)

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def ensure_collections(self) -> None:
        existing = {c.name for c in (await self._client.get_collections()).collections}
        for col_name in self._collections.values():
            if col_name not in existing:
                await self._client.create_collection(
                    collection_name=col_name,
                    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
                )
                log.info("rag.collection_created", name=col_name)

    async def close(self) -> None:
        await self._client.close()

    # ── Embed ─────────────────────────────────────────────────────────────

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._embed_model.embed(texts)]

    # ── Upsert ────────────────────────────────────────────────────────────

    async def upsert(
        self,
        collection_key: str,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> None:
        col = self._collections.get(collection_key, collection_key)
        metadatas = metadatas or [{} for _ in texts]
        vectors = self._embed(texts)
        points = []
        for i, (text, vec, meta) in enumerate(zip(texts, vectors, metadatas)):
            uid = ids[i] if ids else hashlib.sha256(text.encode()).hexdigest()[:16]
            points.append(
                PointStruct(
                    id=abs(hash(uid)) % (2**63),
                    vector=vec,
                    payload={**meta, "text": text, "indexed_at": time.time()},
                )
            )
        await self._client.upsert(collection_name=col, points=points)

    # ── Search ────────────────────────────────────────────────────────────

    async def search(
        self,
        collection_key: str,
        query: str,
        top_k: int | None = None,
        filter_: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        col = self._collections.get(collection_key, collection_key)
        k = top_k or self._top_k
        vec = self._embed([query])[0]

        qdrant_filter = None
        if filter_:
            must = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_.items()
            ]
            qdrant_filter = Filter(must=must)

        results = await self._client.search(
            collection_name=col,
            query_vector=vec,
            limit=k,
            score_threshold=self._min_score,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        return [
            {
                "text": r.payload.get("text", ""),
                "score": r.score,
                "metadata": {k: v for k, v in r.payload.items() if k != "text"},
            }
            for r in results
        ]

    async def search_home_context(self, query: str) -> list[dict[str, Any]]:
        return await self.search("home_context", query)

    async def search_ha_docs(self, query: str) -> list[dict[str, Any]]:
        return await self.search("ha_docs", query)

    # ── Home context ingestion ─────────────────────────────────────────────

    async def ingest_ha_states(self, states: list[dict]) -> None:
        """Convert HA entity states into searchable text chunks."""
        texts, metas = [], []
        for s in states:
            eid = s.get("entity_id", "")
            attrs = s.get("attributes", {})
            friendly = attrs.get("friendly_name", eid)
            state_val = s.get("state", "unknown")
            text = (
                f"Device: {friendly} (entity: {eid})\n"
                f"Current state: {state_val}\n"
                f"Attributes: {attrs}"
            )
            texts.append(text)
            metas.append({"entity_id": eid, "domain": eid.split(".")[0]})

        if texts:
            await self.upsert("home_context", texts, metadatas=metas)
            log.info("rag.ingested_states", count=len(texts))


_rag: RAG | None = None


def get_rag() -> RAG:
    global _rag
    if _rag is None:
        _rag = RAG()
    return _rag
