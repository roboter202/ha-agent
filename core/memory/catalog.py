"""
CatalogManager — loads catalog.yaml and manages its lifecycle.

Responsibilities:
  - Load / hot-reload the catalog from disk
  - Expose lookup helpers (exact dish, fuzzy dish, keyword categories)
  - Accept user corrections and write them to corrections.jsonl
  - Manage the proposal queue (pending entries for review)
  - Bump version and persist changes back to catalog.yaml
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
import yaml

log = structlog.get_logger(__name__)

_CATALOG_PATH    = Path(__file__).parent.parent.parent / "config" / "catalog.yaml"
_CORRECTIONS_PATH = Path(__file__).parent.parent.parent / "data" / "catalog_corrections.jsonl"
_PROPOSALS_PATH  = Path(__file__).parent.parent.parent / "data" / "catalog_proposals.jsonl"


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class DishEntry:
    key: str
    aliases: list[str]
    tags: dict[str, Any]
    source: str = "catalog"          # catalog | correction | proposal
    confidence: float = 1.0
    usage_count: int = 0
    last_used: float = 0.0


@dataclass
class Correction:
    raw_input: str
    tags: dict[str, Any]
    previous_tags: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    session_id: str = ""


@dataclass
class Proposal:
    id: str
    trigger_text: str
    proposed_tags: dict[str, Any]
    confidence: float
    source: str                      # "llm" | "pattern"
    timestamp: float = field(default_factory=time.time)
    auto_accepted: bool = False


# ── CatalogManager ────────────────────────────────────────────────────────────

class CatalogManager:
    def __init__(self, catalog_path: Path = _CATALOG_PATH) -> None:
        self._path = catalog_path
        self._lock = threading.RLock()
        self._raw: dict[str, Any] = {}
        self._dishes: dict[str, DishEntry] = {}     # alias_lower → DishEntry
        self._categories: dict[str, dict[str, list[str]]] = {}
        self._modifiers: dict[str, Any] = {}
        self._inheritance_triggers: list[str] = []
        self._ordered_triggers: list[str] = []
        self._taxonomy: dict[str, list[str]] = {}
        self._loaded_at: float = 0.0
        self.load()

    # ── I/O ───────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load (or reload) catalog.yaml from disk."""
        with self._lock:
            self._raw = yaml.safe_load(self._path.read_text(encoding="utf-8"))
            self._taxonomy = self._raw.get("taxonomy", {})
            self._categories = self._raw.get("categories", {})
            self._modifiers = self._raw.get("modifiers", {})
            self._inheritance_triggers = [
                t.lower() for t in self._raw.get("inheritance_triggers", [])
            ]
            self._ordered_triggers = [
                t.lower() for t in self._raw.get("ordered_triggers", [])
            ]
            self._dishes = {}
            for key, entry in (self._raw.get("dishes") or {}).items():
                de = DishEntry(
                    key=key,
                    aliases=[a.lower() for a in entry.get("aliases", [])],
                    tags=entry.get("tags", {}),
                )
                for alias in de.aliases:
                    self._dishes[alias] = de
            self._loaded_at = time.time()
            log.info(
                "catalog.loaded",
                version=self._raw.get("meta", {}).get("version", 0),
                dishes=len(self._dishes),
            )

    def _save(self) -> None:
        """Persist current _raw back to catalog.yaml."""
        with self._lock:
            self._path.write_text(
                yaml.dump(self._raw, allow_unicode=True, sort_keys=False, default_flow_style=False),
                encoding="utf-8",
            )
            log.info("catalog.saved", version=self._raw.get("meta", {}).get("version", 0))

    # ── Lookups ───────────────────────────────────────────────────────────

    def lookup_dish_exact(self, text: str) -> DishEntry | None:
        """Return the first DishEntry whose alias appears in text."""
        text_lower = text.lower()
        best: DishEntry | None = None
        best_len = 0
        with self._lock:
            for alias, entry in self._dishes.items():
                if alias in text_lower and len(alias) > best_len:
                    best = entry
                    best_len = len(alias)
        return best

    def lookup_dish_fuzzy(self, text: str, threshold: int = 80) -> tuple[DishEntry | None, int]:
        """Fuzzy-match against all dish aliases. Returns (entry, score)."""
        from rapidfuzz import process, fuzz
        text_lower = text.lower()
        with self._lock:
            aliases = list(self._dishes.keys())
        if not aliases:
            return None, 0
        match = process.extractOne(
            text_lower, aliases, scorer=fuzz.partial_ratio
        )
        if match and match[1] >= threshold:
            return self._dishes[match[0]], match[1]
        return None, 0

    def keyword_match(self, text: str) -> dict[str, Any]:
        """
        Scan text against all category keyword lists.
        Returns best-matching tag per dimension.
        Longer keyword = higher priority.
        """
        text_lower = text.lower()
        result: dict[str, Any] = {}
        with self._lock:
            for dimension, categories in self._categories.items():
                best_tag: str | None = None
                best_len = 0
                for tag, keywords in categories.items():
                    for kw in keywords:
                        if kw in text_lower and len(kw) > best_len:
                            best_tag = tag
                            best_len = len(kw)
                if best_tag:
                    result[dimension] = best_tag
        return result

    def get_modifiers(self, text: str) -> dict[str, int]:
        """Return shift values per dimension triggered by modifier keywords."""
        text_lower = text.lower()
        shifts: dict[str, int] = {}
        with self._lock:
            for dimension, mod_groups in self._modifiers.items():
                for mod_name, mod_cfg in mod_groups.items():
                    for trigger in mod_cfg.get("triggers", []):
                        if trigger in text_lower:
                            shift = mod_cfg.get("shift", 0)
                            shifts[dimension] = shifts.get(dimension, 0) + (
                                1 if "+" in str(shift) or shift > 0 else -1
                            )
        return shifts

    def is_ordered(self, text: str) -> bool:
        text_lower = text.lower()
        with self._lock:
            return any(t in text_lower for t in self._ordered_triggers)

    def is_inheritance(self, text: str) -> bool:
        text_lower = text.lower()
        with self._lock:
            return any(t in text_lower for t in self._inheritance_triggers)

    def get_taxonomy(self) -> dict[str, list[str]]:
        with self._lock:
            return dict(self._taxonomy)

    # ── Learning: corrections ─────────────────────────────────────────────

    def record_correction(
        self,
        raw_input: str,
        correct_tags: dict[str, Any],
        previous_tags: dict[str, Any],
        session_id: str = "",
    ) -> None:
        """
        Persist a user correction.
        If the same input has been corrected ≥ 2 times consistently,
        auto-promote it to the dish catalog.
        """
        _CORRECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        correction = Correction(
            raw_input=raw_input.lower().strip(),
            tags=correct_tags,
            previous_tags=previous_tags,
            session_id=session_id,
        )
        with open(_CORRECTIONS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "raw_input": correction.raw_input,
                "tags": correction.tags,
                "previous_tags": correction.previous_tags,
                "timestamp": correction.timestamp,
                "session_id": correction.session_id,
            }, ensure_ascii=False) + "\n")

        log.info("catalog.correction_recorded", input=raw_input)
        self._maybe_promote_correction(raw_input.lower().strip(), correct_tags)

    def _maybe_promote_correction(self, raw_input: str, tags: dict) -> None:
        """Auto-promote to catalog if same input corrected to same tags ≥ 2 times."""
        if not _CORRECTIONS_PATH.exists():
            return
        matching = []
        with open(_CORRECTIONS_PATH, encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                if c["raw_input"] == raw_input and c["tags"] == tags:
                    matching.append(c)
        if len(matching) >= 2:
            self._add_dish_entry(
                key=raw_input.replace(" ", "_"),
                aliases=[raw_input],
                tags=tags,
                source="correction",
            )
            log.info("catalog.auto_promoted", input=raw_input, corrections=len(matching))

    # ── Learning: proposals ───────────────────────────────────────────────

    def add_proposal(
        self,
        trigger_text: str,
        proposed_tags: dict[str, Any],
        confidence: float,
        source: str = "llm",
        auto_accept_threshold: float = 0.92,
    ) -> Proposal:
        """Store a proposed new catalog entry. Auto-accept if confidence is high."""
        import uuid
        _PROPOSALS_PATH.parent.mkdir(parents=True, exist_ok=True)
        proposal = Proposal(
            id=str(uuid.uuid4())[:8],
            trigger_text=trigger_text.lower().strip(),
            proposed_tags=proposed_tags,
            confidence=confidence,
            source=source,
            auto_accepted=confidence >= auto_accept_threshold,
        )
        with open(_PROPOSALS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "id": proposal.id,
                "trigger_text": proposal.trigger_text,
                "proposed_tags": proposal.proposed_tags,
                "confidence": proposal.confidence,
                "source": proposal.source,
                "timestamp": proposal.timestamp,
                "auto_accepted": proposal.auto_accepted,
                "reviewed": proposal.auto_accepted,
            }, ensure_ascii=False) + "\n")

        if proposal.auto_accepted:
            self._add_dish_entry(
                key=proposal.trigger_text.replace(" ", "_"),
                aliases=[proposal.trigger_text],
                tags=proposed_tags,
                source="proposal_auto",
                confidence=confidence,
            )
            log.info("catalog.proposal_auto_accepted", input=trigger_text, score=confidence)
        else:
            log.info("catalog.proposal_queued", input=trigger_text, score=confidence)
            self._bump_pending_count()

        return proposal

    def get_pending_proposals(self) -> list[dict]:
        """Return proposals that haven't been reviewed yet."""
        if not _PROPOSALS_PATH.exists():
            return []
        pending = []
        with open(_PROPOSALS_PATH, encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                if not p.get("reviewed"):
                    pending.append(p)
        return pending

    def accept_proposal(self, proposal_id: str) -> bool:
        """Mark a proposal as accepted and add it to the catalog."""
        if not _PROPOSALS_PATH.exists():
            return False
        lines = _PROPOSALS_PATH.read_text(encoding="utf-8").splitlines()
        updated = []
        found = None
        for line in lines:
            p = json.loads(line)
            if p["id"] == proposal_id and not p.get("reviewed"):
                p["reviewed"] = True
                p["auto_accepted"] = True
                found = p
            updated.append(json.dumps(p, ensure_ascii=False))
        if not found:
            return False
        _PROPOSALS_PATH.write_text("\n".join(updated) + "\n", encoding="utf-8")
        self._add_dish_entry(
            key=found["trigger_text"].replace(" ", "_"),
            aliases=[found["trigger_text"]],
            tags=found["proposed_tags"],
            source="proposal_accepted",
        )
        self._bump_pending_count(delta=-1)
        log.info("catalog.proposal_accepted", id=proposal_id)
        return True

    def reject_proposal(self, proposal_id: str) -> bool:
        if not _PROPOSALS_PATH.exists():
            return False
        lines = _PROPOSALS_PATH.read_text(encoding="utf-8").splitlines()
        updated = []
        for line in lines:
            p = json.loads(line)
            if p["id"] == proposal_id:
                p["reviewed"] = True
                p["auto_accepted"] = False
            updated.append(json.dumps(p, ensure_ascii=False))
        _PROPOSALS_PATH.write_text("\n".join(updated) + "\n", encoding="utf-8")
        self._bump_pending_count(delta=-1)
        return True

    # ── Internal helpers ──────────────────────────────────────────────────

    def _add_dish_entry(
        self,
        key: str,
        aliases: list[str],
        tags: dict,
        source: str = "learned",
        confidence: float = 1.0,
    ) -> None:
        """Add a new dish entry to the live catalog and persist to YAML."""
        with self._lock:
            entry = DishEntry(
                key=key,
                aliases=[a.lower() for a in aliases],
                tags=tags,
                source=source,
                confidence=confidence,
            )
            for alias in entry.aliases:
                self._dishes[alias] = entry

            # Also write into _raw so it persists
            if "dishes" not in self._raw:
                self._raw["dishes"] = {}
            self._raw["dishes"][key] = {
                "aliases": aliases,
                "tags": tags,
            }
            meta = self._raw.setdefault("meta", {})
            meta["auto_entries"] = meta.get("auto_entries", 0) + 1
            meta["version"] = meta.get("version", 1) + 1
            self._save()

    def _bump_pending_count(self, delta: int = 1) -> None:
        with self._lock:
            meta = self._raw.setdefault("meta", {})
            meta["pending_review"] = max(0, meta.get("pending_review", 0) + delta)
            self._save()

    # ── Stats ─────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        with self._lock:
            meta = self._raw.get("meta", {})
        pending = len(self.get_pending_proposals())
        return {
            "version": meta.get("version", 1),
            "dish_entries": len(set(e.key for e in self._dishes.values())),
            "alias_entries": len(self._dishes),
            "pending_proposals": pending,
            "corrections_file": str(_CORRECTIONS_PATH),
            "loaded_at": self._loaded_at,
        }


# ── Singleton ─────────────────────────────────────────────────────────────────

_manager: CatalogManager | None = None


def get_catalog() -> CatalogManager:
    global _manager
    if _manager is None:
        _manager = CatalogManager()
    return _manager
