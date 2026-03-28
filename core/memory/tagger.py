"""
MealTagger — classifies a meal description into structured tags.

Pipeline (each stage only runs if the previous didn't reach confidence threshold):

  Stage 1 — Ordered detection      (rule-based, instant)
            "Lieferando Pizza" → home_cooked=False, effort=none
  Stage 2 — Inheritance detection  (rule-based, instant)
            "Reste vom Curry" → inherit referenced meal + effort=minimal
  Stage 3 — Exact dish lookup      (hash lookup, instant)
            "Käsespätzle" → full TagSet from catalog, confidence=1.0
  Stage 4 — Fuzzy dish match       (rapidfuzz, <5ms)
            "Kaese spaetzle" → matches käsespätzle, confidence=0.85
  Stage 5 — Keyword category scan  (regex/string scan, <1ms)
            "Pasta mit Sahnesoße" → cuisine=italian, heaviness=heavy
  Stage 6 — Modifier application   (rule-based, instant)
            "mit Salat" → shifts heaviness down one step
  Stage 7 — LLM fallback           (Qwen 2.5 1.5B, ~5s, only if confidence < threshold)
            Ambiguous or unknown descriptions
  Stage 8 — Proposal generation    (if LLM was used, queue for catalog)
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from core.memory.catalog import CatalogManager, get_catalog
from core.config import get_settings

log = structlog.get_logger(__name__)

# Confidence threshold below which we fall back to the LLM
_LLM_FALLBACK_THRESHOLD = 0.55
# Threshold above which LLM proposals are auto-accepted into the catalog
_AUTO_ACCEPT_THRESHOLD = 0.92

# Heaviness scale for shift operations
_HEAVINESS_SCALE = ["very_light", "light", "medium", "heavy", "very_heavy"]
_EFFORT_SCALE    = ["none", "minimal", "low", "medium", "high"]


@dataclass
class TagSet:
    """Full classification result for one meal description."""
    raw_input: str = ""
    cuisine: str | None = None
    effort: str | None = None
    heaviness: str | None = None
    meal_type: str | list[str] | None = None
    dietary: list[str] = field(default_factory=list)
    home_cooked: bool | None = None
    confidence: float = 0.0
    source: str = ""                 # which stage produced the result
    inherited_from: str | None = None
    llm_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_input": self.raw_input,
            "cuisine": self.cuisine,
            "effort": self.effort,
            "heaviness": self.heaviness,
            "meal_type": self.meal_type,
            "dietary": self.dietary,
            "home_cooked": self.home_cooked,
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "inherited_from": self.inherited_from,
        }

    def is_complete(self) -> bool:
        """True if all core dimensions are filled."""
        return all([self.cuisine, self.effort, self.heaviness])


# ── Tagger ────────────────────────────────────────────────────────────────────

class MealTagger:
    def __init__(self, catalog: CatalogManager | None = None) -> None:
        self._catalog = catalog or get_catalog()
        cfg = get_settings()
        self._tagger_model = cfg.get("models", "tagger", "name", default="qwen2.5:1.5b")
        self._llm_threshold = cfg.get(
            "tagger", "llm_fallback_threshold", default=_LLM_FALLBACK_THRESHOLD
        )

    # ── Public API ────────────────────────────────────────────────────────

    async def tag(
        self,
        description: str,
        meal_history: list[dict] | None = None,
        session_id: str = "",
    ) -> TagSet:
        """
        Tag a meal description. Runs synchronous stages first,
        falls back to async LLM only when necessary.

        meal_history: recent meals [{date, description, tags}] used for inheritance.
        """
        t0 = time.monotonic()
        text = description.strip()
        result = TagSet(raw_input=text)

        # Stage 1: ordered food
        if self._catalog.is_ordered(text):
            result.home_cooked = False
            result.effort = "none"
            result.confidence = 0.95
            # Still run keyword scan for cuisine
            kw = self._catalog.keyword_match(text)
            result.cuisine = kw.get("cuisine")
            result.heaviness = kw.get("heaviness")
            result.meal_type = kw.get("meal_type")
            result.source = "ordered_rule"
            result.confidence = 0.9
            log.debug("tagger.ordered", text=text)
            self._log_result(result, t0)
            return result

        # Stage 2: inheritance ("Reste vom Curry")
        if self._catalog.is_inheritance(text) and meal_history:
            inherited = self._resolve_inheritance(text, meal_history)
            if inherited:
                result = TagSet(
                    raw_input=text,
                    cuisine=inherited.get("cuisine"),
                    effort="minimal",          # always minimal for leftovers
                    heaviness=inherited.get("heaviness"),
                    meal_type=inherited.get("meal_type"),
                    dietary=inherited.get("dietary", []),
                    home_cooked=True,
                    confidence=0.88,
                    source="inheritance",
                    inherited_from=inherited.get("raw_input"),
                )
                log.debug("tagger.inherited", from_meal=result.inherited_from)
                self._log_result(result, t0)
                return result

        # Stage 3: exact dish lookup
        dish = self._catalog.lookup_dish_exact(text)
        if dish:
            result = self._from_dish(text, dish, confidence=0.97, source="exact_match")
            result = self._apply_modifiers(text, result)
            self._log_result(result, t0)
            return result

        # Stage 4: fuzzy dish match
        dish, score = self._catalog.lookup_dish_fuzzy(text, threshold=78)
        if dish and score:
            conf = score / 100.0 * 0.9    # scale fuzzy score to confidence
            result = self._from_dish(text, dish, confidence=conf, source=f"fuzzy_{score}")
            result = self._apply_modifiers(text, result)
            if result.confidence >= self._llm_threshold:
                self._log_result(result, t0)
                return result

        # Stage 5: keyword category scan
        kw_result = self._keyword_stage(text)
        if kw_result.confidence > result.confidence:
            result = kw_result

        # Stage 6: apply modifiers (may already be applied above)
        result = self._apply_modifiers(text, result)

        # Stage 7: LLM fallback
        if result.confidence < self._llm_threshold or not result.is_complete():
            log.debug("tagger.llm_fallback", text=text, confidence=result.confidence)
            llm_result = await self._llm_fallback(text)
            if llm_result and llm_result.confidence > result.confidence:
                result = llm_result
                result = self._apply_modifiers(text, result)
                # Stage 8: propose to catalog
                if result.confidence >= 0.65:
                    self._catalog.add_proposal(
                        trigger_text=text,
                        proposed_tags=result.to_dict(),
                        confidence=result.confidence,
                        source="llm",
                        auto_accept_threshold=_AUTO_ACCEPT_THRESHOLD,
                    )

        self._log_result(result, t0)
        return result

    def tag_sync(self, description: str) -> TagSet:
        """Synchronous version — skips LLM fallback. Use for bulk processing."""
        text = description.strip()
        result = TagSet(raw_input=text)

        if self._catalog.is_ordered(text):
            kw = self._catalog.keyword_match(text)
            result.home_cooked = False
            result.effort = "none"
            result.cuisine = kw.get("cuisine")
            result.heaviness = kw.get("heaviness")
            result.meal_type = kw.get("meal_type")
            result.source = "ordered_rule"
            result.confidence = 0.9
            return result

        dish = self._catalog.lookup_dish_exact(text)
        if dish:
            return self._apply_modifiers(text, self._from_dish(text, dish, 0.97, "exact_match"))

        dish, score = self._catalog.lookup_dish_fuzzy(text, threshold=78)
        if dish and score >= 78:
            r = self._from_dish(text, dish, score / 100.0 * 0.9, f"fuzzy_{score}")
            return self._apply_modifiers(text, r)

        kw_result = self._keyword_stage(text)
        return self._apply_modifiers(text, kw_result)

    # ── Stage helpers ──────────────────────────────────────────────────────

    def _from_dish(
        self, text: str, dish, confidence: float, source: str
    ) -> TagSet:
        tags = dish.tags
        meal_type = tags.get("meal_type")
        if isinstance(meal_type, list):
            meal_type = meal_type[0] if meal_type else None
        return TagSet(
            raw_input=text,
            cuisine=tags.get("cuisine"),
            effort=tags.get("effort"),
            heaviness=tags.get("heaviness"),
            meal_type=meal_type,
            dietary=tags.get("dietary") or [],
            home_cooked=True,
            confidence=confidence,
            source=source,
        )

    def _keyword_stage(self, text: str) -> TagSet:
        kw = self._catalog.keyword_match(text)
        filled = sum(1 for v in kw.values() if v)
        total_dims = 4   # cuisine, effort, heaviness, meal_type
        confidence = (filled / total_dims) * 0.75 if filled else 0.1
        meal_type = kw.get("meal_type")
        dietary = []
        if kw.get("dietary"):
            dietary = [kw["dietary"]]
        return TagSet(
            raw_input=text,
            cuisine=kw.get("cuisine"),
            effort=kw.get("effort"),
            heaviness=kw.get("heaviness"),
            meal_type=meal_type,
            dietary=dietary,
            home_cooked=None,
            confidence=confidence,
            source="keyword",
        )

    def _apply_modifiers(self, text: str, result: TagSet) -> TagSet:
        shifts = self._catalog.get_modifiers(text)
        if "heaviness" in shifts and result.heaviness:
            result.heaviness = _shift_scale(
                _HEAVINESS_SCALE, result.heaviness, shifts["heaviness"]
            )
        if "effort" in shifts and result.effort:
            result.effort = _shift_scale(
                _EFFORT_SCALE, result.effort, shifts["effort"]
            )
        # "selbstgemacht" in text → override home_cooked
        if any(t in text.lower() for t in ["selbstgemacht", "selbst gemacht", "homemade", "home made"]):
            result.home_cooked = True
            if result.effort in ("none", "minimal"):
                result.effort = "medium"
        return result

    def _resolve_inheritance(
        self, text: str, meal_history: list[dict]
    ) -> dict | None:
        """
        Find the meal being referenced in text like "Reste vom Curry".
        Returns the tags of that meal, or None.
        """
        text_lower = text.lower()
        # Extract the referenced dish name after the trigger word
        from core.memory.catalog import _CORRECTIONS_PATH
        import re
        # e.g. "Reste vom Curry" → "curry"
        m = re.search(
            r"(?:reste?\s+vom?|leftovers?\s+from|aufgewärmte?\s+|übrig\s+von)\s+(.+)",
            text_lower,
        )
        if not m:
            return None
        ref = m.group(1).strip()
        # Find closest match in recent history
        for meal in reversed(meal_history):
            desc = meal.get("description", "").lower()
            if ref in desc or desc in ref:
                return meal.get("tags") or meal
        return None

    # ── LLM fallback ──────────────────────────────────────────────────────

    async def _llm_fallback(self, text: str) -> TagSet | None:
        from core.integrations.ollama_client import get_ollama

        taxonomy = self._catalog.get_taxonomy()
        ollama = get_ollama()

        prompt = f"""\
Classify this meal description. Return valid JSON only, no explanation.

Meal: "{text}"

Rules:
- cuisine must be one of: {taxonomy.get('cuisine', [])}
- effort must be one of: {taxonomy.get('effort', [])}
- heaviness must be one of: {taxonomy.get('heaviness', [])}
- meal_type must be one of: {taxonomy.get('meal_type', [])} or null
- dietary is a list, valid values: {taxonomy.get('dietary', [])}
- home_cooked: true or false

JSON format:
{{"cuisine":"...","effort":"...","heaviness":"...","meal_type":"...","dietary":[],"home_cooked":true,"confidence":0.0}}

confidence is your certainty 0.0-1.0. Be conservative."""

        try:
            resp = await ollama.chat(
                model=self._tagger_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                num_ctx=1024,
                num_predict=128,
            )
            content = resp.get("message", {}).get("content", "").strip()

            # Extract JSON from response (model may wrap it in markdown)
            import re
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if not json_match:
                return None
            data = json.loads(json_match.group())

            # Validate against taxonomy
            taxonomy = self._catalog.get_taxonomy()
            cuisine = data.get("cuisine")
            effort = data.get("effort")
            heaviness = data.get("heaviness")
            if cuisine not in taxonomy.get("cuisine", []):
                cuisine = None
            if effort not in taxonomy.get("effort", []):
                effort = None
            if heaviness not in taxonomy.get("heaviness", []):
                heaviness = None

            return TagSet(
                raw_input=text,
                cuisine=cuisine,
                effort=effort,
                heaviness=heaviness,
                meal_type=data.get("meal_type"),
                dietary=[d for d in (data.get("dietary") or []) if d in taxonomy.get("dietary", [])],
                home_cooked=data.get("home_cooked"),
                confidence=float(data.get("confidence", 0.5)),
                source=f"llm:{self._tagger_model}",
                llm_used=True,
            )
        except Exception as e:
            log.warning("tagger.llm_error", error=str(e), text=text)
            return None

    # ── Helpers ───────────────────────────────────────────────────────────

    def _log_result(self, result: TagSet, t0: float) -> None:
        log.debug(
            "tagger.result",
            input=result.raw_input[:40],
            cuisine=result.cuisine,
            effort=result.effort,
            heaviness=result.heaviness,
            confidence=round(result.confidence, 2),
            source=result.source,
            ms=round((time.monotonic() - t0) * 1000, 1),
        )


# ── Correction API (called by agents when user corrects a tag) ────────────────

async def apply_correction(
    raw_input: str,
    correct_tags: dict[str, Any],
    previous_tags: dict[str, Any],
    session_id: str = "",
) -> None:
    """
    Record a user correction and immediately update in-memory catalog.
    Called from the PersonalAgent or GeneralAgent when the user says
    "actually that was heavy" / "das war eigentlich schwer".
    """
    catalog = get_catalog()
    catalog.record_correction(raw_input, correct_tags, previous_tags, session_id)
    log.info("tagger.correction_applied", input=raw_input, tags=correct_tags)


# ── Scale shift helper ────────────────────────────────────────────────────────

def _shift_scale(scale: list[str], current: str, delta: int) -> str:
    """Shift a value along an ordered scale by delta steps, clamped to bounds."""
    try:
        idx = scale.index(current)
        return scale[max(0, min(len(scale) - 1, idx + delta))]
    except ValueError:
        return current


# ── Singleton ─────────────────────────────────────────────────────────────────

_tagger: MealTagger | None = None


def get_tagger() -> MealTagger:
    global _tagger
    if _tagger is None:
        _tagger = MealTagger()
    return _tagger
