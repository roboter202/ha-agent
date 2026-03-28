"""
Catalog consolidation — weekly Temporal workflow.

What it does:
  1. Reads all corrections from the last N days
  2. Finds patterns: same raw_input corrected consistently to same tags
  3. Uses the 1.5B tagger LLM to propose new keyword rules for the catalog
  4. Writes proposals to the queue; high-confidence ones are auto-accepted
  5. Logs a summary that the notification agent can surface to the user

Schedule: run weekly via Temporal scheduled workflow or a cron n8n trigger.
Can also be triggered manually via POST /catalog/consolidate.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

import structlog
from temporalio import activity, workflow

log = structlog.get_logger(__name__)

_CORRECTIONS_PATH = Path(__file__).parent.parent.parent / "data" / "catalog_corrections.jsonl"


# ── Activities ────────────────────────────────────────────────────────────────

@activity.defn
async def load_recent_corrections_activity(days: int = 7) -> list[dict]:
    """Read corrections from the last N days."""
    if not _CORRECTIONS_PATH.exists():
        return []
    cutoff = time.time() - days * 86400
    corrections = []
    with open(_CORRECTIONS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if c.get("timestamp", 0) >= cutoff:
                corrections.append(c)
    log.info("consolidation.corrections_loaded", count=len(corrections), days=days)
    return corrections


@activity.defn
async def find_consistent_patterns_activity(
    corrections: list[dict],
) -> list[dict]:
    """
    Group corrections by raw_input. If ≥2 corrections agree on tags,
    flag as a consistent pattern ready for catalog promotion.
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    for c in corrections:
        grouped[c["raw_input"].lower().strip()].append(c)

    patterns = []
    for raw_input, entries in grouped.items():
        if len(entries) < 2:
            continue
        # Check if ≥2 entries agree on the same tags
        tag_counts: dict[str, int] = defaultdict(int)
        for e in entries:
            key = json.dumps(e["tags"], sort_keys=True)
            tag_counts[key] += 1
        best_key, best_count = max(tag_counts.items(), key=lambda x: x[1])
        if best_count >= 2:
            patterns.append({
                "raw_input": raw_input,
                "agreed_tags": json.loads(best_key),
                "agreement_count": best_count,
                "total_corrections": len(entries),
                "confidence": min(0.98, 0.7 + best_count * 0.1),
            })
    log.info("consolidation.patterns_found", count=len(patterns))
    return patterns


@activity.defn
async def generate_keyword_proposals_activity(
    corrections: list[dict],
    existing_keywords: list[str],
) -> list[dict]:
    """
    Ask the 1.5B LLM to look at uncovered corrections and suggest
    new keyword rules (not full dish entries — just keyword → tag mappings).
    """
    from core.integrations.ollama_client import get_ollama
    from core.config import get_settings

    cfg = get_settings()
    model = cfg.get("models", "tagger", "name", default="qwen2.5:1.5b")
    ollama = get_ollama()

    # Only process corrections that weren't covered by existing keywords
    uncovered = [c for c in corrections if c["raw_input"] not in existing_keywords][:20]
    if not uncovered:
        return []

    examples = "\n".join(
        f'- "{c["raw_input"]}" → {json.dumps(c["tags"], ensure_ascii=False)}'
        for c in uncovered
    )

    prompt = f"""\
You are building a meal classification catalog.
These meal descriptions were manually corrected by the user.
Suggest new keyword rules that would cover similar meals automatically.

Corrected examples:
{examples}

For each pattern you identify, return a JSON object:
{{"keyword": "...", "dimension": "cuisine|effort|heaviness|meal_type|dietary", "value": "..."}}

Return a JSON array of rule objects only. Keep keywords short (1-3 words).
Only suggest rules you are confident about."""

    try:
        resp = await ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            num_ctx=2048,
            num_predict=256,
        )
        content = resp.get("message", {}).get("content", "").strip()
        import re
        json_match = re.search(r"\[.*\]", content, re.DOTALL)
        if not json_match:
            return []
        proposals = json.loads(json_match.group())
        valid = [
            p for p in proposals
            if isinstance(p, dict)
            and p.get("keyword")
            and p.get("dimension")
            and p.get("value")
        ]
        log.info("consolidation.keyword_proposals", count=len(valid))
        return valid
    except Exception as e:
        log.error("consolidation.llm_error", error=str(e))
        return []


@activity.defn
async def apply_patterns_to_catalog_activity(
    patterns: list[dict],
    keyword_proposals: list[dict],
) -> dict[str, Any]:
    """Promote consistent patterns and keyword proposals to the catalog."""
    from core.memory.catalog import get_catalog

    catalog = get_catalog()
    promoted = 0
    proposed = 0

    # Promote consistent patterns (high confidence → auto-accept)
    for p in patterns:
        conf = p["confidence"]
        catalog.add_proposal(
            trigger_text=p["raw_input"],
            proposed_tags=p["agreed_tags"],
            confidence=conf,
            source="consolidation_pattern",
            auto_accept_threshold=0.88,
        )
        if conf >= 0.88:
            promoted += 1
        else:
            proposed += 1

    # Keyword proposals go into the pending queue for manual review
    # (they modify the categories dict, not just add dishes)
    # Write them separately for now — a future UI can apply them
    if keyword_proposals:
        proposals_path = (
            Path(__file__).parent.parent.parent / "data" / "keyword_proposals.jsonl"
        )
        proposals_path.parent.mkdir(parents=True, exist_ok=True)
        with open(proposals_path, "a", encoding="utf-8") as f:
            for kp in keyword_proposals:
                f.write(json.dumps({**kp, "timestamp": time.time(), "reviewed": False},
                                   ensure_ascii=False) + "\n")

    return {
        "auto_promoted": promoted,
        "queued_for_review": proposed,
        "keyword_proposals": len(keyword_proposals),
    }


@activity.defn
async def get_catalog_stats_activity() -> dict[str, Any]:
    from core.memory.catalog import get_catalog
    return get_catalog().stats()


# ── Workflow ──────────────────────────────────────────────────────────────────

@workflow.defn(name="catalog_consolidation")
class CatalogConsolidation:
    """
    Weekly catalog consolidation.
    Input: {"days": 7}  (default: look back 7 days)
    """

    @workflow.run
    async def run(self, args: dict[str, Any]) -> dict[str, Any]:
        days = int(args.get("days", 7))
        t0 = workflow.now()

        # 1. Load recent corrections
        corrections = await workflow.execute_activity(
            load_recent_corrections_activity,
            args=[days],
            start_to_close_timeout=timedelta(seconds=30),
        )

        if not corrections:
            return {"status": "nothing_to_do", "corrections": 0}

        # 2. Find consistent patterns + generate keyword proposals (parallel)
        patterns, keyword_proposals = await asyncio.gather(
            workflow.execute_activity(
                find_consistent_patterns_activity,
                args=[corrections],
                start_to_close_timeout=timedelta(seconds=30),
            ),
            workflow.execute_activity(
                generate_keyword_proposals_activity,
                args=[corrections, []],
                start_to_close_timeout=timedelta(seconds=120),
            ),
        )

        # 3. Apply to catalog
        result = await workflow.execute_activity(
            apply_patterns_to_catalog_activity,
            args=[patterns, keyword_proposals],
            start_to_close_timeout=timedelta(seconds=30),
        )

        # 4. Get updated stats
        stats = await workflow.execute_activity(
            get_catalog_stats_activity,
            start_to_close_timeout=timedelta(seconds=10),
        )

        summary = {
            "status": "complete",
            "period_days": days,
            "corrections_processed": len(corrections),
            "patterns_found": len(patterns),
            **result,
            "catalog_version": stats.get("version"),
            "catalog_total_dishes": stats.get("dish_entries"),
        }
        log.info("consolidation.complete", **summary)
        return summary


# needed inside the activity that uses asyncio.gather in workflow context
import asyncio
