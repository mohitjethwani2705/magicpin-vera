"""
generate_submission.py — Offline submission JSONL generator for Vera AI.

Loads the base dataset, runs compose() for 30 (merchant, trigger) pairs,
and writes submission.jsonl.

Usage:
    python generate_submission.py

Requires GROQ_API_KEY in environment or .env file.
Output: submission.jsonl (30 lines, one per pair)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap — load .env before any import that needs GROQ_API_KEY
# ---------------------------------------------------------------------------

_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

from bot import compose  # noqa: E402 — must be after .env load

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("vera.generate_submission")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATASET_DIR = Path(__file__).parent / "dataset"
CATEGORIES_DIR = DATASET_DIR / "categories"
MERCHANTS_SEED_PATH = DATASET_DIR / "merchants_seed.json"
TRIGGERS_SEED_PATH = DATASET_DIR / "triggers_seed.json"
OUTPUT_PATH = Path(__file__).parent / "submission.jsonl"

# Canonical test pairs spec: 30 pairs — one trigger per merchant, cycling
# through all 5 categories twice (2 merchants × 3 triggers each = 30 total).
# We build pairs deterministically so the output is reproducible.
PAIRS_PER_CATEGORY = 6   # 5 categories × 6 = 30 pairs
MERCHANTS_PER_CATEGORY = 2
TRIGGERS_PER_MERCHANT = 3


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_categories() -> dict[str, dict]:
    """Load all 5 category JSONs, keyed by slug."""
    categories: dict[str, dict] = {}
    for json_path in sorted(CATEGORIES_DIR.glob("*.json")):
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)
        slug = data.get("slug") or json_path.stem
        categories[slug] = data
    logger.info("Loaded %d categories: %s", len(categories), sorted(categories))
    return categories


def _load_merchants() -> list[dict]:
    """Load merchants_seed.json, return the merchants list."""
    with open(MERCHANTS_SEED_PATH, encoding="utf-8") as fh:
        raw = json.load(fh)
    merchants = raw["merchants"] if isinstance(raw, dict) else raw
    logger.info("Loaded %d seed merchants", len(merchants))
    return merchants


def _load_triggers() -> list[dict]:
    """Load triggers_seed.json, return the triggers list."""
    with open(TRIGGERS_SEED_PATH, encoding="utf-8") as fh:
        raw = json.load(fh)
    triggers = raw["triggers"] if isinstance(raw, dict) else raw
    logger.info("Loaded %d seed triggers", len(triggers))
    return triggers


# ---------------------------------------------------------------------------
# Pair selection
# ---------------------------------------------------------------------------

def _select_pairs(
    merchants: list[dict],
    triggers: list[dict],
    target_count: int = 30,
) -> list[dict]:
    """
    Build up to `target_count` (merchant, trigger) pairs deterministically.

    Strategy:
      1. Group merchants by category_slug.
      2. Build a trigger index: merchant_id -> [triggers for that merchant].
      3. Walk category groups; for each group take 2 merchants, up to 3
         triggers per merchant.  Repeat categories until 30 pairs are reached.
    """
    # Index triggers by merchant_id
    trigger_index: dict[str, list[dict]] = {}
    for trg in triggers:
        mid = trg.get("merchant_id") or trg.get("payload", {}).get("merchant_id")
        if mid:
            trigger_index.setdefault(mid, []).append(trg)

    # Group merchants by category slug
    by_cat: dict[str, list[dict]] = {}
    for m in merchants:
        slug = m.get("category_slug", "unknown")
        by_cat.setdefault(slug, []).append(m)

    pairs: list[dict] = []
    category_order = sorted(by_cat.keys())

    # Cycle through categories until we hit target_count
    while len(pairs) < target_count:
        added_this_pass = 0
        for slug in category_order:
            cat_merchants = by_cat[slug][:MERCHANTS_PER_CATEGORY]
            for merchant in cat_merchants:
                mid = merchant["merchant_id"]
                merchant_triggers = trigger_index.get(mid, [])[:TRIGGERS_PER_MERCHANT]
                if not merchant_triggers:
                    logger.warning("No triggers found for merchant %s — skipping", mid)
                    continue
                for trg in merchant_triggers:
                    if len(pairs) >= target_count:
                        break
                    pairs.append({"merchant": merchant, "trigger": trg})
                    added_this_pass += 1
                if len(pairs) >= target_count:
                    break
            if len(pairs) >= target_count:
                break
        if added_this_pass == 0:
            # No more unique pairs available — stop early
            logger.warning(
                "Exhausted all unique pairs at %d (target was %d)", len(pairs), target_count
            )
            break

    logger.info("Selected %d pairs for composition", len(pairs))
    return pairs[:target_count]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=== Vera AI submission generator starting ===")

    categories = _load_categories()
    merchants = _load_merchants()
    triggers = _load_triggers()

    pairs = _select_pairs(merchants, triggers, target_count=30)

    if not pairs:
        logger.error("No pairs to compose — aborting.")
        sys.exit(1)

    results: list[dict] = []
    errors = 0

    for i, pair in enumerate(pairs, start=1):
        merchant = pair["merchant"]
        trigger = pair["trigger"]
        merchant_id = merchant["merchant_id"]
        trigger_id = trigger["id"]
        category_slug = merchant.get("category_slug", "")
        category = categories.get(category_slug)

        if category is None:
            logger.error(
                "Pair %02d/%02d: category '%s' not found for merchant %s — skipping",
                i, len(pairs), category_slug, merchant_id,
            )
            errors += 1
            continue

        logger.info(
            "Pair %02d/%02d: merchant=%s trigger=%s category=%s",
            i, len(pairs), merchant_id, trigger_id, category_slug,
        )

        try:
            output = compose(
                category=category,
                merchant=merchant,
                trigger=trigger,
                customer=None,
            )

            score_est = output.get("score_estimate", {})
            score_total = score_est.get("total", 0.0) if isinstance(score_est, dict) else 0.0

            record = {
                "merchant_id": merchant_id,
                "trigger_id": trigger_id,
                "body": output.get("body", ""),
                "cta": output.get("cta", "open_ended"),
                "send_as": output.get("send_as", "vera"),
                "suppression_key": output.get("suppression_key", trigger.get("suppression_key", "")),
                "rationale": output.get("rationale", ""),
                "score_estimate": score_total,
            }
            results.append(record)
            logger.info(
                "  OK — score_estimate=%.1f/50 body_len=%d",
                score_total, len(record["body"]),
            )

        except Exception as exc:
            logger.error(
                "  FAILED for merchant=%s trigger=%s: %s",
                merchant_id, trigger_id, exc,
            )
            logger.debug(traceback.format_exc())
            errors += 1
            # Write a fallback stub so the JSONL always has 30 lines
            results.append({
                "merchant_id": merchant_id,
                "trigger_id": trigger_id,
                "body": f"[ERROR: compose() failed — {type(exc).__name__}: {exc}]",
                "cta": "none",
                "send_as": "vera",
                "suppression_key": trigger.get("suppression_key", ""),
                "rationale": "Composition failed — see logs.",
                "score_estimate": 0.0,
            })

    # Write output
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for record in results:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(
        "=== Done: %d/%d lines written to %s (%d errors) ===",
        len(results), len(pairs), OUTPUT_PATH, errors,
    )

    if errors > 0:
        logger.warning("%d pair(s) failed — review logs above.", errors)
        sys.exit(1)


if __name__ == "__main__":
    main()
