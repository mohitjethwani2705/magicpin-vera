"""
bot.py — Vera bot core composition logic.

Entry point: compose(category, merchant, trigger, customer) -> dict

Architecture:
  compose()
    → should_send()                         — dedup + suppression gate
    → prompt_builder.build_user_prompt()    — context → ranked LLM prompt
    → _call_llm_with_retry()               — Groq API call (temperature=0)
    → _parse_llm_output()                   — JSON extraction + validation
    → estimate_score()                      — self-evaluation across 5 dimensions

Design choices:
  - Model: llama-3.1-8b-instant via Groq REST API.
  - Temperature=0 for determinism as required by the challenge brief.
  - estimate_score() is heuristic-only (no second LLM call). It now accurately
    penalises long messages, rewards binary CTA patterns including "Haan / baad
    mein", checks for source citations and owner-name usage, and starts all
    dimension base scores from realistic priors instead of inflated ones.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path

# Load .env file so GROQ_API_KEY is available when running via uvicorn
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

from typing import Any

import requests

from app.prompt_builder import (
    SYSTEM_PROMPT,
    ACTION_TRIGGER_KINDS,
    ASK_TRIGGER_KINDS,
    build_user_prompt,
    _extract_active_offers,
    _extract_primary_language,
    _extract_customer_language,
    _resolve_digest_item,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("vera.bot")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL: str = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_TIMEOUT: int = int(os.environ.get("GROQ_TIMEOUT", "30"))
# Temperature=0 for determinism per challenge brief requirement.
LLM_TEMPERATURE: float = float(os.environ.get("LLM_TEMPERATURE", "0"))
LLM_MAX_TOKENS: int = int(os.environ.get("LLM_MAX_TOKENS", "512"))

MAX_RETRIES: int = 3
BACKOFF_BASE_SECONDS: float = 1.5

# ---------------------------------------------------------------------------
# LLM call — Groq API (OpenAI-compatible)
# ---------------------------------------------------------------------------


def _call_llm_with_retry(system_prompt: str, user_prompt: str) -> str:
    """
    Call the Groq API with exponential backoff retry.
    Returns the raw string content of the model response.
    Raises RuntimeError after MAX_RETRIES exhausted.
    """
    if not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY environment variable is not set. "
            "Export it before running: export GROQ_API_KEY=gsk_..."
        )

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.debug(
                "LLM call attempt %d/%d (model=%s)", attempt, MAX_RETRIES, GROQ_MODEL
            )
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": LLM_TEMPERATURE,
                    "max_tokens": LLM_MAX_TOKENS,
                },
                timeout=GROQ_TIMEOUT,
            )
            response.raise_for_status()
            content: str = (
                response.json()["choices"][0]["message"]["content"].strip()
            )
            logger.debug("LLM responded (%d chars)", len(content))
            return content

        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                sleep_secs = BACKOFF_BASE_SECONDS ** attempt
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt,
                    MAX_RETRIES,
                    exc,
                    sleep_secs,
                )
                time.sleep(sleep_secs)
            else:
                logger.error(
                    "LLM call failed after %d attempts: %s", MAX_RETRIES, exc
                )

    raise RuntimeError(
        f"Groq API failed after {MAX_RETRIES} attempts: {last_error}"
    )


# ---------------------------------------------------------------------------
# Output parsing + validation
# ---------------------------------------------------------------------------

_REQUIRED_KEYS: frozenset[str] = frozenset(
    ["body", "cta", "send_as", "suppression_key", "rationale"]
)
_VALID_SEND_AS: frozenset[str] = frozenset(["vera", "merchant_on_behalf"])


def _parse_llm_output(raw: str, trigger: dict, customer: dict | None) -> dict:
    """
    Extract and validate the JSON object from the LLM's raw response.

    Handles JSON wrapped in markdown fences and leading/trailing prose.
    Raises ValueError with a clear message on validation failure.
    """
    cleaned = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    else:
        obj_match = re.search(r"\{[\s\S]+\}", cleaned)
        if obj_match:
            cleaned = obj_match.group(0)

    try:
        output: dict = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to extract field values manually using regex as fallback
        try:
            import re as _re
            def _extract_field(text: str, key: str) -> str:
                m = _re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', text, _re.DOTALL)
                if m:
                    return m.group(1)
                # Also try with escaped content
                m2 = _re.search(rf'"{key}"\s*:\s*"(.+?)"(?=\s*[,}}])', text, _re.DOTALL)
                return m2.group(1) if m2 else ""
            output = {
                "body": _extract_field(cleaned, "body"),
                "cta": _extract_field(cleaned, "cta"),
                "send_as": _extract_field(cleaned, "send_as") or "vera",
                "suppression_key": _extract_field(cleaned, "suppression_key"),
                "rationale": _extract_field(cleaned, "rationale"),
            }
            if not output["body"]:
                raise json.JSONDecodeError("no body", cleaned, 0)
        except Exception as exc2:
            raise ValueError(
                f"LLM response is not valid JSON.\nRaw:\n{raw[:500]}\nError: {exc2}"
            ) from exc2

    missing = _REQUIRED_KEYS - set(output.keys())
    if missing:
        raise ValueError(
            f"LLM output missing required keys: {missing}. Got: {list(output.keys())}"
        )

    if output.get("send_as") not in _VALID_SEND_AS:
        expected = (
            "merchant_on_behalf"
            if trigger.get("scope") == "customer" and customer
            else "vera"
        )
        logger.warning(
            "Invalid send_as '%s' — correcting to '%s'",
            output.get("send_as"),
            expected,
        )
        output["send_as"] = expected

    expected_key = trigger.get("suppression_key", "")
    if expected_key and output.get("suppression_key") != expected_key:
        logger.warning(
            "LLM changed suppression_key '%s' → '%s' — reverting",
            expected_key,
            output.get("suppression_key"),
        )
        output["suppression_key"] = expected_key

    output["body"] = _strip_markdown(output.get("body", ""))

    if not output["body"].strip():
        raise ValueError("LLM returned an empty message body.")

    return output


def _strip_markdown(text: str) -> str:
    """Remove markdown artifacts that must not appear in WhatsApp messages."""
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Suppression gate
# ---------------------------------------------------------------------------


def should_send(
    merchant: dict,
    trigger: dict,
    suppression_log: set,
) -> bool:
    """
    Decide whether to send a message now or suppress it.

    Returns False (suppress) when:
    1. Trigger's suppression_key is already in suppression_log.
    2. Merchant subscription is expired and trigger is not winback_eligible.
    3. Customer-scope trigger has no customer_id in payload.
    4. Trigger has passed its expires_at timestamp.
    """
    suppression_key = trigger.get("suppression_key", "")
    trigger_kind = trigger.get("kind", "")
    trigger_scope = trigger.get("scope", "merchant")

    if suppression_key and suppression_key in suppression_log:
        logger.info(
            "SUPPRESSED [dedup] key=%s merchant=%s",
            suppression_key,
            merchant.get("merchant_id", ""),
        )
        return False

    sub = merchant.get("subscription", {})
    if sub.get("status") == "expired" and trigger_kind != "winback_eligible":
        logger.info(
            "SUPPRESSED [expired_subscription] merchant=%s trigger=%s",
            merchant.get("merchant_id", ""),
            trigger_kind,
        )
        return False

    if trigger_scope == "customer":
        customer_id = trigger.get("customer_id") or trigger.get("payload", {}).get("customer_id")
        if not customer_id:
            logger.info(
                "SUPPRESSED [no_customer_id] trigger=%s", trigger.get("id", "")
            )
            return False

    expires_at_str = trigger.get("expires_at")
    if expires_at_str:
        try:
            from datetime import datetime, timezone
            expires_dt = datetime.fromisoformat(
                expires_at_str.replace("Z", "+00:00")
            )
            if datetime.now(timezone.utc) > expires_dt:
                logger.info(
                    "SUPPRESSED [trigger_expired] key=%s expired=%s",
                    suppression_key,
                    expires_at_str,
                )
                return False
        except (ValueError, TypeError) as exc:
            logger.warning("Could not parse expires_at '%s': %s", expires_at_str, exc)

    logger.info(
        "SEND_APPROVED merchant=%s trigger=%s key=%s",
        merchant.get("merchant_id", ""),
        trigger_kind,
        suppression_key,
    )
    return True


# ---------------------------------------------------------------------------
# Self-evaluation heuristic
# ---------------------------------------------------------------------------


def estimate_score(
    output: dict,
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: dict | None = None,
) -> dict:
    """
    Heuristic self-evaluation across the 5 judge dimensions (0-10 each).

    Calibrated against the 10 case studies. The heuristic catches the most
    common failure modes. The real judge does deeper semantic evaluation.

    Returns:
      {
        "specificity": float,
        "category_fit": float,
        "merchant_fit": float,
        "trigger_relevance": float,
        "engagement_compulsion": float,
        "total": float,
        "flags": [list of detected issues]
      }
    """
    body: str = output.get("body", "")
    body_lower = body.lower()
    word_count = len(body.split())
    flags: list[str] = []
    scores: dict[str, float] = {}

    identity = merchant.get("identity", {})
    trigger_kind = trigger.get("kind", "unknown")
    payload = trigger.get("payload", {})
    digest_item = _resolve_digest_item(trigger, category)

    # -------------------------------------------------------------------------
    # SPECIFICITY (0-10) — base 4.0
    # Rewards: numbers, source citations, exact payload values
    # Penalties: generic language, word count excess
    # -------------------------------------------------------------------------
    specificity = 4.0

    # Reward: numbers present in body (₹, %, year, patient count)
    numbers_found = re.findall(
        r"₹[\d,]+|\d+[\d,]*(?:\.\d+)?%|\d{4}|\bn=\d+|\d{2,}[-\s](?:patient|member|customer)",
        body,
    )
    specificity += min(len(numbers_found) * 1.2, 3.5)

    # Reward: source citation (e.g. "JIDA Oct 2026 p.14", "DCI circular", batch numbers)
    source_citation_patterns = [
        r"JIDA",
        r"DCI",
        r"IDA\b",
        r"Dental Tribune",
        r"p\.\d+",         # page reference
        r"AT\d{4}-\d{4}",  # batch number pattern
        r"circular",
        r"\bvol\b",
    ]
    for pat in source_citation_patterns:
        if re.search(pat, body, re.IGNORECASE):
            specificity += 1.5
            break

    # Reward: exact digest item values present in body
    if digest_item:
        trial_n = digest_item.get("trial_n")
        if trial_n and (str(trial_n) in body or f"{trial_n:,}" in body):
            specificity += 0.5
        # Check if a percentage from the title appears in the body
        title = digest_item.get("title", "")
        pct_match = re.search(r"(\d+)%", title)
        if pct_match and pct_match.group(0) in body:
            specificity += 0.5

    # Penalize: generic discount / growth language
    generic_patterns = [
        r"\bflat\s+\d+%\s+off\b",
        r"\bspecial\s+discount\b",
        r"\bamazing\s+deal\b",
        r"\bincrease\s+your\s+sales\b",
        r"\bgrow\s+your\s+business\b",
        r"\bboost\s+your\s+(?:sales|revenue|business)\b",
        r"\ba\s+recent\s+study\b",   # un-cited study reference
    ]
    for pat in generic_patterns:
        if re.search(pat, body_lower):
            specificity -= 2.0
            flags.append(f"GENERIC_LANGUAGE: '{pat}'")

    # Penalize: word count > 80 (hard rule from system prompt)
    if word_count > 100:
        specificity -= 3.0
        flags.append(f"TOO_LONG: {word_count} words (limit 80)")
    elif word_count > 80:
        specificity -= 1.5
        flags.append(f"SLIGHTLY_LONG: {word_count} words (limit 80)")

    scores["specificity"] = max(0.0, min(10.0, specificity))

    # -------------------------------------------------------------------------
    # CATEGORY FIT (0-10) — base 6.0
    # Rewards: using allowed vocabulary
    # Penalties: taboo words, hype in clinical categories
    # -------------------------------------------------------------------------
    category_fit = 6.0
    voice = category.get("voice", {})
    if isinstance(voice, str):
        voice = {"tone": voice}
    vocab_allowed = [v.lower() for v in voice.get("vocab_allowed", [])]
    vocab_taboo = [v.lower() for v in voice.get("vocab_taboo", [])]

    # Reward: category vocabulary used (up to +2.5)
    matches = sum(1 for w in vocab_allowed if w in body_lower)
    category_fit += min(matches * 0.5, 2.5)

    # Hard penalty: taboo words
    for word in vocab_taboo:
        if word in body_lower:
            category_fit -= 3.0
            flags.append(f"TABOO_WORD: '{word}'")

    # Penalty: promotional hype in clinical categories
    clinical_slugs = {"dentists", "pharmacies", "gyms", "doctors"}
    if category.get("slug") in clinical_slugs:
        hype_pats = [
            r"\bamazing\b", r"\bincredible\b", r"\bbest in city\b",
            r"\bguaranteed\b", r"\bno\.?\s*1\b",
        ]
        for pat in hype_pats:
            if re.search(pat, body_lower):
                category_fit -= 2.0
                flags.append(f"HYPE_IN_CLINICAL: '{pat}'")

    scores["category_fit"] = max(0.0, min(10.0, category_fit))

    # -------------------------------------------------------------------------
    # MERCHANT FIT (0-10) — base 4.0
    # Rewards: owner name, locality, active offers, performance numbers, language match
    # -------------------------------------------------------------------------
    merchant_fit = 4.0

    # Reward: owner first name in body (+2.5 — big signal per case studies)
    owner_first = identity.get("owner_first_name", "")
    merchant_name_full = identity.get("name", "")
    if owner_first and owner_first.lower() in body_lower:
        merchant_fit += 2.5
    elif merchant_name_full and merchant_name_full.lower() in body_lower:
        merchant_fit += 1.0
    else:
        flags.append("NO_OWNER_NAME: owner first name not found in body — loses merchant fit points")

    # Reward: locality or city referenced
    merchant_locality = identity.get("locality", "").lower()
    merchant_city = identity.get("city", "").lower()
    if merchant_locality and merchant_locality in body_lower:
        merchant_fit += 1.0
    elif merchant_city and merchant_city in body_lower:
        merchant_fit += 0.5

    # Reward: references an active offer from merchant's catalog
    active_offers = _extract_active_offers(merchant)
    for offer in active_offers:
        offer_words = offer.lower().split()
        meaningful_words = [w for w in offer_words if len(w) > 3]
        if any(w in body_lower for w in meaningful_words):
            merchant_fit += 1.5
            break

    # Reward: references merchant performance numbers exactly
    perf = merchant.get("performance", {})
    for metric_val in [perf.get("views"), perf.get("calls"), perf.get("ctr")]:
        if metric_val and str(metric_val) in body:
            merchant_fit += 0.5

    # Reward: references customer aggregate counts
    cust_agg = merchant.get("customer_aggregate", {})
    for count_val in [
        cust_agg.get("active_count"),
        cust_agg.get("lapsed_count"),
        cust_agg.get("high_risk_count"),
    ]:
        if count_val and str(count_val) in body:
            merchant_fit += 0.5
            break

    # Language match check — critical for hi-en merchants
    lang_code = _extract_primary_language(merchant)
    if lang_code == "hi-en":
        hindi_words = [
            "aap", "main", "kya", "hai", "hain", "karo", "chalega", "taaki",
            "apke", "apka", "se", "ka", "ki", "ke", "bhi", "mein", "yeh",
            "ek", "nahi", "sab", "kuch", "liye", "le", "woh", "toh", "haan",
            "baad", "mein", "ke liye", "kar", "raha", "rahi", "dono",
        ]
        body_words_lower = set(body_lower.split())
        hindi_found = any(w in body_words_lower for w in hindi_words)
        # Also check for common Hindi phrases as substrings
        hindi_phrases = ["baad mein", "haan", "ke liye", "apke", "chalega"]
        phrase_found = any(ph in body_lower for ph in hindi_phrases)
        if not hindi_found and not phrase_found:
            merchant_fit -= 2.0
            flags.append(
                "LANGUAGE_MISMATCH: merchant is hi-en but body has no Hindi words"
            )

    scores["merchant_fit"] = max(0.0, min(10.0, merchant_fit))

    # -------------------------------------------------------------------------
    # TRIGGER RELEVANCE (0-10) — base 4.5
    # Rewards: trigger-kind keywords, exact payload values
    # -------------------------------------------------------------------------
    trigger_relevance = 4.5

    kind_keyword_map: dict[str, list[str]] = {
        "research_digest": [
            "jida", "trial", "study", "research", "recall", "fluoride",
            "caries", "p.14", "2026",
        ],
        "regulation_change": [
            "dci", "circular", "limit", "compliance", "deadline", "audit",
        ],
        "recall_due": ["recall", "cleaning", "slot", "appointment", "due", "months"],
        "chronic_refill_due": [
            "refill", "medicine", "metformin", "atorvastatin", "telmisartan",
            "stock", "monthly",
        ],
        "perf_dip": ["dip", "drop", "down", "calls", "views", "declined", "week"],
        "perf_spike": ["spike", "up", "increased", "views", "calls"],
        "seasonal_perf_dip": [
            "seasonal", "dip", "normal", "lull", "retention", "save", "recovery",
        ],
        "ipl_match_today": [
            "ipl", "match", "dc", "mi", "cricket", "delivery", "covers", "saturday",
        ],
        "competitor_opened": [
            "competitor", "nearby", "differentiat", "offer", "opened", "km",
        ],
        "review_theme_emerged": [
            "review", "wait", "delivery", "customer", "feedback", "theme",
        ],
        "renewal_due": ["renewal", "renew", "subscription", "days", "plan"],
        "winback_eligible": ["back", "missed", "subscription", "profile", "return"],
        "festival_upcoming": ["diwali", "festival", "festive", "eid", "holi", "season"],
        "curious_ask_due": [
            "this week", "most", "asked", "service", "demand", "popular",
        ],
        "active_planning_intent": [
            "draft", "ready", "package", "version", "option", "here", "details",
        ],
        "dormant_with_vera": ["profile", "update", "check", "notice"],
        "supply_alert": [
            "recall", "batch", "atorvastatin", "molecule", "replacement",
            "AT2024", "sub-potency",
        ],
        "customer_lapsed_hard": [
            "weeks", "back", "return", "trial", "visit", "months",
        ],
        "customer_lapsed_soft": ["months", "visit", "reminder", "due"],
        "wedding_package_followup": [
            "wedding", "bridal", "skin", "session", "program", "days",
        ],
        "bridal_followup": ["wedding", "bridal", "skin", "session", "days", "prep"],
        "trial_followup": ["trial", "session", "paid", "next", "convert"],
        "gbp_unverified": ["verified", "gbp", "views", "profile", "uplift"],
        "category_trend_movement": [
            "trend", "search", "aligner", "demand", "popular", "yoy",
        ],
        "milestone_reached": ["milestone", "reviews", "crossed", "100", "approaching"],
    }

    keywords = kind_keyword_map.get(trigger_kind, [])
    matched = sum(1 for kw in keywords if kw in body_lower)
    trigger_relevance += min(matched * 1.2, 4.0)

    # Reward: specific payload values in the body
    _trigger_value_checks = [
        ("molecule", lambda v: v.lower() in body_lower),
        ("festival", lambda v: v.lower() in body_lower),
        ("affected_batches", lambda v: any(b.lower() in body_lower for b in v) if isinstance(v, list) else False),
        ("match", lambda v: v.lower().split()[0] in body_lower if v else False),
        ("competitor_name", lambda v: v.lower().split()[0] in body_lower if v else False),
    ]
    for field, checker in _trigger_value_checks:
        val = payload.get(field)
        if val:
            try:
                if checker(val):
                    trigger_relevance += 0.5
            except Exception:
                pass

    scores["trigger_relevance"] = max(0.0, min(10.0, trigger_relevance))

    # -------------------------------------------------------------------------
    # ENGAGEMENT COMPULSION (0-10) — base 3.5
    # Rewards: binary CTA (including "Haan / baad mein"), effort externalization,
    #          loss aversion, social proof
    # Penalties: multiple CTAs, preamble, URL, long message
    # -------------------------------------------------------------------------
    engagement = 3.5

    # Reward: binary CTA present — includes Hinglish pattern
    binary_cta_patterns = [
        r"haan\s*/\s*baad\s*mein",   # primary hi-en pattern
        r"yes\s*/\s*not\s*now",       # primary en pattern
        r"\breply\s+yes\b",
        r"\breply\s+confirm\b",
        r"\bsay\s+go\b",
        r"\bchalega\b",
        r"\bwant\s+me\s+to\b",
        r"\bshall\s+i\b",
        r"\bI\s+can\b",
    ]
    binary_found = False
    for pat in binary_cta_patterns:
        if re.search(pat, body_lower):
            binary_found = True
            # Extra reward for the ideal Hinglish CTA
            if re.search(r"haan\s*/\s*baad\s*mein", body_lower):
                engagement += 2.5
            else:
                engagement += 2.0
            break

    if not binary_found:
        flags.append("NO_BINARY_CTA: body lacks a binary YES/NO or 'Haan / baad mein' CTA")

    # Reward: CTA is at the end (last 30% of body)
    if binary_found:
        last_third_start = max(0, len(body) - len(body) // 3)
        if any(re.search(pat, body_lower[last_third_start:]) for pat in binary_cta_patterns):
            engagement += 0.5  # CTA is correctly placed last

    # Reward: effort externalization language
    effort_patterns = [
        r"want me to",
        r"shall i",
        r"i.ve drafted",
        r"i.ll (?:pull|draft|set|send|create)",
        r"live in \d+ min",
        r"just say go",
        r"no commitment",
        r"ready for you",
        r"i can (?:draft|pull|set|send|create)",
    ]
    for pat in effort_patterns:
        if re.search(pat, body_lower):
            engagement += 1.0
            break

    # Reward: loss aversion language
    loss_patterns = [
        r"missing",
        r"below\s+peer",
        r"before.*closes",
        r"expir",
        r"\d+%\s+below",
        r"not\s+converting",
    ]
    for pat in loss_patterns:
        if re.search(pat, body_lower):
            engagement += 0.8
            break

    # Reward: social proof
    social_proof_patterns = [
        r"\d+\s+(?:dentists?|salons?|gyms?|pharmacies?|practices?)\s+in",
        r"peer\s+median",
        r"other\s+(?:merchants?|practices?)",
        r"most\s+practices",
        r"3\s+(?:clinics?|stores?|gyms?|salons?)\s+in",
    ]
    for pat in social_proof_patterns:
        if re.search(pat, body_lower):
            engagement += 1.0
            break

    # Penalize: multiple CTAs
    cta_count = len(re.findall(r"\breply\s+(?:yes|no|stop|1|2|3|confirm)\b", body_lower))
    if cta_count > 2:
        engagement -= 2.0
        flags.append(f"MULTIPLE_CTAS: {cta_count} CTA patterns found")

    # Penalize: URL in body (challenge brief penalises this)
    if re.search(r"https?://|www\.", body):
        engagement -= 1.0
        flags.append("URL_IN_BODY: URLs penalised per brief rules")

    # Penalize: preamble openers
    preamble_patterns = [
        r"^i hope",
        r"^just checking",
        r"^i.m reaching out",
        r"^hope you.re",
        r"^as vera",
        r"^hi,?\s*i.m vera",
    ]
    for pat in preamble_patterns:
        if re.search(pat, body_lower):
            engagement -= 2.0
            flags.append(f"PREAMBLE: '{pat}'")
            break

    # Penalize: excessive length hurts engagement (people stop reading)
    if word_count > 100:
        engagement -= 2.0
    elif word_count > 80:
        engagement -= 1.0

    scores["engagement_compulsion"] = max(0.0, min(10.0, engagement))

    total = sum(scores.values())

    return {
        "specificity": round(scores["specificity"], 1),
        "category_fit": round(scores["category_fit"], 1),
        "merchant_fit": round(scores["merchant_fit"], 1),
        "trigger_relevance": round(scores["trigger_relevance"], 1),
        "engagement_compulsion": round(scores["engagement_compulsion"], 1),
        "total": round(total, 1),
        "word_count": word_count,
        "flags": flags,
    }


# ---------------------------------------------------------------------------
# Primary composition function
# ---------------------------------------------------------------------------


def compose(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: dict | None = None,
) -> dict:
    """
    Compose a WhatsApp message from the four context layers.

    Args:
        category: CategoryContext dict (from categories/*.json)
        merchant: MerchantContext dict (from merchants/*.json)
        trigger:  TriggerContext dict  (from triggers/*.json)
        customer: CustomerContext dict (from customers/*.json) — optional

    Returns a dict with keys:
        body             — the WhatsApp message body (plain text, 40-80 words)
        cta              — the single call-to-action string, or "none"
        send_as          — "vera" | "merchant_on_behalf"
        suppression_key  — from the trigger, unchanged
        rationale        — why this message, which levers used
        score_estimate   — dict from estimate_score()

    Raises:
        EnvironmentError  — GROQ_API_KEY not set
        RuntimeError      — all LLM retries exhausted
        ValueError        — LLM output could not be parsed or validated
    """
    merchant_id = merchant.get("merchant_id", "unknown")
    trigger_id = trigger.get("id", "unknown")
    trigger_kind = trigger.get("kind", "unknown")

    logger.info(
        "compose() START merchant=%s trigger=%s kind=%s",
        merchant_id,
        trigger_id,
        trigger_kind,
    )

    try:
        user_prompt = build_user_prompt(
            category=category,
            merchant=merchant,
            trigger=trigger,
            customer=customer,
        )
    except Exception as exc:
        logger.error("prompt_builder failed: %s", exc, exc_info=True)
        raise

    logger.debug("User prompt (%d chars):\n%s", len(user_prompt), user_prompt[:800])

    try:
        raw_response = _call_llm_with_retry(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
    except Exception as llm_exc:
        logger.error("compose() LLM failed (rate limit or error): %s — using fallback", llm_exc)
        # Return a canned fallback message so tick doesn't crash
        identity = merchant.get("identity", {})
        owner = identity.get("owner_first_name") or merchant.get("owner_name", "")
        first = owner.split()[0] if owner else "ji"
        lang = category.get("language_preference", "en")
        body = (
            f"{first}, aapke performance mein improvement ki zarurat hai. "
            f"Main aapke liye ek special offer plan kar sakta hoon. Haan / baad mein?"
            if "hi" in lang else
            f"{first}, I noticed your business needs a boost. "
            f"I can help you with a special offer. Yes / not now"
        )
        return {
            "body": body,
            "cta": "Haan / baad mein" if "hi" in lang else "Yes / not now",
            "send_as": "vera",
            "suppression_key": f"fallback:{merchant.get('merchant_id','unknown')}:{trigger.get('id','unknown')}",
            "rationale": "LLM unavailable (rate limited) — canned fallback used.",
            "score_estimate": 0,
        }

    logger.debug("Raw LLM response:\n%s", raw_response[:600])

    output = _parse_llm_output(raw_response, trigger, customer)

    score = estimate_score(output, category, merchant, trigger, customer)
    output["score_estimate"] = score

    logger.info(
        "compose() DONE merchant=%s trigger=%s score=%.1f/50 words=%d flags=%s",
        merchant_id,
        trigger_id,
        score["total"],
        score.get("word_count", 0),
        score["flags"] if score["flags"] else "none",
    )

    return output


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pathlib

    DATA_DIR = pathlib.Path(__file__).parent / "dataset"

    with open(DATA_DIR / "categories" / "dentists.json") as f:
        _category = json.load(f)

    with open(DATA_DIR / "merchants_seed.json") as f:
        _merchants_seed = json.load(f)

    with open(DATA_DIR / "triggers_seed.json") as f:
        _triggers_seed = json.load(f)

    with open(DATA_DIR / "customers_seed.json") as f:
        _customers_seed = json.load(f)

    _merchant = next(
        m for m in _merchants_seed["merchants"]
        if m["merchant_id"] == "m_001_drmeera_dentist_delhi"
    )
    _trigger = next(
        t for t in _triggers_seed["triggers"]
        if t["id"] == "trg_001_research_digest_dentists"
    )

    print("\n=== SMOKE TEST: Dr. Meera + research_digest ===\n")

    _suppression_log: set = set()

    if not should_send(_merchant, _trigger, _suppression_log):
        print("should_send() returned False — message suppressed.")
    else:
        result = compose(_category, _merchant, _trigger)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n--- Smoke test 2: Priya recall reminder (customer-facing) ---\n")

    _trigger_recall = next(
        t for t in _triggers_seed["triggers"]
        if t["id"] == "trg_003_recall_due_priya"
    )
    _customer_priya = next(
        c for c in _customers_seed["customers"]
        if c["customer_id"] == "c_001_priya_for_m001"
    )

    if not should_send(_merchant, _trigger_recall, _suppression_log):
        print("should_send() returned False — message suppressed.")
    else:
        result2 = compose(_category, _merchant, _trigger_recall, _customer_priya)
        print(json.dumps(result2, indent=2, ensure_ascii=False))
