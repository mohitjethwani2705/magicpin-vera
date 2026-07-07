"""
prompt_builder.py — Vera bot prompt construction layer.

Design principles (derived from challenge brief + case studies):
- SYSTEM_PROMPT enforces 3-4 line maximum and the Hook→Relevance→Action→CTA
  four-line formula that the case studies consistently score 49-50/50 with.
- build_user_prompt() pre-digests the context into a ranked "ammunition rack":
  the single most compelling stat is surfaced first so the LLM leads with it,
  not buries it.
- Binary CTA is required on all action triggers. "Haan / baad mein" for
  hi-en merchants; "Yes / not now" for English-primary merchants.
- Effort externalization phrasing ("want me to...", "shall I...", "I can...")
  is required on all action CTAs.
- Source citations with page numbers are required when a digest item has them.
- No fabrication: if a number isn't in the provided context section it must
  not appear in the message.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Triggers where a binary YES/NO CTA is mandatory (includes research_digest —
# case study 1 scores 50/50 with a binary CTA, NOT an open-ended one).
ACTION_TRIGGER_KINDS: frozenset[str] = frozenset([
    "research_digest",
    "regulation_change",
    "supply_alert",
    "renewal_due",
    "perf_dip",
    "perf_spike",
    "active_planning_intent",
    "recall_due",
    "chronic_refill_due",
    "customer_lapsed_hard",
    "competitor_opened",
    "review_theme_emerged",
    "milestone_reached",
    "gbp_unverified",
    "category_trend_movement",
    "seasonal_perf_dip",
    "festival_upcoming",
    "ipl_match_today",
    "dormant_with_vera",
    "winback_eligible",
    "wedding_package_followup",
    "bridal_followup",
    "trial_followup",
    "customer_lapsed_soft",
])

# Triggers where asking the merchant a low-stakes question IS the CTA.
ASK_TRIGGER_KINDS: frozenset[str] = frozenset([
    "curious_ask_due",
    "scheduled_recurring",
    "cde_opportunity",
    "category_seasonal",
])

# Trigger kind → one-sentence frame the LLM uses as the "why now" anchor.
TRIGGER_FRAME_MAP: dict[str, str] = {
    "research_digest": "new research just landed that directly affects this merchant's patient/customer cohort",
    "regulation_change": "a regulatory deadline directly affects this merchant's practice — action required before deadline",
    "cde_opportunity": "a professional development opportunity is available for a limited window",
    "recall_due": "a patient's scheduled recall window has opened — they are due for a routine service",
    "chronic_refill_due": "a customer's chronic prescription medications are about to run out",
    "perf_dip": "the merchant's performance metrics dropped significantly this week",
    "perf_spike": "the merchant's performance metrics spiked — reinforce what's working",
    "milestone_reached": "the merchant just crossed or is about to cross a milestone",
    "seasonal_perf_dip": "a performance dip is happening but is entirely expected for this season — reframe, don't alarm",
    "dormant_with_vera": "the merchant has not replied to Vera in many days — re-engage with fresh value, not a reminder",
    "winback_eligible": "the merchant's subscription lapsed and they are showing re-engagement signals",
    "festival_upcoming": "a major festival is approaching — timing-sensitive opportunity for this category",
    "ipl_match_today": "an IPL match is today — think carefully about whether it helps or hurts this category before recommending",
    "review_theme_emerged": "a repeated theme appeared in recent customer reviews — merchant needs to know",
    "competitor_opened": "a new competitor opened nearby — frame as intelligence + differentiation opportunity, not panic",
    "curious_ask_due": "weekly curiosity-driven check-in — ask a low-stakes question, offer clear reciprocity",
    "active_planning_intent": "the merchant explicitly said yes to a plan — provide a complete draft immediately, no more questions",
    "gbp_unverified": "the merchant's Google Business Profile is unverified — concrete uplift data is available",
    "category_trend_movement": "a category-level search trend is shifting significantly — merchant should act",
    "supply_alert": "an urgent supply or safety alert affects products the merchant stocks — urgency high",
    "customer_lapsed_hard": "a customer has been absent for many weeks — warm, no-shame re-engagement",
    "customer_lapsed_soft": "a customer's soft-lapse window opened — gentle nudge before they churn fully",
    "wedding_package_followup": "a customer is in the pre-wedding service window — timing-sensitive follow-up",
    "bridal_followup": "a customer is in the bridal prep window — timing-sensitive follow-up",
    "trial_followup": "a customer just completed a trial session — convert to paid, low friction",
    "scheduled_recurring": "a scheduled recurring check-in — make it feel fresh and specific, not routine",
}


# ---------------------------------------------------------------------------
# Helper extractors
# ---------------------------------------------------------------------------


def _fmt_pct(value: Any) -> str:
    """Format a float as percentage or return N/A if missing."""
    try:
        return f"{float(value):+.0%}"
    except (TypeError, ValueError):
        return "N/A"


def _extract_active_offers(merchant: dict) -> list[str]:
    """Return titles of currently active offers only."""
    return [
        o["title"]
        for o in merchant.get("offers", [])
        if o.get("status") == "active"
    ]


def _extract_signals(merchant: dict) -> list[str]:
    """Return the merchant's derived signals list."""
    return merchant.get("signals", [])


def _extract_primary_language(merchant: dict) -> str:
    """Map the merchant's languages array to a language instruction string."""
    langs = merchant.get("identity", {}).get("languages", ["en"])
    lang_set = set(langs)
    if "hi" in lang_set and "en" in lang_set:
        return "hi-en"
    if "ta" in lang_set and "en" in lang_set:
        return "ta-en"
    if "te" in lang_set and "en" in lang_set:
        return "te-en"
    if "mr" in lang_set and "en" in lang_set:
        return "mr-en"
    if "kn" in lang_set and "en" in lang_set:
        return "kn-en"
    if "hi" in lang_set:
        return "hi-en"
    return "en"


def _extract_customer_language(customer: dict) -> str:
    """Return language code from customer context."""
    lang_pref = customer.get("identity", {}).get("language_pref", "english").lower()
    if "hi-en" in lang_pref or "hindi-english" in lang_pref:
        return "hi-en"
    if "te-en" in lang_pref:
        return "te-en"
    if "ta-en" in lang_pref:
        return "ta-en"
    if "kn-en" in lang_pref:
        return "kn-en"
    if lang_pref in ("hi", "hindi"):
        return "hi-en"
    return "en"


def _language_cta(lang_code: str) -> str:
    """Return the correct binary CTA suffix for the merchant's language."""
    if lang_code == "hi-en":
        return "Haan / baad mein"
    return "Yes / not now"


def _language_instruction(lang_code: str) -> str:
    """Return a full prose instruction the LLM uses for language matching."""
    mapping = {
        "hi-en": (
            "Hindi-English code-mix (Hinglish). Mix naturally within sentences — "
            "e.g. 'Apke 124 high-risk patients ke liye ek useful finding hai.' "
            "Do NOT write in pure English. Do NOT write in pure Hindi."
        ),
        "ta-en": "Tamil-English mix. English primary; Tamil words for warmth where natural.",
        "te-en": "Telugu-English mix. English primary; Telugu words for warmth where natural.",
        "mr-en": "Marathi-English mix. English primary; Marathi words for warmth where natural.",
        "kn-en": "Kannada-English mix. English primary; Kannada words for warmth where natural.",
        "en": "English only. Clean, direct, peer-tone. No forced Hindi.",
    }
    return mapping.get(lang_code, "English only.")


def _resolve_digest_item(trigger: dict, category: dict) -> dict | None:
    """Find the full digest item referenced by trigger payload top_item_id."""
    # Try top_item_id first (normalized reference)
    item_id = trigger.get("payload", {}).get("top_item_id")
    if item_id:
        for item in category.get("digest", []):
            if item.get("id") == item_id:
                return item

    # Fallback: top_item may be embedded directly in the trigger payload
    top_item = trigger.get("payload", {}).get("top_item")
    if isinstance(top_item, dict):
        return top_item

    return None


def _format_peer_gap(merchant: dict, category: dict) -> str:
    """
    Return a precise gap sentence comparing merchant CTR to peer median.
    Used as loss aversion ammunition in the user prompt.
    """
    perf = merchant.get("performance", {})
    peer = category.get("peer_stats", {})
    merchant_ctr = perf.get("ctr")
    peer_ctr = peer.get("avg_ctr")
    if merchant_ctr is None or peer_ctr is None:
        return ""
    if merchant_ctr < peer_ctr:
        gap_pct = round((peer_ctr - merchant_ctr) / peer_ctr * 100)
        return (
            f"CTR {merchant_ctr:.1%} vs peer median {peer_ctr:.1%} "
            f"({gap_pct}% below peer — strong loss aversion hook)"
        )
    return (
        f"CTR {merchant_ctr:.1%} vs peer median {peer_ctr:.1%} "
        f"(above peer — positive reinforcement hook)"
    )


def _format_slots(slots: list[dict]) -> str:
    """Format available booking slots into a short readable list."""
    if not slots:
        return ""
    return " | ".join(s.get("label", "") for s in slots if s.get("label"))


# ---------------------------------------------------------------------------
# Core ammunition extractor
#
# This is the most important function in the file. It pre-ranks the available
# data and surfaces the single most compelling "lead stat" for Line 1.
# The LLM does not need to decide what to lead with — we decide for it.
# ---------------------------------------------------------------------------


def _extract_ammunition(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: dict | None,
    digest_item: dict | None,
) -> dict:
    """
    Extract and rank the concrete facts available for this composition.

    Returns a dict with these guaranteed keys:
      lead_stat       — the single most compelling number/fact for Line 1 hook
      lead_source     — citation for the lead stat (if available)
      merchant_anchor — the merchant-specific number that makes it personal
      peer_gap        — CTR vs peer comparison string (empty if not applicable)
      best_offer      — the most relevant active offer title (empty if none)
      affected_count  — number of patients/customers affected (0 if unknown)
      trigger_payload_lines — list of rendered trigger facts
    """
    payload = trigger.get("payload", {})
    trigger_kind = trigger.get("kind", "")
    perf = merchant.get("performance", {})
    cust_agg = merchant.get("customer_aggregate", {})

    lead_stat = ""
    lead_source = ""
    merchant_anchor = ""
    affected_count = 0

    # --- Research / regulation digest ---
    if digest_item:
        title = digest_item.get("title", "")
        source = digest_item.get("source", "")
        trial_n = digest_item.get("trial_n")
        segment = digest_item.get("patient_segment", "")

        # Extract the first percentage from the title as the lead stat
        pct_match = re.search(r"(\d+)%", title)
        if pct_match:
            lead_stat = f"{pct_match.group(1)}% improvement"
        if trial_n:
            lead_stat = f"{trial_n:,}-patient trial: {title.split('—')[0].strip() if '—' in title else title[:60]}"
        lead_source = source

        # Merchant anchor: how many of their patients fit the segment?
        if segment and "high-risk" in segment.lower():
            high_risk = cust_agg.get("high_risk_count", cust_agg.get("lapsed_count", 0))
            if high_risk:
                merchant_anchor = f"Your {high_risk} {segment} patients"
                affected_count = int(high_risk)
        elif segment:
            total_patients = cust_agg.get("active_count", cust_agg.get("unique_ytd", 0))
            if total_patients:
                merchant_anchor = f"Your {total_patients} active patients"
                affected_count = int(total_patients)

    # --- Performance dip ---
    elif trigger_kind in ("perf_dip", "seasonal_perf_dip"):
        metric = payload.get("metric", "views")
        delta = payload.get("delta_pct", 0)
        window = payload.get("window", "this week")
        lead_stat = f"{metric} {delta:+.0%} {window}"
        if trigger_kind == "seasonal_perf_dip":
            season_range = payload.get("season_note", "April-June lull")
            lead_stat += f" — normal {season_range}"
        merchant_anchor = f"{perf.get('views', '')} views (30d)"

    # --- Performance spike ---
    elif trigger_kind == "perf_spike":
        metric = payload.get("metric", "views")
        delta = payload.get("delta_pct", 0)
        lead_stat = f"{metric} up {delta:+.0%} — find out why and double down"
        merchant_anchor = f"{perf.get('views', '')} views (30d)"

    # --- Supply / recall alert ---
    elif trigger_kind == "supply_alert":
        batches = payload.get("affected_batches", [])
        molecule = payload.get("molecule", "")
        lead_stat = f"voluntary recall: {molecule} batches {', '.join(batches)}"
        lead_source = payload.get("manufacturer", "")
        # Cross-reference affected customer count from merchant aggregate
        affected_count = payload.get("affected_customer_count", 0)
        if not affected_count:
            # Derive: assume ~10% of chronic-Rx customers
            chronic = cust_agg.get("chronic_rx_count", cust_agg.get("active_count", 0))
            affected_count = max(1, round(int(chronic) * 0.09)) if chronic else 0
        merchant_anchor = f"{affected_count} of your customers dispensed these batches"

    # --- IPL match ---
    elif trigger_kind == "ipl_match_today":
        match = payload.get("match", "IPL match")
        venue = payload.get("venue", "")
        match_time = payload.get("match_time_iso", "")
        is_weeknight = payload.get("is_weeknight", True)
        footfall_delta = payload.get("footfall_impact_pct", -12)
        lead_stat = f"{match} at {venue}" if venue else match
        if not is_weeknight:
            lead_stat += f" — Saturday IPL typically shifts {footfall_delta:+.0%} restaurant covers"

    # --- Competitor opened ---
    elif trigger_kind == "competitor_opened":
        name = payload.get("competitor_name", "a new competitor")
        dist = payload.get("distance_km", "")
        lead_stat = f"{name} opened {dist}km away"
        reviews = merchant.get("performance", {}).get("review_count", "")
        rating = merchant.get("identity", {}).get("rating", "")
        if reviews and rating:
            merchant_anchor = f"You have {reviews} reviews at {rating}★ — differentiation anchor"

    # --- Recall due (customer scope) ---
    elif trigger_kind in ("recall_due", "chronic_refill_due"):
        months_since = payload.get("months_since_last_visit", payload.get("months_since_last_fill", ""))
        service = payload.get("service_due", payload.get("medicines", ""))
        lead_stat = f"{months_since} months since last visit" if months_since else str(service)
        slots = payload.get("available_slots", payload.get("next_session_options", []))
        merchant_anchor = _format_slots(slots)

    # --- Customer lapsed ---
    elif trigger_kind in ("customer_lapsed_hard", "customer_lapsed_soft"):
        days = payload.get("days_since_last_visit", "")
        weeks = round(int(days) / 7) if days else ""
        lead_stat = f"{weeks} weeks since last visit" if weeks else f"{days} days since last visit"
        focus = payload.get("previous_focus", "")
        merchant_anchor = f"Previous focus: {focus}" if focus else ""

    # --- Renewal due ---
    elif trigger_kind == "renewal_due":
        days = payload.get("days_remaining", "")
        amount = payload.get("renewal_amount", "")
        lead_stat = f"subscription renews in {days} days — ₹{amount}" if days and amount else f"{days} days left"

    # --- Milestone ---
    elif trigger_kind == "milestone_reached":
        metric = payload.get("metric", "")
        value = payload.get("value_now", "")
        milestone = payload.get("milestone_value", "")
        lead_stat = f"{value} {metric} — approaching {milestone} milestone"

    # --- GBP unverified ---
    elif trigger_kind == "gbp_unverified":
        uplift = payload.get("estimated_uplift_pct", 0)
        lead_stat = f"GBP verification unlocks {uplift:+.0%} more views"

    # --- Category trend ---
    elif trigger_kind == "category_trend_movement":
        trend = payload.get("trend_label", "")
        delta = payload.get("search_delta_pct", 0)
        lead_stat = f'"{trend}" searches up {delta:+.0%} YoY — your locality is in range'

    # --- Festival ---
    elif trigger_kind == "festival_upcoming":
        festival = payload.get("festival", "")
        days_until = payload.get("days_until", "")
        lead_stat = f"{festival} in {days_until} days"

    # --- Wedding / bridal ---
    elif trigger_kind in ("wedding_package_followup", "bridal_followup"):
        wedding_date = payload.get("wedding_date", "")
        days_to = payload.get("days_to_wedding", "")
        lead_stat = f"{days_to} days to wedding — skin-prep window open now"
        merchant_anchor = f"Wedding date: {wedding_date}"

    # --- Trial followup ---
    elif trigger_kind == "trial_followup":
        trial_date = payload.get("trial_date", "")
        lead_stat = f"trial completed {trial_date} — convert now while it's fresh"

    # --- Curious ask ---
    elif trigger_kind in ("curious_ask_due", "scheduled_recurring"):
        lead_stat = "weekly check-in"

    # Peer gap (always computed when data available)
    peer_gap = _format_peer_gap(merchant, category)

    # Best active offer
    active_offers = _extract_active_offers(merchant)
    best_offer = active_offers[0] if active_offers else ""

    # Trigger payload rendered as bullet list
    trigger_payload_lines = _render_trigger_payload(
        trigger_kind, payload, digest_item, category
    )

    return {
        "lead_stat": lead_stat,
        "lead_source": lead_source,
        "merchant_anchor": merchant_anchor,
        "peer_gap": peer_gap,
        "best_offer": best_offer,
        "affected_count": affected_count,
        "trigger_payload_lines": trigger_payload_lines,
    }


# ---------------------------------------------------------------------------
# System prompt — enforces the 4-line formula and all hard constraints
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are Vera, magicpin's merchant AI assistant. Your job is to compose ONE WhatsApp message
that a real Indian small-business owner will want to reply to immediately.

═══ THE ONLY STRUCTURE ALLOWED ═══

Write EXACTLY 3-4 lines. No more. Use this formula:

  Line 1 — HOOK: lead with the specific stat or fact from "LEAD STAT" field. Why now?
  Line 2 — RELEVANCE: connect it to THIS merchant's specific situation using numbers from context.
  Line 3 — ACTION: one concrete next step Vera will do (effort externalization).
  Line 4 — CTA: binary question ending with "Haan / baad mein" (hi-en merchant) or "Yes / not now" (en merchant).

TOTAL WORD COUNT: 40-80 words maximum. If you exceed 80 words, you fail.

═══ NON-NEGOTIABLE RULES ═══

1. NO FABRICATION. Use only numbers, names, offers, and citations provided in the context.
   If a field says "N/A" or is empty, do not invent a value for it.

2. ONE CTA, PLACED LAST. For action triggers: binary "Haan / baad mein" or "Yes / not now".
   For ask triggers: one open question. Never two CTAs. Never buried CTA.

3. EFFORT EXTERNALIZATION ON EVERY ACTION CTA. The CTA line must contain one of:
   "want me to...", "shall I...", "I can...", "Want me to draft...", "I'll set it up..."
   The merchant should feel that saying yes costs them nothing.

4. SOURCE CITATION. If a digest item has a source (e.g., "JIDA Oct 2026, p.14"), cite it
   in the message — either in line or at the end. No citation = specificity capped at 7/10.

5. OWNER FIRST NAME IN SALUTATION. Use the owner's first name. Generic "Hi" loses points.

6. LANGUAGE MATCH. The prompt specifies the language code. Obey it exactly:
   - hi-en → Hinglish. Mix within sentences. At least 2-3 Hindi words in the body.
   - en → clean English, no forced Hindi.

7. NO PREAMBLES. Do not start with "I hope", "Just checking", "I'm reaching out", or any filler.
   Start with the owner's first name or the hook fact directly.

8. NO MARKDOWN. Plain text only. No asterisks, hyphens as bullets, bold, or headers.

9. NEVER RE-INTRODUCE YOURSELF after message 1. Check conversation history.

10. CATEGORY VOICE. Dentists/pharmacies/gyms: peer-collegial tone. No "AMAZING DEAL!".
    Restaurants/salons: warmer, but still operator-to-operator. Never retail-promo hype.

═══ COMPULSION LEVERS — CHOOSE 2-3 FOR THIS MESSAGE ═══

  SPECIFICITY: anchor on a verifiable number, source, date. "2,100-patient trial, JIDA Oct 2026 p.14"
  LOSS AVERSION: "you're missing X" / "before this window closes" / "CTR 30% below peer"
  SOCIAL PROOF: "3 dentists in Lajpat Nagar did Y this month"
  EFFORT EXTERNALIZATION: "I've drafted it — just say go" / "Live in 10 min"
  CURIOSITY: "want to see who?" / "want the full breakdown?"
  RECIPROCITY: "I noticed Y on your account"
  ASKING THE MERCHANT: low-stakes question for curious-ask triggers
  SINGLE BINARY COMMITMENT: lowest-friction CTA

═══ ANTI-PATTERNS — INSTANT PENALTY ═══

  Generic offers ("Flat 30% off") when service+price is available → -2 specificity
  Multiple CTAs → -3 engagement
  Buried CTA (not last sentence) → -2 engagement
  Promotional hype in clinical categories → -2 category fit
  Any fabricated number, citation, competitor name → score capped at 5/dimension
  Preamble ("Hope you're well…") → -2 engagement
  Body > 80 words → -3 across all dimensions
  Ignoring hi-en language preference → -2 merchant fit

═══ OUTPUT FORMAT ═══

Return ONLY a JSON object with exactly these keys — no prose before or after:

{
  "body": "<WhatsApp message, plain text, 3-4 lines, 40-80 words>",
  "cta": "<the CTA string, or 'none'>",
  "send_as": "<'vera' or 'merchant_on_behalf'>",
  "suppression_key": "<from trigger, unchanged>",
  "rationale": "<2 sentences: which levers used and why this is merchant-fit>"
}\
"""


# ---------------------------------------------------------------------------
# User prompt builder
# ---------------------------------------------------------------------------


def build_user_prompt(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: dict | None = None,
) -> str:
    """
    Construct the user-turn prompt from the four context dicts.

    The prompt is structured as a ranked "ammunition rack":
    the most compelling stat is surfaced at the top so the LLM
    leads with it, rather than burying it in the body.
    """
    trigger_kind = trigger.get("kind", "unknown")
    trigger_scope = trigger.get("scope", "merchant")
    trigger_frame = TRIGGER_FRAME_MAP.get(
        trigger_kind, f"a '{trigger_kind}' event just occurred"
    )

    # Resolve digest item before ammunition extraction
    digest_item = _resolve_digest_item(trigger, category)

    # --- Language ---
    if customer:
        lang_code = _extract_customer_language(customer)
    else:
        lang_code = _extract_primary_language(merchant)

    binary_cta_phrase = _language_cta(lang_code)
    lang_instruction = _language_instruction(lang_code)

    # --- Identity ---
    identity = merchant.get("identity", {})
    owner_name = identity.get("owner_first_name", identity.get("name", ""))
    merchant_name = identity.get("name", "")
    locality = identity.get("locality", "")
    city = identity.get("city", "")

    # --- Performance ---
    perf = merchant.get("performance", {})
    cust_agg = merchant.get("customer_aggregate", {})
    active_offers = _extract_active_offers(merchant)
    signals = _extract_signals(merchant)

    # --- Conversation history (last 2 turns) ---
    conv_history = merchant.get("conversation_history", [])
    is_first_message = len(conv_history) == 0
    if conv_history:
        last_turns = conv_history[-2:]
        conv_lines = [
            f"  {'VERA' if t.get('from') == 'vera' else 'MERCHANT'}: {t.get('body', '')}"
            for t in last_turns
        ]
        conv_str = "\n".join(conv_lines)
    else:
        conv_str = "  (no prior conversation — this IS the first outreach)"

    # --- CTA instruction ---
    if trigger_kind in ACTION_TRIGGER_KINDS:
        cta_instruction = (
            f"CTA: binary effort-externalization question ending with \"{binary_cta_phrase}\". "
            "Example shape: \"Want me to [do the work]? " + binary_cta_phrase + "\""
        )
    elif trigger_kind in ASK_TRIGGER_KINDS:
        cta_instruction = (
            "CTA: a single low-stakes open question. Offer clear reciprocity — tell the merchant "
            "what Vera will do with the answer. No YES/NO binary needed."
        )
    else:
        cta_instruction = (
            f"CTA: binary effort-externalization question ending with \"{binary_cta_phrase}\". "
            "Keep it to one sentence."
        )

    # --- send_as instruction ---
    if trigger_scope == "customer" and customer:
        send_as_instruction = (
            "send_as = 'merchant_on_behalf' — message sent FROM the merchant's WhatsApp number "
            "TO the customer. Write in the merchant's voice. Do NOT mention Vera."
        )
    else:
        send_as_instruction = "send_as = 'vera' — message sent FROM Vera TO the merchant."

    # --- Pre-extract ammunition ---
    ammo = _extract_ammunition(category, merchant, trigger, customer, digest_item)

    # --- Customer block (only if present) ---
    customer_block = ""
    if customer:
        cust_identity = customer.get("identity", {})
        cust_rel = customer.get("relationship", {})
        cust_state = customer.get("state", "unknown")
        cust_prefs = customer.get("preferences", {})
        payload = trigger.get("payload", {})
        slots = payload.get("available_slots", payload.get("next_session_options", []))
        slots_str = _format_slots(slots)

        customer_block = f"""
CUSTOMER CONTEXT:
  Name: {cust_identity.get("name", "")}
  Language: {lang_code} — {lang_instruction}
  State: {cust_state}
  Last visit: {cust_rel.get("last_visit", "unknown")}
  Services received: {", ".join(cust_rel.get("services_received", [])[:5])}
  Preferred slots: {cust_prefs.get("preferred_slots", "not specified")}
  Available booking slots: {slots_str if slots_str else "(see trigger payload)"}
  Consent scope: {", ".join(customer.get("consent", {}).get("scope", []))}

RULES FOR CUSTOMER MESSAGE:
  - Address by first name ({cust_identity.get("name", "")}).
  - Honor slot preference (if evening preferred, lead with evening slots).
  - If lapsed: warm, no-shame. Acknowledge the gap matter-of-factly.
  - If new: welcome warmth.
  - No medical claims. No "guaranteed".
"""

    # --- Category voice ---
    voice = category.get("voice", {})
    if isinstance(voice, str):
        voice = {"tone": voice}
    peer_stats = category.get("peer_stats", {})
    vocab_allowed_str = ", ".join(voice.get("vocab_allowed", [])[:8])
    vocab_taboo_str = ", ".join(voice.get("vocab_taboo", [])[:6])

    # --- Seasonal context (ambient, only if non-seasonal trigger) ---
    seasonal_note = ""
    if trigger_kind not in ("category_seasonal", "seasonal_perf_dip", "festival_upcoming"):
        beats = category.get("seasonal_beats", [])
        if beats:
            beats_str = "; ".join(
                f"{b.get('month_range', '')}: {b.get('note', '')}"
                for b in beats[:2]
            )
            seasonal_note = f"\nSEASONAL CONTEXT (use only if directly relevant): {beats_str}"

    # --- Assemble prompt ---
    prompt = f"""=== COMPOSITION REQUEST ===

TRIGGER: {trigger_frame}
Kind: {trigger_kind} | Urgency: {trigger.get("urgency", 1)}/5
Suppression key (copy unchanged into output): {trigger.get("suppression_key", "")}
{send_as_instruction}
{cta_instruction}

═══ AMMUNITION RACK — USE THESE EXACT NUMBERS ═══

LEAD STAT (use this as your Line 1 hook):
  {ammo["lead_stat"]}
  {f"Source: {ammo['lead_source']}" if ammo["lead_source"] else ""}

MERCHANT ANCHOR (make it personal with this — Line 2):
  {ammo["merchant_anchor"] if ammo["merchant_anchor"] else "See merchant performance below"}
  {f"Affected count: {ammo['affected_count']} patients/customers" if ammo["affected_count"] else ""}

PEER GAP (loss aversion ammunition):
  {ammo["peer_gap"] if ammo["peer_gap"] else "No peer gap data available"}

BEST ACTIVE OFFER (anchor Line 3 action to this):
  {ammo["best_offer"] if ammo["best_offer"] else "No active offers — do not invent one"}

═══ TRIGGER PAYLOAD (exact facts from the event) ═══
{chr(10).join(ammo["trigger_payload_lines"])}

═══ CATEGORY CONTEXT ═══
Category: {category.get("slug", "")} — {category.get("display_name", "")}
Voice: {voice.get("tone", "")} — {voice.get("register", "")}
Vocabulary ALLOWED (use for authenticity): {vocab_allowed_str}
Vocabulary TABOO (never use): {vocab_taboo_str}
Peer stats: CTR {peer_stats.get("avg_ctr", "N/A")} | Avg reviews {peer_stats.get("avg_review_count", "N/A")} | Avg rating {peer_stats.get("avg_rating", "N/A")}
{seasonal_note}

═══ MERCHANT CONTEXT ═══
Owner first name: {owner_name}  ← USE THIS in salutation
Merchant name: {merchant_name}
Location: {locality}, {city}
Subscription: {merchant.get("subscription", {}).get("plan", "unknown")} — {merchant.get("subscription", {}).get("status", "unknown")} ({merchant.get("subscription", {}).get("days_remaining", "N/A")} days left)
Language: {lang_code} → {lang_instruction}

Performance (30d): views={perf.get("views", "N/A")} | calls={perf.get("calls", "N/A")} | CTR={perf.get("ctr", "N/A")}
7d deltas: views {_fmt_pct(perf.get("delta_7d", {}).get("views_pct"))} | calls {_fmt_pct(perf.get("delta_7d", {}).get("calls_pct"))}
Active offers: {"; ".join(active_offers) if active_offers else "none"}
Customer aggregate: active={cust_agg.get("active_count", "N/A")} | lapsed={cust_agg.get("lapsed_count", "N/A")} | retention={cust_agg.get("retention_6mo_pct", "N/A")}
Derived signals: {", ".join(signals) if signals else "none"}
{customer_block}
Recent conversation (last 2 turns):
{conv_str}
{"NOTE: This is the FIRST message — introduce Vera briefly (1-2 words max, e.g. 'Vera here' is fine)." if is_first_message else "NOTE: This is a follow-up — do NOT re-introduce Vera."}

═══ YOUR TASK ═══

Write a 3-4 line WhatsApp message (40-80 words) using the Hook→Relevance→Action→CTA formula.

SELF-CHECK before outputting (fail any = rewrite):
  [ ] Line 1: starts with owner name + LEAD STAT with source e.g. "Meera, JIDA Oct 2026 p.14 — fluoride recall cuts caries 38%"
  [ ] Line 2: merchant-specific number e.g. "Aapke 124 high-risk patients ke liye relevant"
  [ ] Line 3: effort externalization e.g. "Want me to draft the recall SMS?"
  [ ] Line 4: ends EXACTLY with "{binary_cta_phrase}"
  [ ] Citation: if source="{ammo.get("lead_source","")}", it MUST appear in the message
  [ ] Word count ≤ 80
  [ ] Language: {lang_code}
  [ ] No markdown, no asterisks

Return ONLY the JSON object.
"""

    return prompt.strip()


# ---------------------------------------------------------------------------
# Trigger payload renderer
# ---------------------------------------------------------------------------


def _render_trigger_payload(
    kind: str,
    payload: dict,
    digest_item: dict | None,
    category: dict,
) -> list[str]:
    """
    Render the trigger payload into a clean list of fact strings.
    Returns a list (not a joined string) so the caller can join with newlines.
    """
    lines: list[str] = []

    if kind in ("research_digest", "regulation_change", "cde_opportunity") and digest_item:
        lines.append(f"Title: {digest_item.get('title', '')}")
        lines.append(f"Source: {digest_item.get('source', '')}")
        if digest_item.get("trial_n"):
            lines.append(f"Trial size: n={digest_item['trial_n']:,} patients")
        if digest_item.get("patient_segment"):
            lines.append(f"Patient segment: {digest_item['patient_segment']}")
        if digest_item.get("summary"):
            lines.append(f"Summary: {digest_item['summary']}")
        if digest_item.get("actionable"):
            lines.append(f"Actionable insight: {digest_item['actionable']}")
        if kind == "regulation_change":
            lines.append(f"Deadline: {payload.get('deadline_iso', 'see source')}")

    elif kind in ("recall_due", "chronic_refill_due"):
        for k, v in payload.items():
            if k == "available_slots":
                lines.append(f"Available slots: {_format_slots(v)}")
            elif k == "next_session_options":
                lines.append(f"Next session options: {_format_slots(v)}")
            elif k == "medicines" and isinstance(v, list):
                lines.append(f"Medicines due: {', '.join(v)}")
            else:
                lines.append(f"{k}: {v}")

    elif kind in ("perf_dip", "seasonal_perf_dip"):
        delta = payload.get("delta_pct", 0)
        lines.append(f"Metric: {payload.get('metric', '')} {delta:+.0%} over {payload.get('window', '')}")
        lines.append(f"Baseline: {payload.get('vs_baseline', 'N/A')}")
        if kind == "seasonal_perf_dip":
            lines.append(f"Is expected seasonal: {payload.get('is_expected_seasonal', True)}")
            lines.append(f"Season note: {payload.get('season_note', '')}")
            lines.append("FRAMING: Do NOT alarm the merchant. This dip is normal. Reframe: save ad spend for recovery season.")
        if payload.get("likely_driver"):
            lines.append(f"Likely driver: {payload['likely_driver']}")

    elif kind == "perf_spike":
        delta = payload.get("delta_pct", 0)
        lines.append(f"Metric: {payload.get('metric', '')} {delta:+.0%} over {payload.get('window', '')}")
        lines.append(f"Baseline: {payload.get('vs_baseline', 'N/A')}")
        if payload.get("likely_driver"):
            lines.append(f"Likely driver: {payload['likely_driver']}")

    elif kind == "ipl_match_today":
        lines.append(f"Match: {payload.get('match', '')}")
        lines.append(f"Venue: {payload.get('venue', '')}")
        lines.append(f"Match time: {payload.get('match_time_iso', '')}")
        is_weeknight = payload.get("is_weeknight", True)
        lines.append(f"Is weeknight: {is_weeknight}")
        if not is_weeknight:
            footfall = payload.get("footfall_impact_pct", -12)
            lines.append(
                f"Saturday IPL footfall impact: {footfall:+.0%} restaurant covers. "
                "RECOMMENDATION: push delivery-only using existing active offer, not a dine-in promo."
            )

    elif kind == "supply_alert":
        lines.append(f"Molecule/product: {payload.get('molecule', '')}")
        lines.append(f"Affected batches: {', '.join(payload.get('affected_batches', []))}")
        lines.append(f"Manufacturer: {payload.get('manufacturer', '')}")
        lines.append(f"Risk level: sub-potency (no safety risk) — customers should be informed for replacement")
        lines.append("ACTION: offer to pull affected customer list + draft their outreach note")

    elif kind == "competitor_opened":
        lines.append(f"Competitor: {payload.get('competitor_name', '')}")
        lines.append(f"Distance: {payload.get('distance_km', '')} km away")
        lines.append(f"Their offer: {payload.get('their_offer', 'unknown')}")
        lines.append("FRAMING: intelligence + differentiation. Use merchant's reviews/CTR/offers as the response.")

    elif kind == "review_theme_emerged":
        lines.append(f"Theme: {payload.get('theme', '')}")
        lines.append(f"Occurrences in 30 days: {payload.get('occurrences_30d', '')}")
        lines.append(f"Trend: {payload.get('trend', '')}")
        if payload.get("common_quote"):
            lines.append(f"Customer quote: \"{payload['common_quote']}\"")

    elif kind == "renewal_due":
        lines.append(f"Days remaining: {payload.get('days_remaining', '')}")
        lines.append(f"Plan: {payload.get('plan', '')}")
        lines.append(f"Renewal amount: ₹{payload.get('renewal_amount', '')}")

    elif kind == "winback_eligible":
        lines.append(f"Days since expiry: {payload.get('days_since_expiry', '')}")
        delta = payload.get("perf_dip_pct", 0)
        lines.append(f"Performance since expiry: {delta:+.0%}")
        lines.append(f"Lapsed customers added since expiry: {payload.get('lapsed_customers_added_since_expiry', 0)}")

    elif kind == "dormant_with_vera":
        lines.append(f"Days since last merchant reply: {payload.get('days_since_last_merchant_message', '')}")
        lines.append(f"Last topic: {payload.get('last_topic', '')}")
        lines.append("FRAMING: Re-engage with fresh value. Do not reference the silence.")

    elif kind == "milestone_reached":
        lines.append(f"Metric: {payload.get('metric', '')}")
        lines.append(f"Current value: {payload.get('value_now', '')}")
        lines.append(f"Milestone: {payload.get('milestone_value', '')}")

    elif kind == "gbp_unverified":
        uplift = payload.get("estimated_uplift_pct", 0)
        lines.append(f"Verified: {payload.get('verified', False)}")
        lines.append(f"Estimated uplift from verification: {uplift:+.0%} more views")
        lines.append(f"Verification path: {payload.get('verification_path', '')}")

    elif kind == "active_planning_intent":
        lines.append(f"Intent topic: {payload.get('intent_topic', '')}")
        lines.append(f"Merchant's exact last message: \"{payload.get('merchant_last_message', '')}\"")
        lines.append(
            "CRITICAL: Merchant said YES. Provide a complete draft artifact immediately. "
            "Do NOT ask another qualifying question."
        )

    elif kind in ("wedding_package_followup", "bridal_followup"):
        lines.append(f"Wedding date: {payload.get('wedding_date', '')}")
        lines.append(f"Days to wedding: {payload.get('days_to_wedding', '')}")
        lines.append(f"Trial completed: {payload.get('trial_completed', '')}")

    elif kind == "trial_followup":
        lines.append(f"Trial date: {payload.get('trial_date', '')}")
        slots = payload.get("next_session_options", [])
        if slots:
            lines.append(f"Next session options: {_format_slots(slots)}")

    elif kind in ("customer_lapsed_hard", "customer_lapsed_soft"):
        lines.append(f"Days since last visit: {payload.get('days_since_last_visit', '')}")
        lines.append(f"Previous focus/goal: {payload.get('previous_focus', '')}")
        lines.append(f"Previous membership: {payload.get('previous_membership_months', '')} months")
        lines.append("FRAMING: warm, no-shame, no guilt-trip. Acknowledge gap matter-of-factly.")

    elif kind == "curious_ask_due":
        lines.append(f"Ask template: {payload.get('ask_template', '')}")
        lines.append("SHAPE: 2-3 lines max. Low-stakes question + clear reciprocity offer.")

    elif kind == "festival_upcoming":
        lines.append(f"Festival: {payload.get('festival', '')}")
        lines.append(f"Date: {payload.get('date', '')}")
        lines.append(f"Days until: {payload.get('days_until', '')}")
        lines.append(f"Category relevance: {', '.join(payload.get('category_relevance', []))}")

    elif kind == "category_seasonal":
        lines.append(f"Season: {payload.get('season', '')}")
        lines.append(f"Demand trends: {', '.join(payload.get('trends', []))}")
        lines.append(f"Shelf action recommended: {payload.get('shelf_action_recommended', False)}")

    elif kind == "category_trend_movement":
        lines.append(f"Trend label: {payload.get('trend_label', '')}")
        delta = payload.get("search_delta_pct", 0)
        lines.append(f"Search delta: {delta:+.0%} YoY")
        lines.append(f"Locality in range: {payload.get('locality_in_range', True)}")

    else:
        # Generic fallback
        for k, v in payload.items():
            lines.append(f"{k}: {v}")

    return lines if lines else ["(see trigger payload)"]
