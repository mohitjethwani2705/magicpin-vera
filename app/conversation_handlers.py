"""
conversation_handlers.py — Multi-turn conversation engine for Vera.

Handles all reply scenarios defined in challenge-testing-brief.md §4 (Phase 4):
  1. Auto-reply detection  — same/similar message 3+ times → graceful exit
  2. Intent transition     — positive signals ("ok", "haan", "let's do it") → action mode
  3. Hostile / disengaged  — "not interested" / "stop" / abuse → acknowledge, end
  4. Question handling     — merchant asks a question → answer from context, re-anchor CTA
  5. Clarification request — "tell me more" → one specific data point, re-ask

Architecture:
  - ConversationPhase  — enum of valid state machine phases
  - ConversationState  — dataclass tracking all mutable per-conversation state
  - handle_reply()     — top-level dispatcher called by the FastAPI route
  - _detect_*()        — pure signal-detection helpers (no LLM, deterministic)
  - _compose_*()       — LLM call builders for each scenario
"""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

# Belt-and-suspenders .env load — config.py does this too, but we load here
# in case conversation_handlers is ever imported before config initialises.
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.logger import get_logger

log = get_logger(__name__)

# Pull Groq config once at module load — mirrors bot.py pattern exactly.
GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL: str = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")


# ---------------------------------------------------------------------------
# Turn key normaliser — the store saves {role, body, ts}; state saves
# {role, content, timestamp}. _get_content handles both without KeyError.
# ---------------------------------------------------------------------------

def _get_content(turn: dict) -> str:
    """Return the message text from a turn dict regardless of which key is used."""
    return turn.get("content") or turn.get("body", "")


# ---------------------------------------------------------------------------
# Phase enum — no magic strings anywhere below this line
# ---------------------------------------------------------------------------

class ConversationPhase(str, Enum):
    OPENING = "opening"       # First outbound sent; waiting for first reply
    QUALIFYING = "qualifying" # Gathering intent / building case
    ACTION = "action"         # Merchant committed; executing the promised step
    CLOSING = "closing"       # Wrapping up; final confirmation or handover
    ENDED = "ended"           # Conversation closed (gracefully or otherwise)


# ---------------------------------------------------------------------------
# ConversationState dataclass
# ---------------------------------------------------------------------------

@dataclass
class ConversationState:
    """All mutable state for one in-flight conversation.

    Stored in the bot's conversations dict keyed by conversation_id.
    Serialise to dict with `state.__dict__` if you need to persist externally.
    """

    merchant_id: str
    phase: ConversationPhase = ConversationPhase.OPENING

    # Full turn log — each entry: {role, content, timestamp}
    # Also tolerates {role, body, ts} from the external store (use _get_content).
    turns: list[dict] = field(default_factory=list)

    # Positive intent signals collected across turns (e.g. "ok", "let's do it")
    intent_signals: list[str] = field(default_factory=list)

    # How many consecutive messages look like an auto-reply
    auto_reply_count: int = 0

    # The last message Vera sent (used for anti-repetition check)
    last_bot_message: str = ""

    # Suppression key — set when we end; caller can store to block re-contact
    suppression_key: str = ""

    def add_turn(self, role: str, content: str) -> None:
        self.turns.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            }
        )

    def recent_merchant_messages(self, n: int = 5) -> list[str]:
        """Last N messages from the merchant/customer side."""
        return [
            _get_content(t)
            for t in self.turns[-n * 2:]   # rough window
            if t["role"] in ("merchant", "customer")
        ][-n:]


# ---------------------------------------------------------------------------
# LLM call — Groq REST API (direct requests.post, same pattern as bot.py)
# ---------------------------------------------------------------------------

def _groq_chat(messages: list[dict], max_tokens: int = 150) -> str:
    """
    POST to the Groq OpenAI-compatible endpoint and return the reply text.
    Handles 429 rate limits with exponential backoff (up to 3 retries).
    """
    import time as _time
    for attempt in range(3):
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
                "User-Agent": "vera-bot/1.0",
            },
            json={
                "model": GROQ_MODEL,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.5,
            },
            timeout=20,
        )
        if response.status_code == 429:
            wait = 2 ** (attempt + 1)  # 2, 4, 8 seconds
            log.warning("groq_rate_limited", attempt=attempt + 1, wait=wait)
            _time.sleep(wait)
            continue
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    raise RuntimeError("Groq rate limit exceeded after 3 retries")


# ---------------------------------------------------------------------------
# Signal detection — deterministic, no LLM
# ---------------------------------------------------------------------------

# Canonical positive-intent patterns (Hindi + English + Hinglish)
_POSITIVE_INTENT_PATTERNS = re.compile(
    r"\b("
    r"let'?s do it|let'?s go|yes|haan|ha[n]?|ok(?:ay)?|sure|go ahead|proceed"
    r"|bilkul|zaroor|theek hai|chalo|kar do|bhejna|share karo|send it"
    r"|i['']?m in|sign me up|register|join|ready|confirm|approved"
    r"|interesting|achha|sounds good|great idea|perfect|done"
    r")\b",
    re.IGNORECASE,
)

# Patterns that signal disengagement or hostility
_EXIT_PATTERNS = re.compile(
    r"\b("
    r"not interested|nahi chahiye|nahi|no thanks|stop|unsubscribe|remove"
    r"|leave me alone|busy|baad mein|later|koi zaroorat nahi|don'?t contact"
    r"|spam|block|report|abusive|shut up|go away|mujhe mat karo"
    r"|disturb mat karo|disturb na karo|mat bhejo|band karo"
    r")\b",
    re.IGNORECASE,
)

# Patterns that signal a direct question from the merchant
_QUESTION_PATTERNS = re.compile(
    r"(\?|kaise|kyun|kya hai|what is|how|when|kaun|which|kitna|how much"
    r"|kab|tell me more|explain|details|aur batao|samjhao)",
    re.IGNORECASE,
)

# Patterns that signal a clarification / more-info request
_CLARIFICATION_PATTERNS = re.compile(
    r"\b(tell me more|more info|aur batao|elaborate|details|explain|"
    r"samjhao|kya matlab|what do you mean|not clear|confused)\b",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace — for similarity checks."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text.lower().strip())


def _similarity_ratio(a: str, b: str) -> float:
    """
    Simple character-level Jaccard similarity on trigrams.
    Fast, dependency-free, good enough for auto-reply detection.
    """
    def trigrams(s: str) -> set[str]:
        return {s[i: i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else {s}

    na, nb = _normalize(a), _normalize(b)
    ta, tb = trigrams(na), trigrams(nb)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _is_auto_reply(
    incoming: str,
    recent_messages: list[str],
    threshold: float = None,
) -> bool:
    """
    Return True if `incoming` is suspiciously similar to messages already seen.
    Also catches verbatim repeats.
    """
    if threshold is None:
        threshold = settings.auto_reply_similarity_threshold

    for prev in recent_messages:
        if _normalize(incoming) == _normalize(prev):
            return True
        if _similarity_ratio(incoming, prev) >= threshold:
            return True
    return False


def _detect_intent_transition(message: str) -> bool:
    return bool(_POSITIVE_INTENT_PATTERNS.search(message))


def _detect_exit_signal(message: str) -> bool:
    return bool(_EXIT_PATTERNS.search(message))


def _detect_question(message: str) -> bool:
    return bool(_QUESTION_PATTERNS.search(message))


def _detect_clarification(message: str) -> bool:
    return bool(_CLARIFICATION_PATTERNS.search(message))


# ---------------------------------------------------------------------------
# Context extraction helpers — pull real data, never hallucinate
# ---------------------------------------------------------------------------

def _merchant_first_name(merchant: dict) -> str:
    """Extract the first word of the merchant's name (typically the owner's name)."""
    name = merchant.get("identity", {}).get("name", "")
    # Strip "Dr." / "Mr." / "Mrs." prefixes if followed by a word
    name = re.sub(r"^(Dr\.|Mr\.|Mrs\.|Ms\.)\s*", "", name, flags=re.IGNORECASE).strip()
    return name.split()[0] if name else "ji"


def _active_offer_title(merchant: dict) -> str:
    """Return the title of the first active offer, or empty string."""
    # Dataset format: offers = [{status, title, ...}]
    for offer in merchant.get("offers", []):
        if isinstance(offer, dict) and offer.get("status") == "active":
            return offer.get("title", "")
    # Flat format: active_offers = ["offer title", ...]
    flat = merchant.get("active_offers", [])
    if flat and isinstance(flat[0], str):
        return flat[0]
    return ""


def _lapsed_patient_count(merchant: dict) -> int:
    """Return the lapsed patient count from customer_aggregate."""
    return merchant.get("customer_aggregate", {}).get("lapsed_180d_plus", 0)


def _top_digest_item(category: dict) -> dict:
    """Return the first digest item or an empty dict."""
    digest = category.get("digest", [])
    return digest[0] if digest else {}


def _ctr_gap_sentence(merchant: dict, category: dict) -> str:
    """Return a one-line CTR gap sentence using real numbers, or empty string."""
    ctr_m = merchant.get("performance", {}).get("ctr")
    ctr_p = category.get("peer_stats", {}).get("avg_ctr")
    if ctr_m and ctr_p and ctr_p > 0:
        gap_pct = round((ctr_p - ctr_m) / ctr_p * 100)
        return (
            f"Aapka CTR {ctr_m:.1%} hai, category median {ctr_p:.1%} hai "
            f"— {gap_pct}% ka gap hai."
        )
    return ""


# ---------------------------------------------------------------------------
# System prompt builder — grounded, never hallucinates
# ---------------------------------------------------------------------------

def _build_system_prompt(category: dict, merchant: dict) -> str:
    """
    Craft a category-and-merchant-aware system prompt.
    All facts injected here come directly from the context dicts — no invention.
    """
    cat_slug = category.get("slug", "") or category.get("category_id", "general")
    _voice = category.get("voice", {})
    if isinstance(_voice, str):
        _voice = {"tone": _voice}
    tone = _voice.get("tone", "peer")
    taboos = _voice.get("taboos", [])
    _identity = merchant.get("identity", {})
    merchant_name = _identity.get("name") or merchant.get("business_name", "the merchant")
    languages = _identity.get("languages") or [merchant.get("language_preference", "en")]
    signals = merchant.get("signals", [])
    lapsed = _lapsed_patient_count(merchant)
    offer = _active_offer_title(merchant)
    digest = _top_digest_item(category)

    taboo_str = ", ".join(f'"{t}"' for t in taboos) if taboos else "none"
    lang_str = " and ".join(languages)

    # Build a compact fact block the model can draw on without inventing anything.
    fact_lines = []
    if lapsed:
        fact_lines.append(f"- Lapsed customers (180d+): {lapsed}")
    if offer:
        fact_lines.append(f"- Active offer: {offer}")
    if digest:
        fact_lines.append(
            f"- Top research item: \"{digest.get('title', '')}\" "
            f"(source: {digest.get('source', 'N/A')}, n={digest.get('trial_n', 'N/A')})"
        )
    perf = merchant.get("performance", {})
    if perf.get("views"):
        fact_lines.append(f"- Profile views (30d): {perf['views']:,}")
    if perf.get("ctr"):
        fact_lines.append(f"- Current CTR: {perf['ctr']:.1%}")
    peer_ctr = category.get("peer_stats", {}).get("avg_ctr")
    if peer_ctr:
        fact_lines.append(f"- Peer median CTR: {peer_ctr:.1%}")

    fact_block = "\n".join(fact_lines) if fact_lines else "  (none available)"

    return (
        f"You are Vera, magicpin's merchant AI assistant. "
        f"You are mid-conversation with {merchant_name} (category: {cat_slug}). "
        f"Voice: {tone}. Language: {lang_str} (use Hindi-English code-mix if both hi and en are listed). "
        f"Forbidden words: {taboo_str}. "
        f"Merchant signals: {', '.join(signals) if signals else 'none'}.\n\n"
        f"VERIFIED FACTS YOU MAY USE (do not invent anything outside this list):\n"
        f"{fact_block}\n\n"
        "RULES — non-negotiable:\n"
        "- Never introduce yourself again after the first message.\n"
        "- Only cite facts that appear in the VERIFIED FACTS list above or in the conversation history.\n"
        "- Never invent offers, prices, research citations, or statistics.\n"
        "- Always end with exactly ONE clear call-to-action (binary YES/STOP or a single question).\n"
        "- Keep the reply under 80 words unless the merchant asked a detailed question.\n"
        "- Never repeat a message you already sent verbatim.\n"
        "- Match the merchant's language — if they write in Hindi, reply in Hindi/Hinglish.\n"
    )


# ---------------------------------------------------------------------------
# Conversation history builder
# ---------------------------------------------------------------------------

def _build_conversation_history(state: ConversationState) -> list[dict]:
    """Convert state.turns into the messages list expected by the chat API.

    Handles both {role, content, timestamp} (internal) and {role, body, ts}
    (external store) via _get_content().
    """
    role_map = {
        "bot": "assistant",
        "vera": "assistant",
        "merchant": "user",
        "customer": "user",
    }
    messages = []
    for turn in state.turns:
        mapped = role_map.get(turn["role"], "user")
        content = _get_content(turn)
        if content:  # skip empty turns that might appear from store normalization
            messages.append({"role": mapped, "content": content})
    return messages


# ---------------------------------------------------------------------------
# LLM call with retry
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    reraise=True,
)
def _call_llm(
    system: str,
    history: list[dict],
    user_message: str,
    extra_instruction: str = "",
    max_tokens: int = 150,
) -> str:
    """
    Single LLM call via Groq REST API (OpenAI-compatible).
    Retries once with back-off on transient failures.
    Raises on second failure so the caller can fall back to a safe canned reply.

    max_tokens=150 keeps replies short — one response, no essays.
    The full conversation history is always included so the LLM has context.
    """
    messages: list[dict] = [{"role": "system", "content": system}]
    messages.extend(history)  # full history gives the LLM complete context

    if extra_instruction:
        # Inject a brief instruction before the final user turn so the model
        # sees it as the immediate next directive.
        messages.append({"role": "system", "content": extra_instruction})

    messages.append({"role": "user", "content": user_message})

    return _groq_chat(messages, max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# Canned exit messages — deterministic, Hinglish, judge-friendly
# ---------------------------------------------------------------------------

def _compose_graceful_exit(merchant_name: str, reason: str) -> str:
    """
    Deterministic graceful exit — no LLM, always safe.
    Messages are in Hinglish to match the judge's expected exit patterns.
    """
    first = _first_word(merchant_name)

    if reason == "auto_reply":
        # Scenario 1: Auto-reply hell — bot detected 3+ identical messages.
        # Expected pattern from testing brief: "Samajh gaya Meera ji, koi baat nahi. Jab zarurat ho tab batayein."
        return (
            f"Samajh gaya {first} ji, koi baat nahi. "
            f"Jab zarurat ho tab batayein. \U0001F64F"
        )

    # Scenario 3: Hostile / disengaged — "stop", "not interested", "busy", etc.
    # Expected pattern from testing brief: "Bilkul Meera ji, disturb nahi karunga. Jab ready hon tab main yahan hun."
    # Note: repeated "Not interested" (same message 3x) reaches here on turn 1 via exit-signal,
    # which is correct behavior — disengaged on first explicit refusal.
    # The auto_reply path fires when the repeated message is a *neutral* WA Business auto-reply.
    return (
        f"Bilkul {first} ji, disturb nahi karunga. "
        f"Jab ready hon tab main yahan hun. \U0001F64F"
    )


def _safe_question_fallback(category: dict) -> str:
    """Canned fallback for question replies when LLM is unavailable."""
    return (
        "Aapka sawaal samajh aaya. Main details check karke kal tak jawab dunga. "
        "Tab tak koi aur sawaal ho toh batayein?"
    )


def _safe_action_fallback(merchant: dict, category: dict) -> str:
    """Canned fallback when LLM is unavailable (e.g. rate limited)."""
    identity = merchant.get("identity", {})
    name = identity.get("name") or merchant.get("business_name", "")
    first = _first_word(name) if name else "ji"
    return (
        f"Bilkul {first} ji! Main aapke liye yeh kaam abhi shuru karta hoon. "
        f"Kal tak update karunga. Confirm karein?"
    )


def _first_word(name: str) -> str:
    """Return the first non-honorific word of a merchant name."""
    name = re.sub(r"^(Dr\.|Mr\.|Mrs\.|Ms\.)\s*", "", name, flags=re.IGNORECASE).strip()
    parts = name.split()
    return parts[0] if parts else "ji"


# ---------------------------------------------------------------------------
# Scenario-specific reply composers
# ---------------------------------------------------------------------------

def _compose_action_reply(
    state: ConversationState,
    merchant_message: str,
    category: dict,
    merchant: dict,
    customer: Optional[dict],
) -> str:
    """
    Merchant has committed. Switch to concrete action mode immediately.

    Scenario 2 requirement: provide ONE concrete next step — no more qualifying.
    Example: "Great! Main aapke 124 lapsed patients ke liye recall SMS draft kar
    deta hun. Kal tak ready hoga. Confirm karein?"
    """
    system = _build_system_prompt(category, merchant)
    history = _build_conversation_history(state)

    # Build a tight action instruction using real merchant data only.
    lapsed = _lapsed_patient_count(merchant)
    offer = _active_offer_title(merchant)
    merchant_name = merchant.get("identity", {}).get("name", "")
    first = _first_word(merchant_name)
    digest = _top_digest_item(category)

    # Choose the most specific action anchor available in context.
    action_anchor_parts = []
    if lapsed:
        action_anchor_parts.append(
            f"Draft recall SMS for their {lapsed} lapsed patients"
        )
    if offer:
        action_anchor_parts.append(
            f"Activate the '{offer}' offer campaign"
        )
    if digest.get("title"):
        action_anchor_parts.append(
            f"Pull the research abstract: \"{digest['title']}\" ({digest.get('source', '')})"
        )

    action_anchor = (
        action_anchor_parts[0]
        if action_anchor_parts
        else "Proceed with the next step discussed"
    )

    instruction = (
        f"CRITICAL: The merchant just said YES/agreed. Do NOT ask any more qualifying questions. "
        f"Switch immediately to action mode. "
        f"Provide exactly ONE concrete next step using this real data point: {action_anchor}. "
        f"Format: confirm what you will do, give a realistic timeline (e.g. 'kal tak'), "
        f"end with 'Confirm karein?' or 'Theek hai?'. "
        f"Reply in Hindi-English code-mix. Max 60 words. "
        f"Address them as '{first} ji'."
    )

    return _call_llm(system, history, merchant_message, extra_instruction=instruction, max_tokens=150)


def _compose_question_reply(
    state: ConversationState,
    merchant_message: str,
    category: dict,
    merchant: dict,
    customer: Optional[dict],
) -> str:
    """
    Answer the merchant's question using one concrete data point from context,
    then re-anchor back to the conversation's CTA.
    """
    system = _build_system_prompt(category, merchant)
    history = _build_conversation_history(state)

    # Surface the single most relevant digest item as a fact anchor.
    digest = _top_digest_item(category)
    digest_hint = ""
    if digest:
        digest_hint = (
            f"Relevant verified fact: \"{digest.get('title', '')}\" "
            f"(source: {digest.get('source', '')}, n={digest.get('trial_n', 'N/A')}). "
            f"Use this if relevant to their question — do not invent other facts."
        )

    ctr_hint = _ctr_gap_sentence(merchant, category)

    instruction = (
        "Answer the merchant's question using ONLY facts from the VERIFIED FACTS block "
        "or the conversation history. Do not invent any numbers, offers, or citations. "
        "Give one concrete answer (1-2 sentences), then re-anchor with a single CTA. "
        f"{digest_hint} {ctr_hint}".strip()
    )

    return _call_llm(system, history, merchant_message, extra_instruction=instruction, max_tokens=150)


def _compose_clarification_reply(
    state: ConversationState,
    merchant_message: str,
    category: dict,
    merchant: dict,
    customer: Optional[dict],
) -> str:
    """
    Give exactly one specific data point they asked about, then re-ask the CTA.
    Resist the temptation to dump everything — one fact, one ask.
    """
    system = _build_system_prompt(category, merchant)
    history = _build_conversation_history(state)

    one_fact = _ctr_gap_sentence(merchant, category)
    if not one_fact:
        # Fall back to lapsed count or digest item — whichever is available.
        lapsed = _lapsed_patient_count(merchant)
        digest = _top_digest_item(category)
        if lapsed:
            one_fact = f"Aapke {lapsed} patients 180 din se clinic nahi aaye."
        elif digest.get("trial_n"):
            one_fact = (
                f"\"{digest.get('title', '')}\" — "
                f"{digest.get('trial_n')} patients pe trial hua tha "
                f"({digest.get('source', '')})."
            )

    instruction = (
        "The merchant wants more information. Provide EXACTLY ONE specific, "
        "verifiable data point from the VERIFIED FACTS block (not a list, not a paragraph). "
        "Then re-ask your original CTA as a single YES/STOP question. "
        f"Data point to use: {one_fact if one_fact else 'use the most relevant fact from VERIFIED FACTS'}. "
        "Max 50 words total."
    )

    return _call_llm(system, history, merchant_message, extra_instruction=instruction, max_tokens=150)


def _compose_generic_continuation(
    state: ConversationState,
    merchant_message: str,
    category: dict,
    merchant: dict,
    customer: Optional[dict],
) -> str:
    """
    Fallback for messages that don't fit a clear pattern.
    Acknowledge briefly, stay on mission, end with a binary CTA.
    """
    system = _build_system_prompt(category, merchant)
    history = _build_conversation_history(state)

    instruction = (
        "Acknowledge the merchant's message in 1 sentence. "
        "Then continue toward the original conversation goal using only facts from VERIFIED FACTS. "
        "End with exactly one binary CTA (YES/STOP question or a single direct question). "
        "Max 60 words. Do not hallucinate any data not in VERIFIED FACTS."
    )

    return _call_llm(system, history, merchant_message, extra_instruction=instruction, max_tokens=150)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def handle_reply(
    merchant_message: str,
    conversation_state: ConversationState,
    merchant: dict,
    category: dict,
    customer: Optional[dict] = None,
) -> dict:
    """
    Handle a merchant (or customer) reply in a multi-turn conversation.

    Parameters
    ----------
    merchant_message     : The raw incoming message text.
    conversation_state   : Mutable state object for this conversation.
                           Mutated in-place (turns appended, phase updated).
    merchant             : MerchantContext payload dict.
    category             : CategoryContext payload dict.
    customer             : Optional CustomerContext payload dict.

    Returns
    -------
    dict with keys:
        response_body : str   — the text Vera sends back (empty string if action=end)
        new_state     : ConversationState — updated state (same object, mutated)
        should_end    : bool  — True when the conversation must close
        cta           : str   — "open_ended" | "binary_yes_no" | "none"
        action        : str   — "send" | "end" | "wait"
        rationale     : str   — brief explanation for the judge log
    """
    merchant_name = merchant.get("identity", {}).get("name", "merchant")

    # Capture recent merchant messages BEFORE adding the incoming turn,
    # so we have a clean set of prior messages to compare against.
    prior_merchant_messages = conversation_state.recent_merchant_messages(n=4)

    log.info(
        "handle_reply_start",
        merchant_id=conversation_state.merchant_id,
        phase=conversation_state.phase,
        turn=len(conversation_state.turns) + 1,
        auto_reply_count=conversation_state.auto_reply_count,
    )

    # Record the incoming message before doing anything else.
    conversation_state.add_turn("merchant", merchant_message)

    # Build the full recent set (includes current message) for display/logging,
    # but use prior_merchant_messages for auto-reply comparison.
    recent = prior_merchant_messages

    # ------------------------------------------------------------------
    # Guard: conversation already ended
    # ------------------------------------------------------------------
    if conversation_state.phase == ConversationPhase.ENDED:
        log.warning(
            "reply_on_ended_conversation",
            merchant_id=conversation_state.merchant_id,
        )
        return _end_result(
            conversation_state,
            rationale="Conversation was already in ENDED state.",
        )

    # ------------------------------------------------------------------
    # Guard: max turns reached — close gracefully before getting stale
    # ------------------------------------------------------------------
    bot_turn_count = sum(
        1 for t in conversation_state.turns if t["role"] in ("bot", "vera")
    )
    if bot_turn_count >= settings.max_conversation_turns:
        first = _first_word(merchant_name)
        exit_body = (
            f"Main samajhta hun {first} ji — hum sab important points cover kar chuke hain. "
            f"Jab bhi zarurat ho, main yahan hun. \U0001f64f"
        )
        return _send_and_end(
            conversation_state,
            exit_body,
            cta="none",
            rationale=f"Max conversation turns ({settings.max_conversation_turns}) reached; closing gracefully.",
        )

    # ------------------------------------------------------------------
    # Scenario 1 — Auto-reply detection
    #
    # Threshold semantics: "3+ times" means the same message appears 3 times
    # total. On the 1st occurrence there are no prior messages (recent=[]),
    # so auto_reply_count stays 0. By the 3rd occurrence, auto_reply_count
    # reaches 2 (incremented on occurrence 2 and 3). We trigger at count >= 2
    # (i.e., threshold - 1) because auto_reply_count represents "number of
    # times we have seen this message AFTER the first time", so count=2 means
    # the message appeared 3 times total.
    # ------------------------------------------------------------------
    # Count total merchant turns — if merchant keeps sending short/unclear
    # messages without real engagement, treat as auto-reply hell after 3 turns.
    merchant_turn_count = sum(
        1 for t in conversation_state.turns if t["role"] in ("merchant", "customer")
    )

    if _is_auto_reply(merchant_message, recent):
        # Recount from full turn history so count survives across requests
        all_merchant = [
            t.get("content", t.get("body", ""))
            for t in conversation_state.turns
            if t.get("role") in ("merchant", "customer")
        ]
        if len(all_merchant) >= 2:
            last = _normalize(all_merchant[-1])
            count = sum(
                1 for m in all_merchant[:-1]
                if _similarity_ratio(last, _normalize(m)) >= settings.auto_reply_similarity_threshold
            )
            conversation_state.auto_reply_count = count
        else:
            conversation_state.auto_reply_count += 1
        log.info(
            "auto_reply_detected",
            merchant_id=conversation_state.merchant_id,
            count=conversation_state.auto_reply_count,
        )
    else:
        # Real engagement — reset the counter.
        conversation_state.auto_reply_count = 0

    # Trigger on similarity threshold OR after 3+ merchant turns with no intent
    no_intent = not _detect_intent_transition(merchant_message) and not _detect_exit_signal(merchant_message)
    force_exit = merchant_turn_count >= 3 and no_intent and conversation_state.phase == ConversationPhase.OPENING

    if conversation_state.auto_reply_count >= max(1, settings.auto_reply_threshold - 1) or force_exit:
        exit_body = _compose_graceful_exit(merchant_name, reason="auto_reply")
        conversation_state.suppression_key = (
            f"auto_reply:{conversation_state.merchant_id}"
        )
        return _send_and_end(
            conversation_state,
            exit_body,
            cta="none",
            rationale=(
                f"Auto-reply detected {conversation_state.auto_reply_count} consecutive times "
                f"(threshold={settings.auto_reply_threshold}); exiting gracefully."
            ),
        )

    # ------------------------------------------------------------------
    # Scenario 3 — Hostile / disengaged signal
    # Check BEFORE intent transition so "nahi" doesn't accidentally match
    # a positive-intent pattern on a misfire.
    # ------------------------------------------------------------------
    if _detect_exit_signal(merchant_message):
        log.info(
            "exit_signal_detected",
            merchant_id=conversation_state.merchant_id,
            message_snippet=merchant_message[:60],
        )
        exit_body = _compose_graceful_exit(merchant_name, reason="disengaged")
        conversation_state.suppression_key = (
            f"disengaged:{conversation_state.merchant_id}"
        )
        return _send_and_end(
            conversation_state,
            exit_body,
            cta="none",
            rationale="Merchant signaled disengagement or hostility; graceful exit.",
        )

    # ------------------------------------------------------------------
    # Scenario 2 — Intent transition (positive commitment)
    # Must fire AFTER exit-signal check (so "nahi" doesn't bleed through).
    # ------------------------------------------------------------------
    if _detect_intent_transition(merchant_message) and conversation_state.phase in (
        ConversationPhase.OPENING,
        ConversationPhase.QUALIFYING,
    ):
        conversation_state.intent_signals.append(merchant_message[:80])
        conversation_state.phase = ConversationPhase.ACTION
        log.info(
            "intent_transition",
            merchant_id=conversation_state.merchant_id,
            new_phase=conversation_state.phase,
        )
        try:
            body = _compose_action_reply(
                conversation_state, merchant_message, category, merchant, customer
            )
        except Exception:
            log.exception(
                "llm_action_reply_failed",
                merchant_id=conversation_state.merchant_id,
            )
            body = _safe_action_fallback(merchant, category)

        conversation_state.add_turn("bot", body)
        conversation_state.last_bot_message = body
        return {
            "response_body": body,
            "new_state": conversation_state,
            "should_end": False,
            "cta": "binary_yes_no",
            "action": "send",
            "rationale": (
                "Merchant committed; switched to action mode with one concrete next step. "
                "No further qualifying questions asked."
            ),
        }

    # ------------------------------------------------------------------
    # Scenario 5 — Clarification request
    # ------------------------------------------------------------------
    if _detect_clarification(merchant_message):
        log.info("clarification_requested", merchant_id=conversation_state.merchant_id)
        try:
            body = _compose_clarification_reply(
                conversation_state, merchant_message, category, merchant, customer
            )
        except Exception:
            log.exception(
                "llm_clarification_failed",
                merchant_id=conversation_state.merchant_id,
            )
            body = _safe_clarification_fallback(merchant, category)

        conversation_state.add_turn("bot", body)
        conversation_state.last_bot_message = body
        return {
            "response_body": body,
            "new_state": conversation_state,
            "should_end": False,
            "cta": "binary_yes_no",
            "action": "send",
            "rationale": "Merchant requested clarification; gave one data point from context, re-asked CTA.",
        }

    # ------------------------------------------------------------------
    # Scenario 4 — Question handling
    # ------------------------------------------------------------------
    if _detect_question(merchant_message):
        log.info("question_detected", merchant_id=conversation_state.merchant_id)
        try:
            body = _compose_question_reply(
                conversation_state, merchant_message, category, merchant, customer
            )
        except Exception:
            log.exception(
                "llm_question_reply_failed",
                merchant_id=conversation_state.merchant_id,
            )
            body = _safe_question_fallback(category)

        conversation_state.add_turn("bot", body)
        conversation_state.last_bot_message = body
        return {
            "response_body": body,
            "new_state": conversation_state,
            "should_end": False,
            "cta": "open_ended",
            "action": "send",
            "rationale": "Merchant asked a question; answered from context and re-anchored to CTA.",
        }

    # ------------------------------------------------------------------
    # Default — generic continuation
    # ------------------------------------------------------------------
    log.info("generic_continuation", merchant_id=conversation_state.merchant_id)
    try:
        body = _compose_generic_continuation(
            conversation_state, merchant_message, category, merchant, customer
        )
    except Exception:
        log.exception(
            "llm_generic_failed",
            merchant_id=conversation_state.merchant_id,
        )
        body = _safe_generic_fallback(merchant, category)

    conversation_state.add_turn("bot", body)
    conversation_state.last_bot_message = body
    return {
        "response_body": body,
        "new_state": conversation_state,
        "should_end": False,
        "cta": "binary_yes_no",
        "action": "send",
        "rationale": "Standard continuation; no clear intent signal detected.",
    }


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def _end_result(state: ConversationState, rationale: str) -> dict:
    """Return a terminal result without sending a message."""
    state.phase = ConversationPhase.ENDED
    return {
        "response_body": "",
        "new_state": state,
        "should_end": True,
        "cta": "none",
        "action": "end",
        "rationale": rationale,
    }


def _send_and_end(
    state: ConversationState,
    body: str,
    cta: str,
    rationale: str,
) -> dict:
    """Send one final message, then close the conversation."""
    state.phase = ConversationPhase.ENDED
    state.add_turn("bot", body)
    state.last_bot_message = body
    return {
        "response_body": body,
        "new_state": state,
        "should_end": True,
        "cta": cta,
        "action": "end",
        "rationale": rationale,
    }


# ---------------------------------------------------------------------------
# Safe fallbacks — used when the LLM call fails; never hallucinate
# ---------------------------------------------------------------------------

def _safe_action_fallback(merchant: dict, category: dict) -> str:
    """Fallback when action-mode LLM call fails. Uses real context only."""
    lapsed = _lapsed_patient_count(merchant)
    offer = _active_offer_title(merchant)
    merchant_name = merchant.get("identity", {}).get("name", "")
    first = _first_word(merchant_name)

    if lapsed:
        return (
            f"Great {first} ji! Main aapke {lapsed} lapsed patients ke liye "
            f"recall message draft kar deta hun. Kal tak ready hoga. Confirm karein?"
        )
    if offer:
        return (
            f"Perfect {first} ji! Main '{offer}' campaign abhi activate kar deta hun. "
            f"Theek hai?"
        )
    return (
        f"Shukriya {first} ji! Main aage ka kaam shuru kar deta hun — "
        f"kal tak update karunga. Theek hai?"
    )


def _safe_question_fallback(category: dict) -> str:
    """Fallback when question-reply LLM call fails. Uses real digest data only."""
    digest = _top_digest_item(category)
    if digest.get("title"):
        return (
            f"Is topic pe: \"{digest['title']}\" "
            f"({digest.get('source', 'recent research')}). "
            f"Kya main aapke liye full details pull karun?"
        )
    return "Yeh jaankari aapke merchant dashboard pe available hai. Kya aur kuch poochna tha?"


def _safe_clarification_fallback(merchant: dict, category: dict) -> str:
    """Fallback when clarification LLM call fails."""
    stat = _ctr_gap_sentence(merchant, category)
    if stat:
        return f"{stat} Kya aap aage badhna chahte hain? YES ya STOP batayein."
    lapsed = _lapsed_patient_count(merchant)
    if lapsed:
        return (
            f"Aapke {lapsed} patients 180 din se wapas nahi aaye — "
            f"inke liye recall campaign karna chahenge? YES ya STOP."
        )
    return "Aapka sawaal samajh gaya. Kya main aage badhun? YES ya STOP batayein."


def _safe_generic_fallback(merchant: dict, category: dict) -> str:
    """Fallback for generic continuation when LLM fails."""
    offer = _active_offer_title(merchant)
    if offer:
        return f"'{offer}' ke baare mein aage badhein? YES ya STOP batayein."
    return "Samajh gaya — aage badhun? YES ya STOP batayein."
