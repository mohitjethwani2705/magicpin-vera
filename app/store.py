"""
store.py — Thread-safe in-memory context store for Vera's bot server.

Responsibilities:
  - Store context items keyed by (scope, context_id).
  - Enforce idempotency: equal version → reject; higher version → replace.
  - Track suppression keys so the bot never re-sends a message it already sent.
  - Track per-conversation state (state machine: idle → composing → sent → replied → ended).
  - Track per-conversation turn history for anti-repetition enforcement.
  - Expose aggregate counts for /v1/healthz.

Thread safety: a single threading.RLock guards all mutable state.  Because the
FastAPI event loop runs handlers concurrently, all public mutating methods
acquire the lock for the duration of the operation — keeping state consistent
even if multiple requests arrive at the same time.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conversation state machine
# ---------------------------------------------------------------------------


class ConversationState(str, Enum):
    """
    Lifecycle of a single bot-initiated conversation.

    Transitions:
      idle       → composing   (tick received, compose() called)
      composing  → sent        (compose() returned a message, action queued)
      sent       → replied     (judge posted a reply via /v1/reply)
      replied    → sent        (bot sent another message in same conversation)
      replied    → ended       (bot chose action="end")
      sent       → ended       (bot chose action="end" immediately on next turn)
      any        → ended       (merchant opts out / hard stop)
    """

    IDLE = "idle"
    COMPOSING = "composing"
    SENT = "sent"
    REPLIED = "replied"
    ENDED = "ended"


@dataclass
class ConversationRecord:
    """All mutable state for a single conversation thread."""

    conversation_id: str
    merchant_id: str
    customer_id: Optional[str]
    trigger_id: str
    suppression_key: str
    state: ConversationState = ConversationState.IDLE

    # Ordered list of {"role": "vera"|"merchant"|"customer", "body": str, "ts": datetime}
    turns: list[dict[str, Any]] = field(default_factory=list)

    # Track bodies sent by the bot for anti-repetition checks
    sent_bodies: list[str] = field(default_factory=list)

    # Consecutive identical messages from the other side (for auto-reply detection)
    consecutive_identical_replies: int = 0
    last_reply_text: Optional[str] = None

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Context store
# ---------------------------------------------------------------------------


@dataclass
class StoredContext:
    """A single versioned context item."""

    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    stored_at: datetime


class ContextStore:
    """
    Thread-safe, in-memory store for all context items and conversation state.

    Key design decisions:
    - Contexts are keyed by (scope, context_id). A higher version replaces.
    - Suppression keys are stored with a UTC timestamp so they can be expired
      later if needed (the spec doesn't require expiry, but it's architecturally
      sound to track when a suppression was recorded).
    - Conversations are keyed by conversation_id.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()

        # (scope, context_id) -> StoredContext
        self._contexts: dict[tuple[str, str], StoredContext] = {}

        # suppression_key -> datetime when first suppressed
        self._suppression_keys: dict[str, datetime] = {}

        # conversation_id -> ConversationRecord
        self._conversations: dict[str, ConversationRecord] = {}

    # ------------------------------------------------------------------
    # Context CRUD
    # ------------------------------------------------------------------

    def put_context(
        self,
        scope: str,
        context_id: str,
        version: int,
        payload: dict[str, Any],
    ) -> tuple[bool, Optional[int]]:
        """
        Store or update a context item.

        Returns:
            (accepted, current_version)
            - (True, None)      → stored successfully
            - (False, N)        → rejected; caller already holds version N >= requested
        """
        key = (scope, context_id)
        with self._lock:
            existing = self._contexts.get(key)
            if existing is not None and existing.version >= version:
                logger.debug(
                    "Context rejected — stale version: scope=%s id=%s "
                    "requested=%d current=%d",
                    scope, context_id, version, existing.version,
                )
                return False, existing.version

            self._contexts[key] = StoredContext(
                scope=scope,
                context_id=context_id,
                version=version,
                payload=payload,
                stored_at=datetime.now(timezone.utc),
            )
            logger.info(
                "Context stored: scope=%s id=%s version=%d", scope, context_id, version
            )
            return True, None

    def get_context(self, scope: str, context_id: str) -> Optional[dict[str, Any]]:
        """Return the payload of the latest version of a context item, or None."""
        with self._lock:
            item = self._contexts.get((scope, context_id))
            return item.payload if item else None

    def get_stored_context(self, scope: str, context_id: str) -> Optional[StoredContext]:
        """Return the full StoredContext record, or None."""
        with self._lock:
            return self._contexts.get((scope, context_id))

    def iter_contexts_by_scope(self, scope: str) -> list[StoredContext]:
        """Return all stored contexts for a given scope (snapshot, not live view)."""
        with self._lock:
            return [v for (s, _), v in self._contexts.items() if s == scope]

    def count_by_scope(self) -> dict[str, int]:
        """Return item counts keyed by scope. Always returns all four keys."""
        counts: dict[str, int] = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
        with self._lock:
            for (scope, _) in self._contexts:
                if scope in counts:
                    counts[scope] += 1
        return counts

    def clear_all_contexts(self) -> None:
        """Wipe all stored contexts. Called on /v1/teardown."""
        with self._lock:
            self._contexts.clear()
            logger.info("All contexts wiped")

    # ------------------------------------------------------------------
    # Suppression key management
    # ------------------------------------------------------------------

    def is_suppressed(self, suppression_key: str) -> bool:
        """Return True if this suppression key has been recorded."""
        with self._lock:
            return suppression_key in self._suppression_keys

    def record_suppression(self, suppression_key: str) -> None:
        """Mark a suppression key as used (idempotent)."""
        with self._lock:
            if suppression_key not in self._suppression_keys:
                self._suppression_keys[suppression_key] = datetime.now(timezone.utc)
                logger.info("Suppression key recorded: %s", suppression_key)

    def clear_all_suppressions(self) -> None:
        """Wipe suppression state. Called on /v1/teardown."""
        with self._lock:
            self._suppression_keys.clear()

    # ------------------------------------------------------------------
    # Conversation state machine
    # ------------------------------------------------------------------

    def get_or_create_conversation(
        self,
        conversation_id: str,
        merchant_id: str,
        customer_id: Optional[str],
        trigger_id: str,
        suppression_key: str,
    ) -> ConversationRecord:
        """
        Return an existing conversation record, or create a new one in IDLE state.
        The caller must transition state explicitly via transition_conversation().
        """
        with self._lock:
            if conversation_id not in self._conversations:
                self._conversations[conversation_id] = ConversationRecord(
                    conversation_id=conversation_id,
                    merchant_id=merchant_id,
                    customer_id=customer_id,
                    trigger_id=trigger_id,
                    suppression_key=suppression_key,
                )
                logger.debug("Conversation created: %s", conversation_id)
            return self._conversations[conversation_id]

    def get_conversation(self, conversation_id: str) -> Optional[ConversationRecord]:
        """Return an existing conversation record or None."""
        with self._lock:
            return self._conversations.get(conversation_id)

    def transition_conversation(
        self,
        conversation_id: str,
        new_state: ConversationState,
    ) -> None:
        """Move a conversation to a new state; raises KeyError if not found."""
        with self._lock:
            record = self._conversations[conversation_id]
            old_state = record.state
            record.state = new_state
            record.updated_at = datetime.now(timezone.utc)
            logger.debug(
                "Conversation %s: %s → %s", conversation_id, old_state.value, new_state.value
            )

    def append_turn(
        self,
        conversation_id: str,
        role: str,
        body: str,
    ) -> None:
        """
        Append a turn to the conversation history.
        Also tracks consecutive identical replies for auto-reply detection.
        """
        with self._lock:
            record = self._conversations.get(conversation_id)
            if record is None:
                logger.warning("append_turn called for unknown conversation: %s", conversation_id)
                return

            record.turns.append({
                "role": role,
                "body": body,
                "ts": datetime.now(timezone.utc).isoformat(),
            })
            record.updated_at = datetime.now(timezone.utc)

            if role == "vera":
                record.sent_bodies.append(body)
            else:
                # Track consecutive identical messages for auto-reply detection
                if body == record.last_reply_text:
                    record.consecutive_identical_replies += 1
                else:
                    record.consecutive_identical_replies = 1
                    record.last_reply_text = body

    def is_body_repeated(self, conversation_id: str, body: str) -> bool:
        """Return True if the bot already sent this exact body in this conversation."""
        with self._lock:
            record = self._conversations.get(conversation_id)
            if record is None:
                return False
            return body in record.sent_bodies

    def get_consecutive_auto_replies(self, conversation_id: str) -> int:
        """Return count of consecutive identical messages from the other side."""
        with self._lock:
            record = self._conversations.get(conversation_id)
            return record.consecutive_identical_replies if record else 0

    def is_conversation_ended(self, conversation_id: str) -> bool:
        """Return True if this conversation has reached the ENDED state."""
        with self._lock:
            record = self._conversations.get(conversation_id)
            return record is not None and record.state == ConversationState.ENDED

    def all_active_conversation_ids(self) -> list[str]:
        """Return IDs of conversations not yet in ENDED state."""
        with self._lock:
            return [
                cid
                for cid, rec in self._conversations.items()
                if rec.state != ConversationState.ENDED
            ]

    def clear_all_conversations(self) -> None:
        """Wipe all conversation state. Called on /v1/teardown."""
        with self._lock:
            self._conversations.clear()
            logger.info("All conversations wiped")

    # ------------------------------------------------------------------
    # Full teardown
    # ------------------------------------------------------------------

    def teardown(self) -> None:
        """Wipe all state in one shot (contexts, suppressions, conversations)."""
        with self._lock:
            self._contexts.clear()
            self._suppression_keys.clear()
            self._conversations.clear()
            logger.info("Full teardown complete — all state wiped")


# ---------------------------------------------------------------------------
# Module-level singleton — shared across all FastAPI request handlers
# ---------------------------------------------------------------------------

store = ContextStore()
