"""
models.py — Pydantic request/response models for Vera's FastAPI server.

Every model mirrors the exact JSON contract defined in challenge-testing-brief.md §2
and validated against the api-call-examples.md examples.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Shared / utility
# ---------------------------------------------------------------------------

VALID_SCOPES = frozenset({"category", "merchant", "customer", "trigger"})


# ---------------------------------------------------------------------------
# POST /v1/context
# ---------------------------------------------------------------------------


class ContextPushRequest(BaseModel):
    """Payload the judge sends to push a context item to the bot."""

    scope: str = Field(..., description="One of: category | merchant | customer | trigger")
    context_id: str = Field(..., min_length=1, description="Unique identifier for this context item")
    version: int = Field(..., ge=1, description="Monotonically increasing version counter")
    payload: dict[str, Any] = Field(..., description="The full context object for this scope")
    delivered_at: str = Field(..., description="ISO-8601 UTC timestamp of when the judge sent this")

    @field_validator("scope")
    @classmethod
    def scope_must_be_valid(cls, v: str) -> str:
        if v not in VALID_SCOPES:
            raise ValueError(f"scope must be one of {sorted(VALID_SCOPES)}, got '{v}'")
        return v


class ContextAcceptedResponse(BaseModel):
    """Returned when the bot stores the context successfully."""

    accepted: Literal[True] = True
    ack_id: str
    stored_at: str  # ISO-8601 UTC


class ContextRejectedStaleResponse(BaseModel):
    """Returned when the bot already holds an equal or higher version."""

    accepted: Literal[False] = False
    reason: Literal["stale_version"] = "stale_version"
    current_version: int


class ContextRejectedInvalidResponse(BaseModel):
    """Returned when the request payload is structurally invalid."""

    accepted: Literal[False] = False
    reason: Literal["invalid_scope"] = "invalid_scope"
    details: str


# ---------------------------------------------------------------------------
# POST /v1/tick
# ---------------------------------------------------------------------------


class TickRequest(BaseModel):
    """Periodic wake-up signal; bot inspects state and may initiate messages."""

    now: str = Field(..., description="Current simulated time in ISO-8601 UTC")
    available_triggers: list[str] = Field(
        default_factory=list,
        description="Trigger context_ids the judge considers active right now",
    )


class TickAction(BaseModel):
    """A single proactive message the bot wants to send this tick."""

    conversation_id: str = Field(..., description="Unique conversation identifier (new UUID per new convo)")
    merchant_id: str = Field(..., description="Target merchant")
    customer_id: Optional[str] = Field(None, description="Target customer if customer-scoped; else null")
    send_as: str = Field(..., description="'vera' for bot-to-merchant or 'merchant_on_behalf' for merchant-to-customer")
    trigger_id: str = Field(..., description="The trigger context_id that motivated this action")
    body: str = Field(..., min_length=1, description="The actual message body — must not be empty")
    cta: str = Field(..., description="Call-to-action type: open_ended | binary_yes_no | multi_choice_slot | etc.")
    suppression_key: str = Field(..., description="Key used to deduplicate outbound messages")
    rationale: str = Field(..., min_length=1, description="Explanation of why this message is being sent now")


class TickResponse(BaseModel):
    """Bot's reply to a tick — zero or more proactive actions."""

    actions: list[TickAction] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# POST /v1/reply
# ---------------------------------------------------------------------------


class ReplyRequest(BaseModel):
    """Judge delivers a merchant/customer reply into an existing conversation."""

    conversation_id: str = Field(..., description="Existing conversation being continued")
    merchant_id: Optional[str] = Field(None)
    customer_id: Optional[str] = Field(None)
    from_role: str = Field(
        ...,
        description="One of: 'merchant', 'customer', 'judge'",
    )
    message: str = Field(..., description="The reply text")
    received_at: Optional[str] = Field(
        None,
        description="ISO-8601 UTC timestamp (optional — judge may omit)",
    )
    turn_number: Optional[int] = Field(
        None,
        ge=1,
        description="Turn counter; starts at 2 (optional — judge may omit)",
    )

    @field_validator("from_role")
    @classmethod
    def from_role_must_be_valid(cls, v: str) -> str:
        allowed = {"merchant", "customer", "judge"}
        if v not in allowed:
            raise ValueError(f"from_role must be one of {sorted(allowed)}, got '{v}'")
        return v


class SendReplyResponse(BaseModel):
    """Bot continues the conversation with a new outbound message."""

    action: Literal["send"] = "send"
    body: str = Field(..., min_length=1)
    cta: str
    rationale: str


class WaitReplyResponse(BaseModel):
    """Bot asks to be paused for a specified number of seconds."""

    action: Literal["wait"] = "wait"
    wait_seconds: int = Field(..., ge=0)
    rationale: str


class EndReplyResponse(BaseModel):
    """Bot closes the conversation gracefully."""

    action: Literal["end"] = "end"
    rationale: str


# Union type used in controller — the actual response can be any of these three
ReplyResponse = SendReplyResponse | WaitReplyResponse | EndReplyResponse


# ---------------------------------------------------------------------------
# GET /v1/healthz
# ---------------------------------------------------------------------------


class ContextCounts(BaseModel):
    """Count of stored items per scope."""

    category: int = 0
    merchant: int = 0
    customer: int = 0
    trigger: int = 0


class HealthzResponse(BaseModel):
    """Liveness probe response."""

    status: Literal["ok"] = "ok"
    uptime_seconds: int
    contexts_loaded: ContextCounts


# ---------------------------------------------------------------------------
# GET /v1/metadata
# ---------------------------------------------------------------------------


class MetadataResponse(BaseModel):
    """Bot identity card returned to the judge during warmup."""

    team_name: str
    team_members: list[str]
    model: str
    approach: str
    contact_email: str
    version: str
    submitted_at: str


# ---------------------------------------------------------------------------
# POST /v1/teardown  (optional — spec §11)
# ---------------------------------------------------------------------------


class TeardownResponse(BaseModel):
    """Acknowledgment after state wipe."""

    wiped: bool = True
    message: str = "State cleared"


# ---------------------------------------------------------------------------
# Generic error envelope
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Standard error shape returned by the global exception handler."""

    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None
