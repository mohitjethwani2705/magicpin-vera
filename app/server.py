"""
server.py — Production-grade FastAPI server for Vera, the magicpin AI challenge bot.

Architecture:
  - Five endpoints matching the exact contract in challenge-testing-brief.md §2.
  - In-memory context store (store.py) with thread-safe operations.
  - Pydantic models (models.py) for all I/O shapes.
  - Placeholder imports for bot.py (compose) and conversation_handlers.py (handle_reply)
    — those modules are being developed in parallel and are called from the tick
    and reply handlers respectively.
  - Request-ID middleware for end-to-end traceability.
  - Global exception handler returning consistent error JSON.
  - CORS enabled (judge harness may run cross-origin).
  - Structured logging to stdout.

Run:
    uvicorn server:app --host 0.0.0.0 --port 8000 --log-level info

Or with auto-reload during development:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload

Port is read from the PORT environment variable (default: 8000).
"""

from __future__ import annotations

# Load .env FIRST — config.py parses the .env file and sets os.environ entries
# before any downstream module (groq client, bot.py, etc.) reads them.
try:
    import app.config as _config  # noqa: F401  — side-effect import for .env loading
    from app.config import settings as _settings  # validates GROQ_API_KEY is set
except Exception as _cfg_err:  # pragma: no cover
    import logging as _early_log
    _early_log.warning("config.py could not load (continuing anyway): %s", _cfg_err)

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional, Union

import uvicorn
from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.models import (
    ContextAcceptedResponse,
    ContextCounts,
    ContextPushRequest,
    ContextRejectedInvalidResponse,
    ContextRejectedStaleResponse,
    EndReplyResponse,
    ErrorResponse,
    HealthzResponse,
    MetadataResponse,
    ReplyRequest,
    SendReplyResponse,
    TeardownResponse,
    TickAction,
    TickRequest,
    TickResponse,
    WaitReplyResponse,
    VALID_SCOPES,
)
from app.store import ConversationState, store

# ---------------------------------------------------------------------------
# Placeholder imports — replace with real implementations once available
# ---------------------------------------------------------------------------
try:
    from app.bot import compose  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    async def compose(  # type: ignore[misc]
        trigger_payload: dict[str, Any],
        merchant_payload: dict[str, Any],
        category_payload: dict[str, Any],
        customer_payload: Optional[dict[str, Any]],
        conversation_id: str,
        simulated_now: str,
    ) -> Optional[dict[str, Any]]:
        """
        Stub: returns None (no action) until bot.py is implemented.

        The real implementation should return a dict with keys:
          merchant_id, customer_id, send_as, trigger_id, template_name,
          template_params, body, cta, suppression_key, rationale
        or None to skip sending for this trigger.
        """
        return None

try:
    from app.conversation_handlers import handle_reply  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    async def handle_reply(  # type: ignore[misc]
        reply: ReplyRequest,
        conversation_history: list[dict[str, Any]],
        merchant_payload: Optional[dict[str, Any]],
        category_payload: Optional[dict[str, Any]],
        customer_payload: Optional[dict[str, Any]],
        consecutive_identical_replies: int,
    ) -> Union[SendReplyResponse, WaitReplyResponse, EndReplyResponse]:
        """
        Stub: gracefully ends all conversations until conversation_handlers.py is implemented.

        The real implementation must:
          - Detect auto-replies and return WaitReplyResponse or EndReplyResponse.
          - Handle intent transitions (merchant says 'let's do it').
          - Handle hostile / off-topic messages.
          - Return SendReplyResponse for normal continuation.
        """
        return EndReplyResponse(rationale="[stub] conversation_handlers.py not yet implemented")


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s request_id=%(request_id)s %(message)s"
    if False  # structured formatter set below
    else "%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("vera.server")


# ---------------------------------------------------------------------------
# App bootstrap
# ---------------------------------------------------------------------------

START_TIME: float = time.monotonic()

app = FastAPI(
    title="Vera — magicpin AI Challenge Bot",
    description=(
        "Stateful bot server implementing the 5-endpoint contract defined in "
        "challenge-testing-brief.md. Receives context pushes from the judge harness, "
        "proactively initiates conversations on /v1/tick, and continues multi-turn "
        "conversations via /v1/reply."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — permissive for the challenge environment; tighten for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware: attach a Request-ID to every request for traceability
# ---------------------------------------------------------------------------


@app.middleware("http")
async def request_id_middleware(request: Request, call_next: Any) -> Response:
    """
    Attach a unique request_id to every inbound request.

    The ID is sourced from the X-Request-ID header (if the judge sends one)
    or generated as a UUID4. It is echoed back in the X-Request-ID response
    header and stored on request.state so handlers can log it.
    """
    request_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:12]}"
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all handler that returns a consistent error envelope.
    Never exposes Python tracebacks to the caller.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception(
        "Unhandled exception on %s %s request_id=%s",
        request.method,
        request.url.path,
        request_id,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="internal_server_error",
            detail="An unexpected error occurred. Check server logs.",
            request_id=request_id,
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    """Return current UTC time as an ISO-8601 string with trailing Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _uptime_seconds() -> int:
    """Return integer seconds since the server process started."""
    return int(time.monotonic() - START_TIME)


def _log(request: Request, msg: str, level: str = "info") -> None:
    """Log a message annotated with the current request_id."""
    request_id = getattr(request.state, "request_id", "unknown")
    getattr(logger, level)("[%s] %s", request_id, msg)


# ---------------------------------------------------------------------------
# Endpoint: GET /v1/healthz
# ---------------------------------------------------------------------------


@app.get(
    "/v1/healthz",
    response_model=HealthzResponse,
    summary="Liveness probe — polled every 60s by the judge",
    tags=["operations"],
)
async def healthz(request: Request) -> HealthzResponse:
    """
    Return server liveness status and counts of stored context items per scope.

    The judge checks this before the test window and every 60 s thereafter.
    Three consecutive non-200 responses disqualify the bot for that test slot.
    """
    counts = store.count_by_scope()
    _log(request, f"healthz — uptime={_uptime_seconds()}s contexts={counts}")
    return HealthzResponse(
        status="ok",
        uptime_seconds=_uptime_seconds(),
        contexts_loaded=ContextCounts(**counts),
    )


# ---------------------------------------------------------------------------
# Endpoint: GET /v1/metadata
# ---------------------------------------------------------------------------


@app.get(
    "/v1/metadata",
    response_model=MetadataResponse,
    summary="Bot identity card",
    tags=["operations"],
)
async def metadata(request: Request) -> MetadataResponse:
    """
    Return static bot identity information.

    The judge reads this during warmup (Phase 1) to verify the bot is the
    expected team's submission and to record model + approach for the scorecard.
    """
    _log(request, "metadata requested")
    return MetadataResponse(
        team_name="Vera AI",
        team_members=["Vera AI"],
        model="llama-3.1-8b-instant via Groq API",
        approach=(
            "4-context composer (category + merchant + trigger + customer) with "
            "trigger-aware prompt engineering, suppression deduplication, "
            "auto-reply detection, and intent-transition handling."
        ),
        contact_email="team@vera-ai.com",
        version="1.0.0",
        submitted_at="2026-07-07T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# Endpoint: POST /v1/context
# ---------------------------------------------------------------------------


@app.post(
    "/v1/context",
    summary="Receive a context push from the judge",
    tags=["context"],
    status_code=status.HTTP_200_OK,
)
async def push_context(
    body: ContextPushRequest,
    request: Request,
) -> JSONResponse:
    """
    Store or update a context item.

    Idempotency rules (per spec §2.1):
    - Same (context_id, version) re-sent → 409 stale_version (no-op).
    - Higher version for same context_id → replaces atomically.
    - Lower version after a higher one was stored → 409 stale_version.

    Payload size is capped at 500 KB by the judge harness; we do not enforce
    that limit here but do validate the scope field.
    """
    _log(request, f"context push: scope={body.scope} id={body.context_id} v={body.version}")

    # scope is validated by the Pydantic model; belt-and-suspenders check here
    # gives us a chance to return a 400 with a helpful message if somehow bypassed
    if body.scope not in VALID_SCOPES:
        _log(request, f"invalid scope '{body.scope}'", level="warning")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ContextRejectedInvalidResponse(
                details=f"scope must be one of {sorted(VALID_SCOPES)}, got '{body.scope}'"
            ).model_dump(),
        )

    accepted, current_version = store.put_context(
        scope=body.scope,
        context_id=body.context_id,
        version=body.version,
        payload=body.payload,
    )

    if not accepted:
        # Equal or higher version already stored — 409 per spec
        _log(
            request,
            f"context rejected stale: id={body.context_id} "
            f"requested={body.version} current={current_version}",
            level="warning",
        )
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content=ContextRejectedStaleResponse(
                current_version=current_version  # type: ignore[arg-type]
            ).model_dump(),
        )

    ack_id = f"ack_{body.context_id}_v{body.version}"
    stored_at = _utc_now_iso()
    _log(request, f"context accepted: ack_id={ack_id} stored_at={stored_at}")
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=ContextAcceptedResponse(ack_id=ack_id, stored_at=stored_at).model_dump(),
    )


# ---------------------------------------------------------------------------
# Endpoint: POST /v1/tick
# ---------------------------------------------------------------------------

# Maximum real-wall-clock seconds before we abort and return empty actions.
# The spec gives 30s; we use 25s to leave 5s of buffer for network.
_TICK_TIMEOUT_SECONDS = 25.0

# Hard cap from spec §5 — max actions returned per tick
_TICK_ACTION_CAP = 20


@app.post(
    "/v1/tick",
    response_model=TickResponse,
    summary="Periodic wake-up; bot decides whether to send proactive messages",
    tags=["conversation"],
)
async def tick(body: TickRequest, request: Request) -> TickResponse:
    """
    The judge calls this every N simulated minutes.  The bot:
      1. Iterates available_triggers.
      2. Checks suppression keys — skips triggers already suppressed.
      3. Resolves merchant + category + optional customer context from the store.
      4. Calls compose() to build a message (may return None → skip).
      5. Records the suppression key and conversation state.
      6. Returns up to _TICK_ACTION_CAP actions within _TICK_TIMEOUT_SECONDS.

    If composition takes too long the handler returns an empty actions list
    rather than timing out — the spec prefers an empty response over a timeout
    penalty.
    """
    _log(request, f"tick: now={body.now} triggers={body.available_triggers}")

    actions: list[TickAction] = []

    async def _process_triggers() -> None:
        """Inner coroutine so we can wrap it with asyncio.wait_for."""
        for trigger_id in body.available_triggers:
            if len(actions) >= _TICK_ACTION_CAP:
                logger.warning("tick: action cap (%d) reached — skipping remaining triggers", _TICK_ACTION_CAP)
                break

            # Resolve trigger context
            trigger_payload = store.get_context("trigger", trigger_id)
            if trigger_payload is None:
                logger.debug("tick: trigger '%s' not in store — skipping", trigger_id)
                continue

            suppression_key: str = trigger_payload.get("suppression_key", "")

            # Check suppression — don't re-send a message we already sent
            if suppression_key and store.is_suppressed(suppression_key):
                logger.info("tick: suppressed — trigger=%s key=%s", trigger_id, suppression_key)
                continue

            # Resolve merchant context
            merchant_id: Optional[str] = trigger_payload.get("merchant_id")
            if not merchant_id:
                logger.warning("tick: trigger '%s' has no merchant_id — skipping", trigger_id)
                continue

            merchant_payload = store.get_context("merchant", merchant_id)
            if merchant_payload is None:
                logger.warning("tick: merchant '%s' not in store — skipping", merchant_id)
                continue

            # Resolve category context
            category_slug: str = merchant_payload.get("category_slug", "")
            category_payload = store.get_context("category", category_slug)
            if category_payload is None:
                logger.warning(
                    "tick: category '%s' not in store for merchant '%s' — skipping",
                    category_slug, merchant_id,
                )
                continue

            # Resolve optional customer context (for customer-scoped triggers)
            customer_id: Optional[str] = trigger_payload.get("customer_id")
            customer_payload: Optional[dict[str, Any]] = None
            if customer_id:
                customer_payload = store.get_context("customer", customer_id)
                if customer_payload is None:
                    logger.warning(
                        "tick: customer '%s' not in store — composing without customer context",
                        customer_id,
                    )

            # Generate a stable conversation_id for this trigger + merchant pair
            conversation_id = f"conv_{merchant_id}_{trigger_id}"

            # If this conversation already exists and is not ended, skip to avoid duplicate sends
            existing = store.get_conversation(conversation_id)
            if existing is not None and existing.state != ConversationState.ENDED:
                logger.debug(
                    "tick: conversation '%s' already active (state=%s) — skipping",
                    conversation_id, existing.state.value,
                )
                continue

            # Call compose() — sync function, run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            composed: Optional[dict[str, Any]] = await loop.run_in_executor(
                None,
                lambda: compose(
                    category=category_payload,
                    merchant=merchant_payload,
                    trigger=trigger_payload,
                    customer=customer_payload,
                )
            )

            if composed is None:
                logger.info(
                    "tick: compose() returned None for trigger=%s merchant=%s — skipping",
                    trigger_id, merchant_id,
                )
                continue

            # Validate compose() result has mandatory fields before building TickAction
            required_keys = {"body", "cta", "rationale"}
            missing = required_keys - composed.keys()
            if missing:
                logger.error(
                    "tick: compose() result missing keys %s for trigger=%s — skipping",
                    missing, trigger_id,
                )
                continue

            body_text: str = composed["body"]
            if not body_text.strip():
                logger.error("tick: compose() returned empty body for trigger=%s — skipping", trigger_id)
                continue

            # Record conversation state
            store.get_or_create_conversation(
                conversation_id=conversation_id,
                merchant_id=merchant_id,
                customer_id=customer_id,
                trigger_id=trigger_id,
                suppression_key=suppression_key,
            )
            store.transition_conversation(conversation_id, ConversationState.COMPOSING)
            store.append_turn(conversation_id, "vera", body_text)
            store.transition_conversation(conversation_id, ConversationState.SENT)

            # Record suppression key so we don't send this again
            if suppression_key:
                store.record_suppression(suppression_key)

            actions.append(
                TickAction(
                    conversation_id=conversation_id,
                    merchant_id=merchant_id,
                    customer_id=customer_id,
                    send_as=composed.get("send_as", "vera"),
                    trigger_id=trigger_id,
                    body=body_text,
                    cta=composed.get("cta", ""),
                    suppression_key=suppression_key,
                    rationale=composed.get("rationale", ""),
                )
            )
            logger.info(
                "tick: action queued — conversation=%s merchant=%s trigger=%s",
                conversation_id, merchant_id, trigger_id,
            )

    try:
        await asyncio.wait_for(_process_triggers(), timeout=_TICK_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        # Return whatever actions were built before the timeout — do not raise
        logger.warning(
            "tick: timed out after %.1fs — returning %d partial actions",
            _TICK_TIMEOUT_SECONDS, len(actions),
        )

    _log(request, f"tick complete — {len(actions)} action(s) returning")
    return TickResponse(actions=actions)


# ---------------------------------------------------------------------------
# Endpoint: POST /v1/reply
# ---------------------------------------------------------------------------

_REPLY_TIMEOUT_SECONDS = 28.0  # 30s budget from spec; 2s buffer


@app.post(
    "/v1/reply",
    summary="Judge delivers a merchant/customer reply; bot responds with next move",
    tags=["conversation"],
)
async def reply(body: ReplyRequest, request: Request) -> JSONResponse:
    """
    Continue a multi-turn conversation.

    The handler:
      1. Looks up the conversation record.
      2. Appends the incoming reply to the turn history.
      3. Resolves all context payloads for the conversation.
      4. Calls handle_reply() from conversation_handlers.py.
      5. Appends the bot's response to the turn history.
      6. Transitions conversation state based on the chosen action.
      7. Returns the response within _REPLY_TIMEOUT_SECONDS.

    State transitions:
      - action="send"  → REPLIED → (next tick or reply continues)
      - action="wait"  → SENT (still open, bot is backing off)
      - action="end"   → ENDED (conversation closed; suppression key stays)
    """
    _log(
        request,
        f"reply: conversation={body.conversation_id} "
        f"from={body.from_role} turn={body.turn_number}",
    )

    # Retrieve conversation record
    conversation = store.get_conversation(body.conversation_id)

    if conversation is None:
        # Unknown conversation — the judge may send replies for conversations we
        # didn't initiate (edge case).  Create a minimal record to handle it.
        logger.warning(
            "reply: unknown conversation '%s' — creating on-the-fly record",
            body.conversation_id,
        )
        conversation = store.get_or_create_conversation(
            conversation_id=body.conversation_id,
            merchant_id=body.merchant_id or "unknown",
            customer_id=body.customer_id,
            trigger_id="unknown",
            suppression_key="",
        )

    if conversation.state == ConversationState.ENDED:
        # Conversation already ended — politely refuse to continue
        logger.warning(
            "reply: conversation '%s' is already ENDED — returning end",
            body.conversation_id,
        )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=EndReplyResponse(
                rationale="Conversation already ended; not continuing."
            ).model_dump(),
        )

    # Append the incoming turn to history
    store.append_turn(body.conversation_id, body.from_role, body.message)
    store.transition_conversation(body.conversation_id, ConversationState.REPLIED)

    # Resolve context payloads for the handler
    merchant_payload = store.get_context("merchant", conversation.merchant_id)
    category_slug = (merchant_payload or {}).get("category_slug", "")
    category_payload = store.get_context("category", category_slug) if category_slug else None
    customer_payload = (
        store.get_context("customer", conversation.customer_id)
        if conversation.customer_id
        else None
    )

    consecutive_identical = store.get_consecutive_auto_replies(body.conversation_id)

    async def _call_handle_reply() -> dict:
        from app.conversation_handlers import ConversationState as CHState
        # Build a ConversationState for the handler
        ch_state = CHState(merchant_id=conversation.merchant_id)
        def _normalize_turn(t: Any) -> dict:
            if not isinstance(t, dict):
                return {"role": t.role, "content": t.content, "timestamp": t.timestamp}
            # store.py saves as {role, body, ts} — normalize to {role, content, timestamp}
            return {
                "role": t.get("role", ""),
                "content": t.get("content") or t.get("body", ""),
                "timestamp": t.get("timestamp") or t.get("ts", ""),
            }
        ch_state.turns = [_normalize_turn(t) for t in conversation.turns]
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: handle_reply(
                merchant_message=body.message,
                conversation_state=ch_state,
                merchant=merchant_payload or {},
                category=category_payload or {},
                customer=customer_payload,
            )
        )

    try:
        result = await asyncio.wait_for(_call_handle_reply(), timeout=_REPLY_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.error(
            "reply: handle_reply timed out for conversation=%s", body.conversation_id
        )
        result = WaitReplyResponse(
            wait_seconds=300,
            rationale="Internal timeout composing reply — backing off 5 minutes.",
        )

    # result is a dict from handle_reply: {action, response_body, cta, rationale, should_end}
    action = result.get("action", "end")
    response_body = result.get("response_body", "")
    rationale = result.get("rationale", "")

    if action == "send" and response_body:
        if store.is_body_repeated(body.conversation_id, response_body):
            logger.warning("reply: anti-repetition guard triggered conversation=%s", body.conversation_id)
            store.transition_conversation(body.conversation_id, ConversationState.ENDED)
            reply_result = {"action": "end", "rationale": "Prevented duplicate message."}
        else:
            store.append_turn(body.conversation_id, "vera", response_body)
            store.transition_conversation(body.conversation_id, ConversationState.SENT)
            reply_result = {"action": "send", "body": response_body, "cta": result.get("cta", ""), "rationale": rationale}
    elif action == "wait":
        store.transition_conversation(body.conversation_id, ConversationState.SENT)
        reply_result = {"action": "wait", "wait_seconds": 120, "rationale": rationale}
    else:
        store.transition_conversation(body.conversation_id, ConversationState.ENDED)
        reply_result = {"action": "end", "rationale": rationale}

    _log(request, f"reply complete: action={action} conversation={body.conversation_id}")
    return JSONResponse(status_code=status.HTTP_200_OK, content=reply_result)


# ---------------------------------------------------------------------------
# Endpoint: POST /v1/teardown  (optional per spec §11)
# ---------------------------------------------------------------------------


@app.post(
    "/v1/teardown",
    response_model=TeardownResponse,
    summary="Wipe all state at end of test (optional endpoint)",
    tags=["operations"],
)
async def teardown(request: Request) -> TeardownResponse:
    """
    Wipe all in-memory state: contexts, suppression keys, conversations.

    The spec marks this as optional — magicpin may or may not call it.
    Implementing it is good hygiene and avoids stale state if the server
    is reused across multiple test runs.
    """
    _log(request, "teardown requested — wiping all state")
    store.teardown()
    return TeardownResponse(wiped=True, message="All state cleared successfully")


# ---------------------------------------------------------------------------
# Startup / shutdown events
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def on_startup() -> None:
    """Log server startup with initial context counts."""
    counts = store.count_by_scope()
    logger.info(
        "Vera bot server started — version=1.0.0 contexts=%s",
        counts,
    )


@app.on_event("shutdown")
async def on_shutdown() -> None:
    """Log graceful shutdown."""
    logger.info("Vera bot server shutting down — uptime=%ds", _uptime_seconds())


# ---------------------------------------------------------------------------
# Entrypoint for direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    _port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=_port,
        log_level="info",
        access_log=True,
    )
