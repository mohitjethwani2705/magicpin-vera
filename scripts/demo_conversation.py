"""Quick demo: shows what Vera sends to a merchant and handles a reply."""
import sys, os, json, requests

sys.stdout.reconfigure(encoding="utf-8")

BASE = "http://localhost:8000"

# ── 1. Push contexts ──────────────────────────────────────────────────────────
category = {
    "category_id": "restaurant",
    "name": "Restaurant",
    "voice": "warm, conversational",
    "language_preference": "hi-en",
}

merchant = {
    "merchant_id": "demo_merchant_01",
    "business_name": "Sharma Ji Ka Dhaba",
    "owner_name": "Ramesh Sharma",
    "category_id": "restaurant",
    "category_slug": "restaurant",
    "city": "Delhi",
    "monthly_orders": 180,
    "avg_rating": 3.8,
    "active_offers": [],
    "last_active_days_ago": 6,
    "gmv_trend": "declining",
}

trigger = {
    "id": "win_back_lapsed",
    "kind": "action",
    "why_now": "Orders dropped 40% in last 2 weeks",
    "offer_hook": "Free premium listing for 30 days",
    "urgency": "high",
    "merchant_id": "demo_merchant_01",  # required so tick can resolve merchant
}

print("=" * 60)
print("DEMO: Vera talking to Sharma Ji Ka Dhaba")
print("=" * 60)

# Push category
import time as _time
_v = int(_time.time())  # unique version each run to avoid 409

r = requests.post(f"{BASE}/v1/context", json={"scope": "category", "context_id": "restaurant", "version": _v, "payload": category, "delivered_at": "2026-07-07T10:00:00Z"})
print(f"[context] category -> {r.status_code}")

# Push merchant
r = requests.post(f"{BASE}/v1/context", json={"scope": "merchant", "context_id": "demo_merchant_01", "version": _v, "payload": merchant, "delivered_at": "2026-07-07T10:00:00Z"})
print(f"[context] merchant -> {r.status_code}")

# Push trigger
r = requests.post(f"{BASE}/v1/context", json={"scope": "trigger", "context_id": "win_back_lapsed", "version": _v, "payload": trigger, "delivered_at": "2026-07-07T10:00:00Z"})
print(f"[context] trigger  -> {r.status_code}")

# ── 2. Tick: ask Vera to compose the opening message ─────────────────────────
tick_payload = {
    "now": "2026-07-07T10:00:00Z",
    "available_triggers": ["win_back_lapsed"],
}

print("\n--- Tick (compose opening message) ---")
r = requests.post(f"{BASE}/v1/tick", json=tick_payload)
tick = r.json()
print(f"Status: {r.status_code}")
actions = tick.get("actions", [])
if not actions:
    print("No actions returned. Check server logs.")
    exit(1)
action = actions[0]
conv_id = action.get("conversation_id", "conv_demo_01")
print(f"\nVera's message to Sharma Ji:\n")
print(f"  {action.get('body', '')}")
print(f"\nCTA: {action.get('cta', '')}")
print(f"Rationale: {action.get('rationale', '')}")

# ── 3. Merchant replies positively ───────────────────────────────────────────
merchant_reply = "Haan bhai, batao kya hai offer"

print("\n--- Merchant replies ---")
print(f"  Merchant: \"{merchant_reply}\"")

reply_payload = {
    "conversation_id": conv_id,
    "merchant_id": "demo_merchant_01",
    "from_role": "merchant",
    "message": merchant_reply,
}

r = requests.post(f"{BASE}/v1/reply", json=reply_payload)
reply = r.json()
print(f"\nVera's follow-up:\n")
print(f"  {reply.get('body', '')}")
print(f"\nAction: {reply.get('action', 'continue')}")

# ── 4. Merchant says no ───────────────────────────────────────────────────────
merchant_reply2 = "Nahi chahiye, band karo yeh sab"

print("\n--- Merchant declines ---")
print(f"  Merchant: \"{merchant_reply2}\"")

reply_payload2 = {
    "conversation_id": conv_id,
    "merchant_id": "demo_merchant_01",
    "from_role": "merchant",
    "message": merchant_reply2,
}

r = requests.post(f"{BASE}/v1/reply", json=reply_payload2)
reply2 = r.json()
print(f"\nVera's exit message:\n")
print(f"  {reply2.get('body', '')}")
print(f"\nAction: {reply2.get('action', 'continue')}")

print("\n" + "=" * 60)
print("Demo complete!")
