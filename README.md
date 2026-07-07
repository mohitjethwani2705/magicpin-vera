# Vera AI — magicpin AI Challenge Submission

## Approach

**4-context composer** — every message is built from all four layers simultaneously:
`category + merchant + trigger + customer → WhatsApp message`

**Model**: `llama-3.1-8b-instant` via Groq API (temperature=0 for determinism, <2s median latency).

**Trigger-aware prompt routing**: the system prompt varies by trigger kind. Research-digest triggers get a clinical-peer frame; perf-spike triggers lead with loss-aversion framing; recall-due triggers use the customer's language preference and relationship history.

**Suppression deduplication**: the server tracks `suppression_key` per conversation so the same trigger is never sent twice in a session.

**Auto-reply detection**: the reply handler checks for consecutive identical messages (≥3 repeats = auto-reply). On detection it backs off once with a reframe, then exits gracefully — no more than 2 wasted turns.

**Intent-transition handling**: if a merchant reply contains acceptance signals ("yes", "let's do it", "go ahead") the handler switches to action mode immediately rather than asking another qualifying question.

**Score self-evaluation**: `estimate_score()` runs a heuristic pass over the composed body before returning — checking for specificity anchors (numbers, dates, source citations), CTA shape (binary vs. open-ended), and language match. Flags are logged; compositions that fail hard checks are retried once.

---

## How to Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Groq API key
cp .env.example .env
echo "GROQ_API_KEY=your_key_here" >> .env

# 3. Start the server (defaults to port 8000)
uvicorn server:app --host 0.0.0.0 --port 8000 --reload

# 4. Generate submission.jsonl (30 pairs)
python generate_submission.py
```

---

## How to Run the Judge Simulator

```bash
export BOT_URL=http://localhost:8000
python judge_simulator.py
```

The simulator runs Phase 1 (warmup), a short Phase 2 (3 tick cycles), and one replay scenario. Each scenario prints the judge's prompts, bot responses, and a mock score per dimension.

---

## Tradeoffs Made

| Decision | Tradeoff |
|---|---|
| `llama-3.1-8b-instant` via Groq | Fast (sub-2s) but lower ceiling than frontier models. Chosen for 30s latency budget compliance. |
| Heuristic `estimate_score()` | Avoids a second LLM call per composition. Less accurate than an LLM judge but adds zero latency. |
| In-memory state only | Zero setup friction, but state is lost on restart. Acceptable for a 60-min test window. |
| Single system prompt + trigger-kind branches | Simpler than per-trigger fine-tuned prompts; easier to debug and extend. |
| No retrieval / embedding layer | The category digest fits in the context window at ~8k tokens total. RAG overhead not justified. |

---

## What Additional Context Would Have Helped Most

1. **Real merchant reply corpora** — knowing how merchants in each category actually phrase acceptance vs. deflection would sharpen intent detection significantly.
2. **Suppression history per merchant** — knowing what Vera already sent this week would prevent topic repetition without needing to infer it from `conversation_history`.
3. **Seasonal calendar** — exact festival dates for the test window would let the bot time festival triggers more precisely.
4. **CTR by message style** — empirical data on which compulsion levers actually move CTR for each category (e.g., loss-aversion vs. curiosity for dentists) would let us rank lever selection rather than using a fixed heuristic.
