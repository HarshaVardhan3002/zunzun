# NPU Router / Concierge — design

**Date:** 2026-07-17 · **Status:** approved (user, this date) · **Phase covered: 1**
**Prior art:** MTP speculation spec follow-ups (`2026-07-16-mtp-speculation-design.md`),
AMD GAIA router pattern. **Hardware:** Ryzen AI Max+ 395 (XDNA2 NPU ~50 TOPS, idle today).

## Problem

GLM-5.2 744B answers everything, so "hi" costs the same minutes of prompt
processing as "refactor this module". The XDNA2 NPU sits idle. Expert prefetch
is heuristic-only (usage-history pinning at load time); nothing warms the cache
between a query arriving and GLM decoding it.

## Goal

A small always-on router (Qwen3-1.7B) that:
1. answers trivial queries instantly (flagged, never impersonating GLM),
2. routes real work to GLM as **stateless JSON tasks** (GLM remembers nothing
   across tasks; the router owns the conversation),
3. fires expert prefetch for the classified intent while the user reads the
   router's acknowledgment,
4. logs (intent → expert usage) pairs from day one — the training corpus for a
   future 1M–10M dedicated expert-recommender model.

Explicitly NOT a goal: speeding up GLM decode (bandwidth-bound; see MTP spec).
Gains are first-token latency on trivial queries and cache warmth on hard ones.

## XDNA2 capability audit (2026-07, decision inputs)

- Runtimes exist; no custom port needed. FastFlowLM (NPU-first, all XDNA2 chips
  incl. Strix Halo) and AMD Lemonade ≥10.0 (OpenAI-compatible server, NPU
  backend, Linux support) both run this model class.
- Measured decode for 0.6B–1B class on XDNA2: 60–89 tok/s. Concurrent NPU
  workloads cost ~5.8% decode; NPU+GPU together ≈1.42× wall-time vs sequential.
- Ceiling: roughly two 1B-class models resident, or one 4B (256k ctx works via
  SRAM-resident KV in FastFlowLM).
- The NPU cannot help 744B decode (same 256 GB/s LPDDR5X, bandwidth-bound).
- Verdict on the two-model split (user-facing 1B + engine-facing recommender):
  right eventual shape, wrong day-1 — the recommender lacks training data, and
  at 1M–10M it runs in microseconds on CPU anyway. Phase it (below).

## Architecture (approach A, chosen over B: standalone daemon, C: heuristics-only)

Three processes:
- **`zun`** — gains a router layer; owns chat history and all user I/O.
- **Lemonade server** — hosts Qwen3-1.7B (int4 GGUF). CPU in phase 1; phase 2
  flips the same server to the NPU backend (config change, not migration —
  the reason Lemonade beats a bare llama.cpp dependency).
- **`glm.exe serve`** — existing protocol plus one new command (`\x02PREFETCH`).

### Components

**RouteGate** (pure Python, deterministic, runs before any model):
- greeting/ack/farewell patterns → local;
- hard keywords (`code`, `think`, `reason`), fenced code blocks, file paths →
  GLM, always;
- everything else → the concierge judges, biased to escalate: only clearly
  conversational queries stay local ("default to GLM", user decision).

**Concierge** (Qwen3-1.7B via Lemonade's OpenAI API):
- answers local queries; every such answer is flagged `[mini]` (mandatory);
- for GLM-bound queries, emits a JSON task envelope (below) and a one-line
  flagged acknowledgment while GLM works;
- decides task continuity (`new` vs `continue`) from conversation flow, biased
  toward `continue` (wrongly dropping context = expensive re-prefill).
  Overrides: `:new` forces fresh, `:continue` forces same, `!glm <query>`
  re-fires any query as a GLM task.

**Task envelope** (the only thing GLM ever sees):
```json
{"task_id": "t7", "action": "new" | "continue", "intent": "coding",
 "instructions": "<what to do>", "context": "<code/data the task needs>"}
```
`intent` is a profile name (`[A-Za-z0-9_-]`, ≤32 chars — same validation as
`profile.h`). The envelope is rendered into the GLM chat template as a single
user turn; `context` carries anything from the conversation the task needs,
because GLM sees no chat history.

**GLMClient** (extends zun's existing serve client):
- `task_id` → KV slot (serve supports 16). `continue` appends to that slot's
  transcript, so the existing prompt-prefix matching reuses KV — statelessness
  costs no new engine machinery. `new` takes a free slot; when none is free,
  the least-recently-used task's slot is RESET and the eviction logged.
- GLM output streams to the user unflagged — it is the product.

**`\x02PREFETCH <profile>`** (the one new engine command):
- fired by the router the moment intent is classified, before task composition;
- engine validates the name (`prof_valid_name`), loads `.coli_usage.<profile>`,
  and stages top-scored experts into idle cache capacity via the existing
  blend/pin machinery (never evicting pinned experts, never blocking
  generation); replies `PREFETCH <n_staged>`;
- missing/empty profile → `PREFETCH 0`, no error. Always best-effort.

**Intent logger** (phase-3 seed): per task, append
`(task_id, intent, per-turn expert-usage delta)` as JSONL to
`<SNAP>/.zun_intent_log`. Grows the recommender training corpus passively.

### Data flow

Hard query: user types → RouteGate: GLM → router fires `PREFETCH <intent>` →
concierge streams `[mini]` ack + composes envelope → GLMClient submits to the
task's slot → GLM streams the answer. Trivial query: RouteGate or concierge
answers in milliseconds; GLM never wakes.

## Error handling — never worse than today

| Failure | Behavior |
|---|---|
| Lemonade down / >5 s timeout | warn once, raw passthrough to GLM (= current behavior) |
| Broken JSON from concierge | one retry with the parse error shown to it; then raw passthrough |
| `PREFETCH` unknown/empty profile | `PREFETCH 0`, nothing staged, no error |
| Task slots exhausted | RESET least-recently-used task's slot, log the eviction |
| Kill switch | `ZUN_ROUTER=0` disables the layer entirely |

## Testing

- **Unit (Python):** RouteGate decision table (~20 fixture queries incl. the
  hard keywords), envelope schema validation, continue/new/eviction slot logic.
  All pure functions.
- **Unit (C):** `\x02PREFETCH` parse + staging-selection logic extracted into a
  testable header (the `kvdisk.h` pattern).
- **Integration (PROBES.md style):** scripted session measuring
  (a) trivial-query latency — target <1 s;
  (b) hard-query handoff overhead vs direct GLM — target <5%;
  (c) cold coding-task expert hit rate with vs without `PREFETCH` — the number
  that proves the idea.

## Phases

1. **This spec's implementation scope:** router on CPU via Lemonade,
   `\x02PREFETCH`, intent logging, flagging, overrides, kill switch.
2. Flip the concierge onto the XDNA2 via Lemonade's NPU backend; re-measure
   latency and power. Config change; own (small) spec.
3. Train the 1M–10M expert recommender on the logged corpus; only then decide
   whether it earns residency as a second NPU model. Own spec.

## User decisions on record

- Router answers trivial queries itself; flagging is mandatory (2026-07-17).
- "Hi/bye/basic" only from the 1B; `code`/`think`/`reason` and all real work
  from GLM; ambiguous middle defaults to GLM.
- GLM is stateless per task; the 1B decides `new` vs `continue`; user override.
- Router model: Qwen3-1.7B.
- Two-model NPU split deferred to phase 3 pending logged data (capability
  audit accepted).
