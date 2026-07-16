# MTP Speculation: Bug Gate, then Acceptance Recovery

**Date:** 2026-07-16
**Status:** Approved (brainstorm complete)
**Target:** Zunzun engine (`c/glm.c`) on Strix Halo, GLM-5.2 744B int4

## Why

Speculation is the only optimization that multiplies tokens per weight-read, so it
keeps paying at every tier of the memory hierarchy (NVMe today, DRAM once hit rate
is solved). The engine already has a lossless speculation stack — native MTP head,
n-gram fallback, grammar drafts, batch-union verification — but it underperforms:

- **Measured:** 28% MTP acceptance, 1.85 tokens/forward (2026-07-15 baseline run).
- **Reference:** DeepSeek-V3's equivalent MTP head reports ~85–90% acceptance.
- **Live uncertainty in the code:** `MTP_PRENORM`, `MTP_SWAP`, `MTP_DEBUG` env
  experiments toggle norm placement and the concat order of `[embedding, hidden]`
  into `eh_proj` — the head's wiring was never settled against the reference.
- **Quality bug:** recent chat sessions produce dropped tokens in generated code
  (`this.x -= ;`, `score = ;`, `shortest_path.get(, ...)`) — plausibly in the same
  speculation/emit/rejection-sampling neighborhood.

If wiring fixes lift acceptance to ~75%, tokens/forward goes 1.85 → ~3.5 and warm
chat throughput roughly doubles (0.49 → ~1 tok/s) with zero extra bytes read.

## Goals

1. **Correctness first:** find and fix the dropped-token bug; prove output is
   trustworthy before optimizing the path that produces it.
2. **Acceptance ≥60%** on the native MTP head (or a documented finding of why 28%
   is the real ceiling).
3. **Tokens/forward ≥3** at the tuned default draft depth; warm chat tok/s
   measurably improved in a real session.
4. **Portability:** every artifact this project builds is model-agnostic (see
   Portability Requirement) so the follow-up Qwen3-Coder-Next project inherits it.

## Non-Goals

- No tree/multi-branch speculation (Medusa/EAGLE style) — revisit only after the
  single-chain head is healthy.
- No grammar-draft expansion, no new draft sources.
- No Qwen3-Coder-Next bring-up in this project (separate spec, see Follow-up).

## Phase 0 — Bug Gate (correctness)

Reproduce the dropped-token garbling with a fixed code-generation prompt, then
bisect by configuration:

| Run | Config | Isolates |
|-----|--------|----------|
| A | full stack as shipped (`run_glm52.sh` defaults) | baseline reproduction |
| B | `DRAFT=0` (no speculation) | speculation guilty vs innocent |
| C | `MTP=0` (n-gram drafts only) | MTP head vs draft machinery |
| D | greedy `temp=0` with drafts | rejection sampling vs exact-match path |

Ranked suspects, in order of prior probability:

1. **Rejection-sampling refusal path** (`carry_ban` + resample): Leviathan
   requires resampling from the renormalized residual distribution after a
   refused draft; an error here drops or bans the wrong token — and only shows
   up under temperature, matching the symptom (chat sampled, greedy verification
   runs were byte-identical).
2. **Emit path for accepted drafts** (token → text streaming through `zun` chat).
3. **`mtp_absorb` KV corruption** (verified-pair absorption desyncing positions).
4. **Not a bug:** int4 quantization quality — if run B garbles too, speculation
   is innocent; document and pass the gate anyway.

Executed under the systematic-debugging skill. Deliverable: root cause, fix, and
a regression test that fails on the old behavior.

**Gate:** Phases 1–3 do not start until the bug is root-caused (fixed, or proven
external to speculation).

## Phase 1 — Acceptance Probe Harness

A small offline mode (`PROBE=1` env or equivalent flag): greedy decode of a fixed
prompt (~64 tokens) that prints an acceptance table and exits:

- acceptance rate at each draft position 1..G (how deep drafts survive),
- tokens/forward, forwards, wall time,
- draft source attribution (MTP / n-gram / grammar).

Every later experiment becomes a ~2-minute run instead of a 30-minute chat
session. This is the only new tooling the project builds.

**Portability Requirement:** the probe reads only the engine's existing
draft/verify seam (draft proposal → batch verify → accept count). It must not
hard-code GLM specifics (layer counts, head names); it works for any model the
engine loads, so the Qwen3 bring-up gets it for free.

## Phase 2 — Head Wiring Diagnosis (the payoff)

1. **Reference audit:** pull the HF reference modeling code for GLM-5.2's MTP
   head. The expected formulation (DeepSeek-V3 lineage):
   `h' = Block(eh_proj(concat(RMSNorm_e(emb(t_next)), RMSNorm_h(h))))`, with
   `enorm` on the embedding, `hnorm` on the hidden state, then
   `argmax(lm_head(shared_head.norm(h')))`. Audit `mtp_draft`/`mtp_absorb`
   line-by-line against it: concat order, norm-weight assignment, position
   indices, `kv_start` handling.
2. **Probe sweep:** the 2×2 `MTP_PRENORM` × `MTP_SWAP` grid, plus whatever
   corrected wiring the audit dictates. Winner = highest acceptance.
3. **Quantization check (only if wiring proves correct):** the MTP layer is
   stored int8. Measure whether head quantization caps acceptance (e.g., compare
   draft logits against a float recompute on a small sample).

**Exit condition (either way):** acceptance ≥60%, **or** a documented finding
that ~28% is the genuine ceiling and why. The negative result is valuable: it
also kills the tree-speculation idea cheaply.

## Phase 3 — Depth Tuning and Defaults

With a healthy head:

- Probe-sweep `DRAFT` depth 2/4/6/8. Deeper drafts enlarge the batch-union
  expert set (more unique experts per forward), so there is an optimum — find it.
- Confirm end-to-end tok/s in one real chat session.
- Set the winning default in `run_glm52.sh` / `zun` (currently `DRAFT` is
  opt-in; a healthy head justifies on-by-default).
- Re-verify the lossless guarantee: byte-identical greedy output with
  speculation on vs off (MD5 method, as in the intent-profiles verification).

## Testing

- Phase 0 regression test (fails on pre-fix behavior).
- Existing test suite stays green (`make test`).
- Byte-identical greedy A/B is the ship gate for any change touching
  draft/verify/emit.
- Probe outputs for each experiment recorded in `bench/` for the record.

## Risks

- **Acceptance may be capped by head quantization,** not wiring. Phase 2's exit
  condition turns this into a cheap, documented stop instead of an open-ended
  hunt.
- **The quality bug may be outside speculation** (int4 quality, chat pipe).
  Phase 0's design (run B) detects this early; the gate passes either way once
  root-caused.
- **Reference code availability:** GLM-5.2's HF modeling file is needed for the
  audit; if unavailable, fall back to DeepSeek-V3's (same MTP lineage).

## Follow-up Project (separate spec)

**Qwen3-Coder-Next bring-up / general MoE engine.** `Qwen/Qwen3-Coder-Next`
(80B total / ~3B active, `qwen3_next` architecture, Apache-2.0, Jan 2026):

- ~40–45 GB at int4 — the whole model fits in the 128 GB unified memory; the
  NVMe tier disappears entirely.
- ~1.5–2 GB read per token → physics ceiling >100 tok/s on the 256 GB/s bus;
  realistic CPU/iGPU target 15–40 tok/s.
- Ships a native MTP head — this project's probe, audit method, and depth
  tuning transfer directly.
- Real new work: Gated DeltaNet (linear attention) + gated full-attention
  hybrid kernels, Qwen tokenizer, 512-expert top-10 + shared-expert routing,
  zero-centered RMSNorm, converter support. A full project of its own.

That project gets its own brainstorm → spec → plan cycle and inherits a healthy,
model-agnostic speculation stack from this one.
