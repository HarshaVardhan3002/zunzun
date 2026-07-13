# Intent Profiles — Design

**Date:** 2026-07-13
**Status:** Approved by user (brainstorming session)
**Goal:** Let the user declare a session's domain at startup (`PROFILE=coding`) so the
engine pre-pins the expert hot set matching that domain instead of the blended global
history. Fewer SSD misses per token, faster warm-up.

## Background

Routing in GLM-class MoE is per token, per layer: every token touches ~616
expert-instances (~11.6 GB). The engine already keeps a persistent usage histogram at
`<SNAP>/.coli_usage` (loaded at startup, saved atomically each serve turn) and AUTOPIN
pins the historically hottest experts into RAM within a confidence-scaled budget
(glm.c: `usage_load`, `usage_save`, `pin_load`, AUTOPIN block near the end of `main`).
Pinned experts are never evicted — the SSD "to-and-fro" only exists in the LRU half of
the cache. Routing statistics differ by domain, so a domain-specific history sharpens
the pin set.

Current measured baseline on the target box (Strix Halo, 128 GB): ~0.30 tok/s warm chat,
~79% of time in expert disk reads, 57% cache hit rate. Realistic expectation for this
feature: hit rate toward ~90% in a matching session → roughly 2–2.5× tok/s plus a much
faster warm-up. It is one layer of the stack toward the 4–5 tok/s goal, not the whole
path.

## Decisions made during brainstorming

- **Blend at load** (chosen over seed-then-diverge and fully-separate): a profile
  sharpens the global history rather than replacing it. No profile ever starts cold.
- **Env var + CLI flag** (chosen over in-chat switching and auto-detection): profile is
  fixed for the session.
- **Engine-native** (chosen over wrapper-level merging and generalized multi-file
  weights): all logic at the existing AUTOPIN seam in glm.c; every entry point
  (`run_glm52.sh`, `coli`, bare `glm`) gets it.
- **No on-disk index/repack** (user idea, declined with measurement): each expert read
  is a single ~19 MB contiguous read at a known offset computed from shard headers —
  there is no "find" step, and at that read size NVMe already delivers near-full
  bandwidth (3.65 GB/s measured with DIRECT=1). Layout changes buy nothing; a repack
  would rewrite 362 GB.
- **Speculative co-occurrence prefetch deferred, not rejected** (follow-up below).

## Data model

Two files in the snapshot dir, same `"layer eid count"` text format:

- `<SNAP>/.coli_usage` — global blended history (existing, unchanged).
- `<SNAP>/.coli_usage.<profile>` — per-profile history.

Profile names are sanitized to `[A-Za-z0-9_-]`, max 32 chars. An invalid name warns and
the session runs profile-less (no odd file paths are ever created).

In memory:

- `m->eusage[l][e]` changes meaning from "loaded history + live" to **session deltas
  only** (starts at zero; router increments as today).
- `gbase[l][e]` — counts loaded from the global file.
- `pbase[l][e]` — counts loaded from the profile file (allocated only when `PROFILE` is
  set). Each array is ~81 KB (79 layers × 256 experts × u32) — negligible.

**Pin score:** `gbase + W × pbase + eusage`, `W = 4` by default, overridable via
`PROFILE_W`. A few thousand profile selections outvote the ~33k global where they
disagree; the global decides ties.

**AUTOPIN confidence:** thresholds (≥ 5000 selections to pin at all, full trust at
≥ 200k) apply to the raw unweighted total `sum(gbase) + sum(pbase)`, preserving their
current meaning.

## Data flow

**Startup:**
1. Load global file → `gbase`.
2. If `PROFILE` set: load `<SNAP>/.coli_usage.<profile>` → `pbase`. Missing file is
   fine — new profile, starts empty, announced as such.
3. Blend into an in-memory record list; AUTOPIN pins from it via a new `pin_load_mem`
   (the file-reading `pin_load` remains for the explicit `PIN=` flag).
4. Stderr: `[PROFILE] coding: 5,231 selections (+ 33,412 global, weight 4x)`.

**Each serve turn (`usage_save`):**
- Global file ← `gbase + eusage` (atomic tmp+rename, as today).
- Profile file ← `pbase + eusage` (same mechanism).
- A selection made under a profile lands in both histories: the global stays
  comprehensive, the profile stays sharp. Saving is idempotent — baselines are never
  re-added to themselves, so repeated saves cannot inflate counts.

**`STATS=<file>` diagnostic dump:** writes the full blended view
(`gbase + pbase + eusage`) — "everything this run knew and did", matching current
semantics as closely as possible.

**No `PROFILE` set:** `pbase` is never allocated; pinning and saving reduce to exactly
today's behavior.

**Observability:** the existing per-turn stats line gains nothing new, but the startup
`[PROFILE]` line plus the existing hit-rate reporting let the user compare a profiled
vs unprofiled session directly. The engine also prints the pin-set overlap between
profile-blended and global-only rankings at startup (one line, e.g.
`[PROFILE] pin set: 61% differs from global`), so the user can see whether the profile
is actually changing anything.

## UX surface

- Shell: `PROFILE=coding ./run_glm52.sh` (env passthrough — no script change needed;
  add a commented example line like the other opt-ins).
- CLI: `coli chat --profile coding` / `coli serve --profile coding` — one argparse flag,
  one line in `env_for` setting `PROFILE`.
- Fixed for the session; no in-chat switching. REPIN (opt-in, separate decaying heat
  map) continues to adapt within a session, unchanged.
- Non-goal: profile management commands. Profiles are plain files in the snapshot dir;
  `ls` and `rm` are the management UI.

## Error handling

- Corrupt/truncated profile file: same tolerance as the global file today (`fscanf`
  parsing stops at the first malformed line; non-fatal).
- Save failure (disk full, permissions): non-fatal, existing quiet/warn behavior of
  `stats_dump_q`.
- Concurrent sessions on the same profile: last-writer-wins per turn, same as the
  global file today. Accepted; single-user box.
- Invalid profile name: warn once on stderr, run without a profile.

## Testing

- **Unit (`c/tests/test_profile.c`, follows the `test_cache` pattern):**
  - Blend math: `gbase + W×pbase + eusage` ordering and weighting.
  - Delta save: save twice, reload — counts must not inflate (no double-count of
    baselines).
  - Name sanitization: bad names rejected.
  - Round-trip: save profile + global from a synthetic session, reload both, verify
    each contains baseline + delta.
- **Integration (manual, on the box):**
  - Greedy output byte-identical with and without `PROFILE` — pinning changes only
    where weights are read from, never the result.
  - After a profiled session: both files grew; second start announces the profile
    history and pins a visibly different set.

## Follow-up (recorded, not in scope)

**Speculative co-occurrence prefetch:** use transition statistics ("layer 12 expert 87
→ layer 13 expert 203, X%") to issue reads before the router decides. Only pays if
routing transitions are predictable enough; mispredictions waste bandwidth on a
bandwidth-bound box. First step is measurement, not code: log transition pairs for a
real session, analyze predictability offline. Revisit after profiles ship.
