# Phase-0 bisect — 2026-07-16, SEED=1, NGEN=192, CPU-only, commit 47945fb

Prompt: Dijkstra/heapq + JS Particle + scoring function, `Output only code.`
All runs: `SNAP=C:/Users/Von/Downloads/model PIPE=1 SEED=1 NGEN=192`, no COLI_HIP/CUDA_DENSE.
Hardware note: executed at a 64/64 GB BIOS memory split (system/VRAM), changed this
session — this explains the low expert hit rates (27–45%) and low tok/s vs earlier
baselines; it does NOT affect garbling verdicts, MD5 checks, or acceptance %.

| run | config            | garbled? | evidence (quoted lines) |
|-----|-------------------|----------|-------------------------|
| A   | temp .7, draft 3  | no — clean, no dropped tokens found | 192 tokens read in full; all expressions complete, e.g. `pq = [(0, start_node, [start_node])]`. Ends mid-comment at NGEN cap (`# Keep track of the shortest known distance to each`) — truncation, not garbling. |
| B   | temp .7, draft 0  | no — clean, no dropped tokens found | All expressions complete, e.g. `pq = [(0, [start_node], start_node)]`. Ends at NGEN cap on `shortest_distances` (truncation, not a dropped token). |
| C   | temp .7, MTP=0    | not run — trigger was "A garbles AND B clean"; A was clean | — |
| D   | greedy, draft 3   | no — clean, no dropped tokens found | Generated text is byte-identical to run A (md5 `335cdf6c138582929d9aca69e3927eed` for both text regions). |
| E   | greedy, draft 0   | no — clean, no dropped tokens found | Generated text is byte-identical to run B (md5 `a92f7eab92cf258e5d4fc7dd9cd7b07e` for both text regions). |

The known dropped-token pattern (`this.x -= ;`, `score = ;`) did NOT reproduce inside
any single run's 192-token window. The mechanical greedy check below caught the bug
instead.

## Greedy MD5 check (D vs E)

Full file:

```
8acfbdce3e7e792116af95f300aaca4e  bench/spec/run_D.out
6311f22b92044e0bca1225d81fa45aa7  bench/spec/run_E.out
```

Differ — expected in part, since headers echo `draft=3` vs `draft=0` and the stats
blocks legitimately differ. Header-stripped (`tail -n +4`):

```
475db3788c66c5f1b7b0da7e205467be  run_D.out (tail -n +4)
ba520ad6ce20c65d5185bebf4ca447fc  run_E.out (tail -n +4)
```

Still differ, and the difference is in the GENERATED TOKEN STREAM itself, not just
stats. **The lossless greedy invariant is BROKEN.** First divergent line (docstring
line 9 of the generated text; identical up to that point):

```
D (draft=3):        graph (dict): An adjacency list representing the graph.
E (draft=0):        graph (dict): Adjacency list representing the graph.
```

D emits `An ` where E (plain autoregressive greedy = ground truth) emits `Adjacency`;
the streams fully diverge from that token onward, as expected once one token differs.
Both continuations are individually coherent, but under TEMP=0 with exact-match
verification they must be identical — they are not.

Cross-run identity check (md5 of generated-text region only, lines 4..`---`):

```
run_A 335cdf6c138582929d9aca69e3927eed   == run_D
run_B a92f7eab92cf258e5d4fc7dd9cd7b07e   == run_E
```

So TEMP=0.7 SEED=1 produced text byte-identical to TEMP=0 in both draft settings —
the temperature path consumed RNG without ever changing a token choice across 192
code tokens. Plausible for peaked code distributions but worth an eyebrow; the
operative split in this data is draft=3 vs draft=0, not temperature.

## Baseline stats (from run_A.out)

- 192 tokens in 983.26s (0.20 tok/s), RSS 46.84 GB
- speculation: **2.26 tokens/forward** (85 forwards per 192 tokens)
- **MTP acceptance 42%** (106/255)
- expert hit rate 26.8% (64/64 split; see hardware note), experts loaded/token 1213.7, TOPK=0 TOPP=0.00
- Run B (draft=0) reference: 679.23s (0.28 tok/s) — speculation was net SLOWER on
  this box/config; hit rate 43.9% vs 26.8% (drafting thrashes the expert cache at
  the 64/64 split)

## Interpretation

- D ≠ E under TEMP=0 means speculation changes greedy output. That indicts the
  **verify/accept machinery (or the batched speculative forward pass itself)**, not
  the Leviathan rejection-sampling path: temperature never enters at TEMP=0, and the
  divergence appears with default MTP drafts anyway. Two candidate mechanisms:
  1. a logic bug in the exact-match accept path (off-by-one in accepted length,
     bonus-token handling, or KV rollback), or
  2. numerics: multi-token speculative forwards reduce in a different order than
     single-token forwards, nudging near-tied logits so argmax flips (`An` vs
     `Adjacency` is a classic near-tie). Either way the invariant as shipped does
     not hold, and the refactor target must make D ≡ E.
- The A≡D / B≡E identities mean the sampled runs traversed the exact same token
  paths as greedy, so this dataset cannot separate "speculation×sampling
  interaction" from "speculation alone" — but it doesn't need to: speculation alone
  already breaks the guarantee.
- The user-visible dropped-token garbling did not reproduce here. These runs used
  TOPP unset (exact expert routing) while the garbled chat sessions ran **TOPP=0.7**
  — since A and B are both clean, TOPP=0.7 (approximate expert routing) is the next
  suspect axis to bisect (deliberately NOT run in Phase 0).
- Golden outputs for the refactor: `run_D.out` / `run_E.out` (greedy). Note they
  are golden per-config; they do not currently match each other, and making them
  match is a success criterion, not a regression.

## Run ledger

| run | duration | tok/s | tok/forward | MTP acceptance | hit rate |
|-----|----------|-------|-------------|----------------|----------|
| A   | 983.26s  | 0.20  | 2.26        | 42% (106/255)  | 26.8%    |
| B   | 679.23s  | 0.28  | 1.01        | 0% (0/0)       | 43.9%    |
| D   | 983.20s  | 0.20  | 2.26        | 42% (106/255)  | 30.2%    |
| E   | 681.20s  | 0.28  | 1.01        | 0% (0/0)       | 45.2%    |

## Phase 0 verdict — 2026-07-16, commit d331318 + MTP_DEBUG instrumentation

**Root cause of D≠E: H2 — batched-vs-single forward numerics.** Two concrete,
independently demonstrated mechanisms; no bookkeeping bug (H1 exonerated below).
All probes: same prompt/SEED=1/TEMP=0, NGEN=96 for the gap probes, NGEN=4 for the
layer-trace micro-probes. Instrumentation added (committable, `MTP_DEBUG`-gated):
`[gap]` top-2 logits per verify row (MTP_DEBUG=1), `[ltrace]`/`[kvsum]` per-layer
per-row activation checksums + per-layer KV-prefix checksums (MTP_DEBUG=3).

### Mechanism 1 — I4S kernel S-gate (dominant, affects every batch row)

`matmul_qt` (c/glm.c ~line 506) selects the kernel for int4 (fmt=2) tensors by
batch size: `S>=g_i4s` (g_i4s=2 on AVX2/VNNI) uses the int8-activation IDOT
kernel (`matmul_i4_idot`, activations quantized per row, ~0.3% RMS added error);
S=1 uses the exact-f32 path (`matmul_i4`). Plain greedy decode runs every forward
at S=1; speculative verify runs S=1+draft. **Same weights, same inputs, different
kernel** — every row of a verify forward, including row 0, differs from the plain
forward from layer 0 onward.

Evidence (probe_ltrace_D/E.err, NGEN=4, MTP_DEBUG=3):
- Prefill traces (S=43 in both runs): **bit-identical**, 3397/3397 checksum lines
  — both runs take the S>=2 kernel there.
- KV prefix at the first decode forward (kv=43): identical across all 78 layers.
- Embed row identical, yet row 0 of the S=4 verify batch diverges at **layer 0**
  (sum 2.995758175e-01 vs 2.916891106e-01), i.e. inside a single layer with
  identical inputs — kernel choice, not state.
- Control (probe_i4s1_D/E.err, `I4S=1` forces the IDOT kernel at S=1 too):
  row 0 becomes **bit-identical end-to-end**; its logits match E to all printed
  digits (top1=12663 @ 29.777710, top2=49235 @ 20.655693 in both).

### Mechanism 2 — MoE batch-union accumulation order (rows s>=1)

In `moe()`, a batch row's routed-expert outputs are accumulated into `out` in the
batch-union order (first-seen across all rows), not the row's own routing order
used at S=1 → floating-point reassociation. The seed difference is tiny but gets
amplified downstream by the IDOT activation-quantization rounding cliffs
(qrow_i8's lrintf flips on ~1e-6 perturbations).

Evidence (probe_i4s1_D/E.err, with mechanism 1 removed via I4S=1): row 1
(pos 44) is bit-identical through embed and layers 0–2 — exactly the dense
layers (`first_k_dense_replace=3`) — and first diverges at **layer 3, the first
MoE layer**. Attention/dense/lm_head kernels are per-row S-invariant once the
I4S gate is neutralized; only the MoE union path remains, and that is where the
divergence enters.

### The argmax flip itself (probe_gap_D/E.err, NGEN=96, MTP_DEBUG=1)

All 46 positions before the divergence have the same top-1 in D and E (top-1
logit drift up to 0.93 without a flip). At pos 89 (`An` vs `Adjacency`, the
golden divergence, reproduced byte-for-byte at NGEN=96):

```
E: pos=89 row=0/1 top1=62475 24.427853 top2=1527  24.162287 gap=0.265566
D: pos=89 row=1/4 top1=1527  23.866558 top2=62475 23.840719 gap=0.025839
```

Same top-2 SET, opposite order, near-tie gaps (1.1e-2 and 1.1e-3 of |logit|) —
the H2 signature. Under exact-argmax greedy verify the flipped draft is its own
verifier, so the wrong token is accepted and the streams fork.

### H1 (bookkeeping bug) exonerated

- `spec_decode`/`mtp_absorb` row↔position indexing verified correct: probe `pos`
  and `in_tok` fields align exactly with E's positions across all 162 verify rows.
- Row 0 bit-exactness under I4S=1 rules out any KV/position/rollback bookkeeping
  error on the verify path (a stale or misplaced KV row would not heal by
  changing a matmul kernel).
- Code inspection of `all+kv+1`, `lo+k*V`, `hlast` row-k copy, `kv+=1+k`,
  stale-KV overwrite: coherent; MTP-head KV inconsistencies can only change
  PROPOSALS (rejected drafts), never emitted text, under exact-match verify.
- The n-gram discriminator run (Run C) was deemed unnecessary: both mechanisms
  were localized directly to the shared batched-forward path (layer 0 kernel
  gate + layer 3 MoE union), which n-gram drafting exercises identically.

### Verdict and implication

**Greedy speculative ≡ greedy plain cannot be guaranteed bit-exact on this
engine.** Verify forwards (S=1+draft) and plain decode (S=1) are numerically
different programs: kernel selection is S-gated and MoE accumulation order is
batch-dependent. Making them bit-identical would require forcing one kernel for
all S (a measured perf regression on AVX2 — the gate exists on purpose) AND
per-row-ordered MoE accumulation; out of scope per plan. Losslessness holds **in
distribution**: the Leviathan accept/resample math is exact (c/sample.h
statistical test, 400k samples/draft, 3σ, PASS), so speculation does not bias
sampling — it just cannot promise the same single trajectory under argmax
near-ties (~1 flip per ~50–90 code tokens observed here, drift-dependent).
`I4S=1` removes the dominant mechanism if closer (not exact) greedy agreement is
ever wanted.

### TOPP=0.7 garbling probe (closes the Phase-0 garbling question)

`run_D_topp07.out` (greedy, draft=3, TOPP=0.7, NGEN=96): **clean** — coherent
code, no dropped-token pattern (`x -= ;` absent), ends at the NGEN cap
mid-expression (truncation). Text differs from run D as expected (TOPP
approximates expert routing, perturbing all logits), acceptance 38%, 0.42 tok/s.
The user-visible garbling did NOT reproduce on the TOPP axis in this window;
with A/B/D/E also clean, the original garbling remains unreproduced in Phase 0
(plausibly fixed by intervening commits, a longer-context effect, or an
interactive-chat-path issue — it is NOT explained by the speculation invariant,
which this verdict closes).

### Phase-0 probe ledger

| probe | config | NGEN | artifact |
|-------|--------|------|----------|
| gap D | greedy draft=3, MTP_DEBUG=1 | 96 | probe_gap_D.out/.err (reproduces run_D text) |
| gap E | greedy draft=0, MTP_DEBUG=1 | 96 | probe_gap_E.out/.err (reproduces run_E text) |
| ltrace D | draft=3, MTP_DEBUG=3 | 4 | probe_ltrace_D.err |
| ltrace E | draft=0, MTP_DEBUG=3 | 4 | probe_ltrace_E.err |
| i4s1 D | draft=3, I4S=1, MTP_DEBUG=3 | 4 | probe_i4s1_D.err |
| i4s1 E | draft=0, I4S=1, MTP_DEBUG=3 | 4 | probe_i4s1_E.err |
| topp07 | greedy draft=3, TOPP=0.7 | 96 | run_D_topp07.out/.err |
