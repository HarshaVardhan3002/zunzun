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
