# MTP speculation acceptance probes

Running log — every measured probe appends a block here via `bench/spec_probe.sh <tag> [ENV=VAL ...]`.
Fixed prompt (Dijkstra/heapq), `SEED=1 TEMP=0 NGEN=96`, CPU-only. Reference baseline
(2026-07-15 full session): 0.49 tok/s, 1.85 tok/forward, 28% MTP acceptance.

Pending rows (measurement session): `base`, the Task-8 wiring sweep
(`base2`, `prenorm`, `swap`, `both`), audit candidates from WIRING-AUDIT.md
(`audit1` = `MTP_CHAINNORM=1`, the prime suspect; `audit2` = `MTP_MASK0=1`, control),
Task-10 depth sweep (`d2 d4 d6 d8`).

---
## base  (2026-07-17 10:46:38)  []
96 tokens in 293.85s (0.33 tok/s) | expert hit rate 59.5% | RSS 99.26 GB
speculation: 2.53 tokens/forward (38 forwards per 96 tokens) | MTP acceptance 51% (58/114)
acceptance by depth [mtp]: d1 87% (33/38) d2 56% (18/32) d3 39% (7/18)

## base2  (2026-07-17 10:51:59)  []
96 tokens in 289.62s (0.33 tok/s) | expert hit rate 60.4% | RSS 101.67 GB
speculation: 2.53 tokens/forward (38 forwards per 96 tokens) | MTP acceptance 51% (58/114)
acceptance by depth [mtp]: d1 87% (33/38) d2 56% (18/32) d3 39% (7/18)

## prenorm  (2026-07-17 10:58:06)  [MTP_PRENORM=1]
96 tokens in 334.50s (0.29 tok/s) | expert hit rate 62.3% | RSS 102.64 GB
speculation: 2.13 tokens/forward (45 forwards per 96 tokens) | MTP acceptance 38% (51/135)
acceptance by depth [mtp]: d1 89% (40/45) d2 28% (11/39) d3 0% (0/11)

## swap  (2026-07-17 11:03:28)  [MTP_SWAP=1]
96 tokens in 290.33s (0.33 tok/s) | expert hit rate 64.7% | RSS 99.46 GB
speculation: 1.01 tokens/forward (95 forwards per 96 tokens) | MTP acceptance 0% (0/24)
acceptance by depth [mtp]: d1 0% (0/8)

## both  (2026-07-17 11:08:44)  [MTP_PRENORM=1 MTP_SWAP=1]
96 tokens in 284.53s (0.34 tok/s) | expert hit rate 64.8% | RSS 99.30 GB
speculation: 1.01 tokens/forward (95 forwards per 96 tokens) | MTP acceptance 0% (0/24)
acceptance by depth [mtp]: d1 0% (0/8)

## audit1  (2026-07-17 11:13:31)  [MTP_CHAINNORM=1]
96 tokens in 254.51s (0.38 tok/s) | expert hit rate 60.5% | RSS 103.61 GB
speculation: 3.00 tokens/forward (32 forwards per 96 tokens) | MTP acceptance 67% (64/96)
acceptance by depth [mtp]: d1 94% (30/32) d2 72% (21/29) d3 62% (13/21)

## audit2  (2026-07-17 11:19:21)  [MTP_MASK0=1]
96 tokens in 316.64s (0.30 tok/s) | expert hit rate 60.6% | RSS 104.12 GB
speculation: 2.46 tokens/forward (39 forwards per 96 tokens) | MTP acceptance 49% (57/117)
acceptance by depth [mtp]: d1 90% (35/39) d2 53% (18/34) d3 22% (4/18)

## default3  (2026-07-17 11:27:04)  []
96 tokens in 296.64s (0.32 tok/s) | expert hit rate 60.6% | RSS 103.49 GB
speculation: 3.00 tokens/forward (32 forwards per 96 tokens) | MTP acceptance 67% (64/96)
acceptance by depth [mtp]: d1 94% (30/32) d2 72% (21/29) d3 62% (13/21)

## rawchain  (2026-07-17 11:32:51)  [MTP_RAWCHAIN=1]
96 tokens in 313.47s (0.31 tok/s) | expert hit rate 61.7% | RSS 103.84 GB
speculation: 2.53 tokens/forward (38 forwards per 96 tokens) | MTP acceptance 51% (58/114)
acceptance by depth [mtp]: d1 87% (33/38) d2 56% (18/32) d3 39% (7/18)

## d2  (2026-07-17 11:37:12)  [DRAFT=2]
96 tokens in 227.36s (0.42 tok/s) | expert hit rate 61.2% | RSS 104.17 GB
speculation: 2.74 tokens/forward (35 forwards per 96 tokens) | MTP acceptance 86% (60/70)
acceptance by depth [mtp]: d1 94% (33/35) d2 82% (27/33)

## d4  (2026-07-17 11:41:56)  [DRAFT=4]
96 tokens in 251.11s (0.38 tok/s) | expert hit rate 61.8% | RSS 103.13 GB
speculation: 3.43 tokens/forward (28 forwards per 96 tokens) | MTP acceptance 61% (68/112)
acceptance by depth [mtp]: d1 93% (26/28) d2 85% (22/26) d3 64% (14/22) d4 43% (6/14)

## d6  (2026-07-17 11:47:02)  [DRAFT=6]
96 tokens in 273.77s (0.35 tok/s) | expert hit rate 59.1% | RSS 103.08 GB
speculation: 4.00 tokens/forward (24 forwards per 96 tokens) | MTP acceptance 50% (72/144)
acceptance by depth [mtp]: d1 100% (24/24) d2 88% (21/24) d3 60% (12/20) d4 58% (7/12) d5 71% (5/7) d6 60% (3/5)

## d8  (2026-07-17 11:54:00)  [DRAFT=8]
96 tokens in 385.88s (0.25 tok/s) | expert hit rate 60.0% | RSS 103.57 GB
speculation: 3.31 tokens/forward (29 forwards per 96 tokens) | MTP acceptance 29% (67/232)
acceptance by depth [mtp]: d1 93% (27/29) d2 62% (16/26) d3 69% (11/16) d4 73% (8/11) d5 50% (4/8) d6 25% (1/4) d7 0% (0/1)

## gate_e  (2026-07-17 12:01:19)  [DRAFT=0]
96 tokens in 249.76s (0.38 tok/s) | expert hit rate 66.5% | RSS 95.87 GB
speculation: 1.01 tokens/forward (95 forwards per 96 tokens) | MTP acceptance 0% (0/0)

## gate_d  (2026-07-17 12:06:06)  [MTP_DEBUG=1]
96 tokens in 254.36s (0.38 tok/s) | expert hit rate 62.8% | RSS 105.54 GB
speculation: 3.43 tokens/forward (28 forwards per 96 tokens) | MTP acceptance 61% (68/112)
acceptance by depth [mtp]: d1 93% (26/28) d2 85% (22/26) d3 64% (14/22) d4 43% (6/14)

## gate_e2  (2026-07-17 12:12:29)  [DRAFT=0 MTP_DEBUG=1]
96 tokens in 241.63s (0.40 tok/s) | expert hit rate 67.8% | RSS 100.19 GB
speculation: 1.01 tokens/forward (95 forwards per 96 tokens) | MTP acceptance 0% (0/0)


---
## Task-11 ship gate — near-tie divergence check (2026-07-17): PASS

Greedy plain (`gate_e`/`gate_e2`, DRAFT=0) vs greedy speculative at the adopted
default (`gate_d`, chain-norm, DRAFT auto=4), same prompt/seed:

- **Clean text both sides** — valid Dijkstra/heapq Python, no dropped tokens.
- **Plain greedy is reproducible**: gate_e and gate_e2 text regions hash identical
  (md5 499f6b6e74a263c56185a3fc5e7633df).
- **First divergence = one near-tie flip at pos 40** (input token 1191), same top-2
  candidates in both runs, order reversed by batched-vs-single FP numerics
  (Phase-0 verdict, BISECT.md):
  - E (S=1 row): `top1=5084 26.657 top2=11 26.523 gap=0.135`
  - D (S=5 row): `top1=11 26.960 top2=5084 26.889 gap=0.071`
  Typical gaps elsewhere run 2–13; every later difference is conditioned on this
  flip (different context, legitimately different continuation).
- Remaining gate item (user session): real GPU-config chat vs the 2026-07-15
  baseline 0.49 tok/s / 1.85 tok/fw / 28% acceptance.

## Summary (Tasks 6–11)

| config | acceptance | tok/fw | probe tok/s |
|---|---|---|---|
| old default (raw-hx chain, d3) | 51% | 2.53 | 0.31–0.33 |
| chain-norm d3 (audit1) | 67% | 3.00 | 0.32–0.38 |
| **chain-norm d4 (new default)** | **61%** | **3.43** | **0.38** |
| chain-norm d2 | 86% | 2.74 | 0.42 |
| chain-norm d6 | 50% | 4.00 | 0.35 |
| chain-norm d8 | 29% | 3.31 | 0.25 |

WIRING-AUDIT seam 4 (recycle post-shared_head.norm hidden into chained drafts,
as vLLM/SGLang) was the acceptance bug; adopted in glm.c with `MTP_RAWCHAIN=1`
as the legacy control. Spec goals met at the new default: acceptance ≥60,
tok/fw ≥3. Controls: MTP_SWAP 0%, MTP_PRENORM kills d3, MTP_MASK0 noise.

## Task-11(c) real CPU-config chat session (2026-07-17, post-cleanup)

`CPU=1 ./run_glm52.sh`, fresh KV, topp 0.7, draft auto=4 (chain-norm): turn 1
46 tok 0.39 tok/s hit 63%; turn 2 535 tok **0.44 tok/s** hit 66%, RSS 107 GB.
Output quality clean (valid single-file HTML, no dropped tokens — the original
bug symptom did not reproduce). Reference warm baseline 0.49 tok/s (old wiring,
warmer cache). Interpretation: at 63-66% expert hit rate the session is
NVMe-bound; speculation multiplies tokens/forward but batch-union also
multiplies unique experts/forward (probe: 775/token at DRAFT=0 vs 1073 at d4),
so wall-clock is ~parity cold (probe A/B: 0.40 vs 0.38). The fix's real effect:
OLD wiring speculation was a net LOSS (d3 probe 0.31-0.33 vs 0.40 no-spec);
NEW wiring is parity cold and net-positive as hit rate rises (accepted tokens
amortize cached-expert reads). tok/s headroom on this box lives in the
hit-rate/caching project, not deeper speculation.
