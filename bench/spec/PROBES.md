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

