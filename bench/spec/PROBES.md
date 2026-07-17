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

