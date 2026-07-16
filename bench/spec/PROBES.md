# MTP speculation acceptance probes

Running log — every measured probe appends a block here via `bench/spec_probe.sh <tag> [ENV=VAL ...]`.
Fixed prompt (Dijkstra/heapq), `SEED=1 TEMP=0 NGEN=96`, CPU-only. Reference baseline
(2026-07-15 full session): 0.49 tok/s, 1.85 tok/forward, 28% MTP acceptance.

Pending rows (measurement session): `base`, the Task-8 wiring sweep
(`base2`, `prenorm`, `swap`, `both`), audit candidates from WIRING-AUDIT.md
(`audit1` = `MTP_CHAINNORM=1`, the prime suspect; `audit2` = `MTP_MASK0=1`, control),
Task-10 depth sweep (`d2 d4 d6 d8`).

---
