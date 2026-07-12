# Phase 0 — Baseline (Strix Halo 395+, 128 GB, Win11 primary)

Goal: measure disk/matmul split and validate the AVX-512 VNNI build win before writing any code. No source changes in this phase.

## Prereqs
1. MSYS2 (UCRT64 shell) with `gcc make binutils` installed: `pacman -S mingw-w64-ucrt-x86_64-gcc make binutils`
2. Model converted at some `<model_dir>` (needs `tokenizer.json` + shards). Verify the MTP head is **int8**, not int4 (issue #8 sizes `1765523544/2686077736/536747200` = int4 = bad).
3. ~30 GB free RAM headroom; close heavy apps. Set Windows "variable graphics memory" low for now (CPU-only phase).

## Run
```bash
cd /c/Users/Von/Desktop/colibri
./phase0/bench.sh /c/path/to/glm52_i4
```
Runtime: model reloads per run (~min each) + decode at 0.1–0.5 tok/s. Full matrix ≈ 1–2 h. Tune with `NGEN=16 RUNS=1 WARMUP=1 ./phase0/bench.sh ...`.

## What it does
1. Builds `glm_v3.exe` (AVX2 baseline, current default) and `glm_native.exe` (znver5 → AVX-512 VNNI kernels active). Prints `vpdpbusd` count as proof VNNI compiled in (v3 should be 0, native > 0).
2. `iobench` on one shard, buffered + O_DIRECT flag (no-op on Windows — if both numbers match, that confirms the compat.h gap).
3. Warmup runs (feed AUTOPIN usage history + OS page cache), then the matrix, greedy, seed 42, `NGEN=32` (matches README community protocol):

| config | isolates |
|---|---|
| A_v3_stock | today's default build |
| B_native | pure AVX-512/VNNI build win (B vs A) |
| C_native_topp | expert top-p 0.7 read cut (C vs B) |
| D_native_pilot | `PILOT=1` — expect ≈ C on Win (fadvise no-op) → quantifies Phase 2 stakes |

Dropped from matrix: int4-IDOT-at-S=1. `g_i4s` is a compile-time static on x86 (glm.c:313–320, no `I4S` env knob despite the comment) → becomes a 1-line env-override patch in Phase 1.

Output: `phase0/results.csv` (config, tok, secs, tok/s, hit%, RSS, tok/fw) + full logs in `phase0/logs/`.

## Read the results
- **B/A ratio** = the free build-flag win. If ≥1.15x, Phase 1 makes `native` the documented default for this class.
- **hit%** low on early runs is normal; AUTOPIN learns across runs (#39 went 0.16→0.40 tok/s in 5 runs). Re-run the matrix next day for warm numbers.
- **D ≈ C** confirms Windows prefetch is dead → Phase 2 (overlapped I/O, `VirtualLock`, `FILE_FLAG_NO_BUFFERING`) justified.
- **iobench GB/s** vs #39's 3.27 GB/s Optane: tells us if your NVMe or matmul is the binding constraint (their profile: 49% disk / 47% matmul).
- Later Linux cross-check (same script runs unmodified in bash) gives the fadvise/mlock delta for free.

## Next gates
- B/A ≥ 1.15x → Phase 1 (VNNI default, `g_i4s` env-override + tuning).
- D ≈ C and disk% ≥ 40 → Phase 2 (Windows I/O).
- Both done → Phase 3 (HIP zero-copy tier for 8060S).
