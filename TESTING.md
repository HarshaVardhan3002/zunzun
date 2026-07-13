# Testing colibrì without the 500 GB model

`glm.c` only runs the GLM-5.2 `glm_moe_dsa` architecture, so a generic
Llama/Qwen/Mistral you already have will NOT load. You don't need the real model
though — there's a ladder, each rung tests a different layer.

## Prereq check (for rungs 2-3)
The fixture generators need transformers with the GLM-5.2 classes:
```
python -c "from transformers import GlmMoeDsaConfig; print('ok')"
```
If that fails, tell me — I'll ship a transformers-free generator (numpy -> the
int4 container directly). Rung 1 and the C unit tests need none of this.

## Rung 0 — C unit tests (no model, no GPU). Tests R0/R1/R3 logic.
```
cd c && make test
```
Runs test_cache (ExpertCache policy, R1), test_sched (placement policy, R3),
plus the existing json/st/tier/grammar tests.

## Rung 1 — G1 bandwidth gate (no model, needs ROCm). The R2 decision.
```
hipcc -O3 bench/g1_bandwidth.cpp -o bench/g1 && ./bench/g1
```
Read the verdict: host-GTT >= ~70% of VRAM -> experts stay zero-copy.

## Rung 2 — tiny oracle (~2.4 MB, generated locally). Tests R0/R1 end-to-end.
Real glm_moe_dsa architecture, token-exact validated. This proves the runtime
refactors didn't change the forward pass.
```
cd c
python tools/make_glm_oracle.py             # writes glm_tiny/ + ref_glm.json
make glm                                    # CPU build (or: make HIP=1 for the GPU path)
SNAP=./glm_tiny TF=1 ./glm 64 16 16         # expect "32/32 posizioni"
```

## Rung 3 — medium synthetic fixture (local, no download). Perf A/B for R2/R3.
```
cd c
python tools/make_glm_bench_model.py --output ./glm_bench --device cpu
SNAP=./glm_bench ./glm 64                    # CPU baseline
# scheduler knobs (R3): SCHED_SGPU (batch->GPU threshold), SCHED_MINKB (S=1 GPU min weight KB)
SNAP=./glm_bench SCHED_SGPU=2 make HIP=1 && SNAP=./glm_bench ./glm 64   # once HIP dispatch is wired
```

## What each rung proves
- Rung 0: R1 cache policy + R3 placement policy are correct (pure logic).
- Rung 1: whether the zero-copy premise holds on your silicon (G1).
- Rung 2: R0 naming + R1 cache didn't break the GLM forward pass.
- Rung 3: CPU vs GPU / zero-copy vs mirror once the GPU path is active on the box.
