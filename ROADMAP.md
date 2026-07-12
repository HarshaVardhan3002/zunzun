# Colibrì → MoE Routing OS — Roadmap

Target hardware: Ryzen AI Max+ 395 (Strix Halo), 128 GB LPDDR5X unified, Radeon 8060S (gfx1151), XDNA2 NPU (50 TOPS), 16× Zen 5. Windows 11 primary, Linux for benching.

## Direction (2026-07-12, runtime-first)
Thesis: **heterogeneous unified-memory runtime for streamed MoE**, not another engine.
colibrì grows its own CPU+GPU co-execution on a single zero-copy expert cache. Full
design in **ARCHITECTURE.md**. llama.cpp is demoted to baseline/reference; the gateway
(`router/`) survives as an optional top layer (R5), not the thesis.
- **runtime** = the 4-subsystem stack (ModelProvider / ExpertCache / Scheduler / Backend); model-agnostic (GLM/DeepSeek/Qwen).
- **colibrì engine** = becomes the reference implementation of that runtime.
- **gateway** = optional multi-model front door on top.
- **XDNA / NPU** = out of scope for now (prefill-only at best; see engines/ggml-xdna/DESIGN.md).

Superseded earlier framing: "llama.cpp primary for GPU" and "router routes requests" are
no longer the center. Runtime track = R0–R5 below.

## Phases

### ✅ 0 — Baseline (deferred by user)
`phase0/` runbook + bench.sh. Not run; user skipped. Numbers still open.

### ✅ 1 — Free CPU wins (done)
- `I4S` runtime env override (glm.c:2514).
- Windows Makefile default `native` → AVX-512 VNNI on by default (Makefile:31).
- Portable AVX2 build preserved via `make portable`.

### Runtime track — R0–R5 (see ARCHITECTURE.md for interfaces + gates)
- **R0** Extract `ModelProvider` from glm.c (GLM adapter). No behavior change; verify vs `ref_glm.json`.
- **R1** `ExpertCache` abstraction wrapping `tier.h`. CPU-only, pure refactor.
- **R2** HIP GPU backend: dense + hot-expert matmul on 8060S, zero-copy from cache. **Gated by G1** (host-GTT vs VRAM read BW).
- **R3** Scheduler: size-aware CPU/GPU placement + prefetch overlap. Gated by G2 (matmul crossover S), G3 (overlap efficiency).
- **R4** Second `ModelProvider` (Qwen or DeepSeek) to prove model-agnostic.
- **R5** Gateway (`router/`, built this session) as optional multi-model top layer.

### Built this session (now supporting, not central)
- `engines/llamacpp/` (setup.ps1 + presets.yaml) — llama.cpp as baseline/reference on the 8060S.
- `router/` (router.py + registry.yaml) — OpenAI-compatible gateway, tested. Re-slotted to R5.

### Empirical gates (blocking, ordered)
- **G1** host-GTT vs VRAM read BW on 8060S — existential, run before R2.
- **G2** iGPU expert-matmul crossover S (incl. dispatch).
- **G3** prefetch+GPU overlap hides SSD latency on the 49%/47% profile.

## Non-goals
NPU decode (DMA-bound). Multi-NUMA (single memory domain; only `OMP_PLACES` across the 2 CCDs worth a look). Training. Dense-only models (run, no streaming win).
