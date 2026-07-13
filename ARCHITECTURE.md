# Heterogeneous Unified-Memory Runtime — Architecture

Run any streamed MoE (GLM / DeepSeek / Qwen) on AMD Strix Halo with CPU and GPU
co-executing on a **single zero-copy expert cache** in unified memory. Model stays
on NVMe; only active experts live in the 128 GB LPDDR5X pool; the pool is one
allocation with two device views (no CPU↔GPU duplication).

Contribution is the runtime, not the engine: SSD-streamed MoE + zero-copy shared
expert cache + predictive CPU/GPU co-scheduling. Each piece exists in isolation;
the combination does not.

## Four subsystems (the model-agnostic contract)

The design is a stack of four interfaces. Swap a **ModelProvider** to run a new
model; swap a **Backend** to run new hardware. Scheduler and Cache never change.
That separation is the reusability claim.

### 1. ModelProvider (storage)
Owns on-disk layout. Turns a model family into a uniform expert map.
```
metadata()               -> {n_layers, experts_per_layer, top_k, dims, quant, mtp?}
expert_ref(layer, eid)   -> {fd, offset, len}          # one coalesced pread
dense_ref(name)          -> {fd, offset, len}          # attn/embed/shared experts
```
One adapter per family parses config.json + the safetensors index into this map.
Today glm.c hardcodes the GLM shard layout; R0 extracts it behind this interface.

### 2. ExpertCache (unified memory)
The shared pool. Allocates expert slots in memory that is CPU-addressable **and**
GPU-mappable in the same bytes.
- ROCm: `hipHostRegister` the existing 4K-aligned pinned slabs (coarse-grained),
  or `hipHostMalloc`. GTT-backed => iGPU reads host memory directly, zero copy.
- Slot manager: fixed ~19 MB slabs, LRU + pinned hot-store. Reuse `tier.h` +
  `.coli_usage` AUTOPIN as-is.
```
acquire(layer, eid) -> slot          # resident or triggers fetch+wait (miss)
host_ptr(slot) / dev_ptr(slot)       # two views, same allocation
pin(eid) / evict()
residency(slot) -> HOST | HOST_PLUS_GPU_MIRROR
```
**Coherence contract:** experts are read-mostly. The SSD reader fills a slab and
issues a release fence; the GPU consumer acquires before read. No fine-grained
coherence needed, so we use the cheaper coarse-grained path.
**Conditional (gated by G1):** if iGPU host-GTT read BW << VRAM carveout BW, keep
a VRAM **mirror** for pinned hot experts only; `residency()` exposes it and the
scheduler prefers the mirror for GPU ops. Cold/one-shot experts stay zero-copy.

### 3. Scheduler (placement + overlap)
Each forward pass is a small task graph. The scheduler places every op on a device
and keeps three lanes busy.
```
place(op) = f(op_type, S, tensor_bytes, residency, device_queue_depth)
```
Policy:
- `S >= S_gpu` (prefill, MTP-verify batch): matmuls -> GPU (real compute win).
- `S == 1` decode: expert matvec -> **CPU by default**; -> GPU only if the expert
  is GPU-resident and bytes exceed the crossover (G2). iGPU dispatch overhead can
  exceed a single matvec, so CPU owns the S=1 tail.
- attention / dense-resident -> GPU when weights resident, else CPU.
Three lanes overlapped: (a) I/O prefetch thread, (b) CPU OpenMP pool, (c) GPU
stream. Layer L+1 routing is predicted from layer L post-attention state
(PILOT, ~71.6% predictable) and feeds the prefetcher so SSD latency hides
under current-layer compute.

### 4. Backend (device execution)
Uniform op interface; each backend advertises what it supports, scheduler routes
around gaps.
```
supports(op, quant, S) -> bool
matmul(out, x, weight_handle, dims, quant, S)   # weight_handle from ExpertCache
attention(...) / dense(...)
```
- CPU backend: existing glm.c kernels (AVX-512 VNNI IDOT).
- GPU backend: HIP kernels (int4/int8 × int8 matmul, attention), operating on
  `dev_ptr(slot)` directly (zero-copy) or the VRAM mirror.
- `supports()==false` => scheduler places the op on the other device. Guarantees
  correctness before the GPU backend is complete.

## Decode data flow
`tokenize(CPU) -> route(CPU, top-k) -> cache.acquire (hit | SSD fetch) ->`
`schedule expert matmuls (CPU tail / GPU hot) -> attention+dense(GPU) -> combine`
`-> sample(CPU)`. Prefetch thread is already pulling L+1 experts from predicted
routing.

## Concurrency
- 1 I/O prefetch thread (or small pool): `pread` slabs, release fence.
- N CPU workers (OpenMP), pinned across the 2 CCDs (`OMP_PLACES` experiment).
- 1+ HIP stream, async.
- Slot refcount: a slab in use by the GPU is unevictable until the stream retires.

## Empirical gates (ordered, blocking)
- **G1 — host-GTT vs VRAM read BW on 8060S.** Microbench. Existential: decides
  pure zero-copy vs hot-mirror. Run before R2.
- **G2 — iGPU expert-matmul crossover S** (incl. dispatch cost). Sets `S_gpu`.
- **G3 — overlap efficiency.** Does prefetch + GPU compute hide SSD latency on the
  balanced 49% disk / 47% matmul profile.

## Phases (runtime track, supersedes old Phase 3–5)
- **R0** Extract `ModelProvider` from glm.c (GLM adapter). No behavior change;
  verify against `ref_glm.json` golden output.
- **R1** `ExpertCache` abstraction wrapping current `tier.h`. Still CPU-only.
  Pure refactor, same output.
- **R2** HIP backend: dense + hot-expert matmul on 8060S, zero-copy from cache.
  Gated by G1. A/B vs CPU.
- **R3** Scheduler: size-aware placement + prefetch overlap. Gated by G2, G3.
- **R4** Second `ModelProvider` (Qwen or DeepSeek) to prove model-agnostic.
- **R5** Gateway (existing `router/`) sits on top as the multi-model layer. Optional.

## Risks
- **G1 bandwidth** is the existential risk; the whole zero-copy novelty rests on it.
- **ROCm gfx1151 on Windows** is newer than Linux. Plan: bench/develop on Linux;
  if Windows HIP is too raw, the GPU Backend can target **Vulkan compute** instead
  (more code, portable). Interface #4 is written so the backend is swappable.
- **Refactor regression** (R0/R1) on the working CPU path. Mitigate with golden
  output tests (`ref_glm.json`, `tests/`).

## Non-goals
NPU decode (DMA-bound, starves). Multi-NUMA (single memory domain). Training.
Dense-only models (run fine, no streaming win).
