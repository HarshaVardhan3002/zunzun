# ggml-xdna — XDNA2 NPU backend (design + skeleton, long-horizon)

Status: **design only.** No working driver. Gated on Phase 3 being useful.

## Scope
A ggml backend that offloads `GGML_OP_MUL_MAT` for **prefill** to the XDNA2 NPU
(50 TOPS int8) via XRT. Decode stays on CPU/iGPU.

## Why prefill-only (non-negotiable)
Decode at batch 1 is matvec: arithmetic intensity ≈ 2 ops/byte, pure DMA-bound.
The NPU can't be fed and would just contend for the same LPDDR5X fabric the
CPU/iGPU already use. NPUs win on high reuse — that's prefill (S≫1, weights
reused across the whole prompt) and MTP verification batches. AMD's own Lemonade
splits exactly this way (NPU for TTFT, iGPU for sustained gen), and the audit
reached the same verdict. Anyone promising NPU decode speedup on this class is wrong.

## Stack
- **XRT** (Xilinx Runtime) — device open, BO alloc, kernel dispatch.
- **MLIR-AIE / IRON** — author the AIE tile program (the int8 GEMM microkernel).
- **amdxdna** driver — mainlined Linux 6.14 (Windows: AMD's own runtime). → **Linux-first.**

## Surface (mirrors ggml-backend API)
```
ggml_backend_xdna_init(device)
ggml_backend_xdna_buffer_type()         # NPU-visible BOs
supports_op(op)  -> true only for MUL_MAT, f16/int8 weights, S >= XDNA_MIN_PREFILL
graph_compute(cgraph)                    # intercept MUL_MAT, else fall through
```

## Hard parts (the months)
1. **int4 → AIE tiles.** colibrì/GGUF weights are int4-packed; XDNA2 GEMM wants
   int8 tiles in a specific layout. Need an unpack+repack that isn't slower than
   the matmul it feeds.
2. **Tile scheduling.** MLIR-AIE tiling of GEMM across the AIE array; correctness
   before speed.
3. **Threshold `XDNA_MIN_PREFILL`.** Below some S the DMA setup dominates — measure.
4. **Windows.** amdxdna is Linux-mainlined; Windows XDNA runtime is separate and
   newer. Bench on Linux first.

## Build path (when ready)
1. Prototype standalone: XRT + MLIR-AIE int8 GEMM, correctness vs CPU ref.
2. Wrap as ggml-backend, `supports_op` = MUL_MAT prefill only.
3. Wire into llama.cpp as `-DGGML_XDNA=ON`; router adds `device: npu-prefill` hint.
4. A/B TTFT vs iGPU-only prefill. Ship only if it wins TTFT without hurting gen.

## Decision gate
Build this only after the router (Phase 3) is carrying real traffic and TTFT is
the measured bottleneck. Until then it's speculative silicon.
