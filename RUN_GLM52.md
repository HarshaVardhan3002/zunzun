# Running GLM-5.2 (744B) on Zunzun — Strix Halo + DGX Spark

Pre-converted int4 model (colibrì container, identical format), no FP8 download / conversion needed. Tuned for
your box: Ryzen AI Max+ 395 (16C Zen 5, Radeon 8060S / RDNA 3.5 40 CU, 128 GB
LPDDR5X-8000, 80 W APU), and portable to an NVIDIA DGX Spark (arm64 Blackwell,
128 GB unified) with a one-flag toggle.

## 0. Pick the right model (MTP matters)
Two pre-converted model repos exist (published for upstream colibrì; the container is the same). **Use the int8-MTP one** — the other's MTP
head is int4 and speculative decoding won't fire (0-4% draft acceptance):

| repo | MTP head | decode |
|---|---|---|
| `jlnsrk/GLM-5.2-colibri-int4` | **int4 (unusable)** | no speculation |
| `mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp` | **int8** | ~2-2.8 tok/forward (recommended) |

If you already pulled jlnsrk, just overlay the int8 MTP shards:
`hf download mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp out-mtp-*.safetensors --local-dir /nvme/glm52_i4`

## 1. Download (~370 GB) to a FAST local disk
NVMe, native filesystem. NOT a network/9p mount, NOT the 180 GB F: drive, NOT a
WSL2 VHDX (its random reads cap ~1 GB/s and this model is I/O-bound).
```
hf download mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp --local-dir /nvme/glm52_i4
```
You have 414 GB free on C: — enough, but keep ~40 GB headroom.

## 2. Build — the AMD <-> NVIDIA toggle is one flag
The int4 container is identical on both machines; only the engine build changes.

| machine | build | CPU kernels | GPU |
|---|---|---|---|
| Strix Halo (AMD, x86) | `make HIP=1` | AVX-512 VNNI (Zen 5) | 8060S via ROCm/HIP |
| DGX Spark (NVIDIA, arm64) | `make CUDA=1` | ARM NEON (auto) | Blackwell via CUDA |
| either, CPU-only | `make` | native | none |

```
cd zunzun/c && ./setup.sh
make HIP=1        # Strix Halo   (ROCM_PATH=... if ROCm isn't at 7.1)
# make CUDA=1     # DGX Spark    (CUDA_ARCH=native picks Blackwell)
```
Same `zun` commands run on both after this. (DGX/arm64+CUDA path is wired via the
coli_gpu_* seam + NEON kernels; not yet run on DGX hardware — report back.)

## 3. Run
```
# CPU-only baseline (auto RAM budget, expert cache, MTP)
COLI_MODEL=/nvme/glm52_i4 ./zun chat

# iGPU accelerated (Strix): resident dense + hot experts mirrored to VRAM
COLI_HIP=1 CUDA_DENSE=1 CUDA_EXPERT_GB=24 COLI_MODEL=/nvme/glm52_i4 ./zun chat
# or let the planner size it (detects the 8060S via PyTorch-rocm — use your 3.12 interpreter):
COLI_MODEL=/nvme/glm52_i4 ./zun chat --auto-tier
```

## 4. The BIOS memory split — read this first (biggest lever)

Measured on this box: the BIOS carves **96 GB of the 128 GB as dedicated iGPU
VRAM**, so Windows sees only ~32 GB system RAM. This is backwards for Zunzun:
- the engine's speed comes from the **RAM** expert LRU cache; 32 GB gives only ~1-2
  slots/layer, so warm hit rate plateaus ~30-40% (not the 60-71% a big cache gives).
- On a unified APU the iGPU reaches system RAM via GTT anyway, and hot experts are
  mirrored into VRAM (`CUDA_EXPERT_GB`) — so a 96 GB VRAM carveout is oversized and
  starves the cache. `CUDA_EXPERT_GB=24` has headroom but is limited by RAM backing
  (an expert must be pinned in RAM before it mirrors to VRAM).

**Do this experiment (no code):** in BIOS/UEFI, shrink the iGPU "dedicated/UMA
VRAM" to ~**32-40 GB**. That frees ~64-96 GB for the RAM expert cache -> far higher
hit rate. Keep enough VRAM for the 9.3 GB resident dense + your `CUDA_EXPERT_GB`
mirror (so ~32 GB VRAM is plenty). This likely helps more than any tuning flag.

Measured today (current 96/32 split): cold 0.08 tok/s (1.2% hit), warming 0.29
tok/s (34% hit) — climbs as AUTOPIN learns. With a RAM-favoring split, expect the
warm number and hit rate to rise.

## 4b. Tuning for 40 CU / 80 W
| knob | value | why |
|---|---|---|
| RAM cache | auto (~88% of *system* RAM) | drives the warm hit rate — but see the BIOS split below, it is the #1 lever |
| `CUDA_EXPERT_GB` | 24-48 | mirror the hottest experts to the 8060S VRAM carveout (G1: iGPU reads VRAM at 226 vs host-GTT 125 GB/s, so mirror hot) |
| `--topp 0.7` | on | adaptive expert top-p, ~1.6x by cutting 30-40% of expert reads |
| MTP | int8 head (step 0) | ~2x decode via native speculation |
| `--repin N` | 256-512 | live re-pin hot experts every N tokens |
| AUTOPIN | warm it | run a few sessions; `.coli_usage` learns hot experts -> they pin to RAM + mirror to VRAM automatically |
| `OMP_PLACES=cores` | set | pin the 16 threads across the 2 CCDs cleanly |

80 W note: the APU shares 80 W across CPU+iGPU. Decode is I/O/bandwidth-bound, not
compute-bound, so don't chase GPU utilization — the R3 scheduler keeps tiny S=1
matmuls on CPU where GPU dispatch would only add power draw and latency.

## 5. Perf reality — OS matters a lot here
This is a 370 GB model streamed from NVMe; disk I/O dominates. The engine's async
prefetch (`PILOT`/`PREFETCH`), `mlock` pinning, and `O_DIRECT` all rely on POSIX.

- **Native Linux: fastest.** `posix_fadvise` readahead, `mlock`, `O_DIRECT` all work.
- **Native Windows (your MinGW build): runs, but I/O is unoptimized.** `fadvise`/
  `mlock`/`O_DIRECT` are no-ops, so prefetch is inert and streaming is slower. This
  is the un-built "Phase 2" (overlapped `ReadFileEx` prefetch + `VirtualLock` +
  `FILE_FLAG_NO_BUFFERING`). Biggest remaining Windows perf lever.
- **WSL2: avoid** for the model files (VHDX random-read cap ~1 GB/s).

Realistic warm throughput on this class: ~0.4-1.5 tok/s (it's a 744B model behind
a ~256 GB/s memory system + NVMe, not tens of tok/s).

## Sources
- https://hf.co/mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp (int8 MTP, recommended)
- https://hf.co/jlnsrk/GLM-5.2-colibri-int4 (int4 MTP)
