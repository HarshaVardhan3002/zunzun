#!/usr/bin/env bash
# GPU-only TPS benchmark for the 96 GB VRAM / 32 GB RAM BIOS split (2026-07-17).
# Same fixed prompt/seed as spec_probe.sh; results append to PROBES.md.
#   ./run_gpu96.sh            # all three probes, ~15-45 min total (cold cache)
#   ./run_gpu96.sh mirror     # just one: dense | mirror | direct
# RAM budget stays AUTO (auto = 88% of MemAvailable ~= 22 GB): do NOT pass
# RAM_GB=70 style overrides from the old 127.5/0.5 split — that OOMs this one.
set -e
cd "$(dirname "$0")"
run(){ tag="$1"; shift
  ../spec_probe.sh "$tag" "$@"
  # GPU tier + RAM plan lines land on stderr; fold them into PROBES.md too
  { grep -E "^\[CUDA\]|^\[PIN\]|^\[RAM_GB" "probe_${tag}.err" || true; echo; } >> PROBES.md
}
sel="${1:-all}"
# 1. dense-on-GPU only: experts all CPU — isolates the dense-tier speedup
if [ "$sel" = all ] || [ "$sel" = dense ]; then
  run gpu96_dense  COLI_HIP=1 CUDA_DENSE=1
fi
# 2. + expert mirrors: mirror cap 4 GB — that is all the pin the 32 GB RAM side
#    can host (mirrors upload FROM RAM pins; ~10.4 GB dense-host + ~6 GB slack
#    leave ~6 GB expert budget, AUTOPIN takes half). Bigger numbers do nothing.
if [ "$sel" = all ] || [ "$sel" = mirror ]; then
  run gpu96_mirror COLI_HIP=1 CUDA_DENSE=1 CUDA_EXPERT_GB=4
fi
# 3. + DIRECT I/O: with page cache this small, unbuffered expert reads may win
if [ "$sel" = all ] || [ "$sel" = direct ]; then
  run gpu96_direct COLI_HIP=1 CUDA_DENSE=1 CUDA_EXPERT_GB=4 DIRECT=1
fi
