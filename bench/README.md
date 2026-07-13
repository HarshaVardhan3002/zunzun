# R2 — HIP GPU backend (Radeon 8060S, gfx1151) + G1 gate

Not built in the dev sandbox (no ROCm/GPU). Build and run on the Strix Halo box.

## G1 — the gate (run this first)
Decides the whole design: can experts stay zero-copy in unified memory, or do
hot experts need a VRAM mirror? Measures iGPU read bandwidth, host-GTT vs VRAM.
```
hipcc -O3 bench/g1_bandwidth.cpp -o bench/g1
./bench/g1 2048
```
Read the verdict line. host-GTT >= ~70% of VRAM => zero-copy wins.

## HIP backend build
```
cd c && make HIP=1            # links backend_hip.o (hipcc, --offload-arch=gfx1151)
```
Raise the BIOS "variable graphics memory" / GTT so the iGPU can map a ~60-90 GB
expert pool. The backend exposes two modes (backend_hip.h):
- mirror: coli_hip_matmul (upload weight to VRAM, like CUDA) — for when G1 says host-GTT is slow.
- zero-copy: coli_hip_map(slab) once at load, then coli_hip_matmul_mapped — reads the expert in place, no VRAM copy.

Wiring these into the decode loop (device placement per size/residency) is R3.
