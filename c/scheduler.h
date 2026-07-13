/* Scheduler — ARCHITECTURE.md subsystem 3 (placement policy).
 *
 * Decides CPU vs GPU for one op from batch size, weight size and residency.
 * Pure function, no state -> unit-testable without hardware. Thresholds are
 * env-tunable and get pinned once G2/G3 are measured on the box:
 *   SCHED_SGPU  = batch S at/above which a matmul goes to GPU (prefill/MTP).      default 2
 *   SCHED_MINKB = min weight KB for an S=1 op to be worth the GPU dispatch cost.  default 4096
 *
 * Policy (from the audit's bandwidth reasoning): batched work is real GPU
 * compute -> GPU; single-token decode is a bandwidth-bound matvec whose GPU
 * dispatch overhead only pays when the weight is already resident AND large. */
#ifndef SCHED_H
#define SCHED_H
#include <stdlib.h>
#include <stdint.h>

typedef enum { SCHED_CPU = 0, SCHED_GPU = 1 } sched_dev;
typedef enum { SCHED_MATMUL = 0, SCHED_ATTN = 1, SCHED_DENSE = 2 } sched_op;

static inline int sched_sgpu(void){
    static int v = -1;
    if(v < 0){ const char *e = getenv("SCHED_SGPU"); v = e ? atoi(e) : 2; if(v < 1) v = 1; }
    return v;
}
static inline long sched_minkb(void){
    static long v = -1;
    if(v < 0){ const char *e = getenv("SCHED_MINKB"); v = e ? atol(e) : 4096; if(v < 0) v = 0; }
    return v;
}

/* gpu_ok       = a GPU backend is compiled and initialized.
 * gpu_resident = the weights are already on/mapped to the GPU (no upload cost). */
static inline sched_dev sched_place(sched_op op, int S, int64_t weight_bytes,
                                    int gpu_resident, int gpu_ok){
    (void)op;
    if(!gpu_ok) return SCHED_CPU;                              /* CPU-only build/run */
    if(S >= sched_sgpu()) return SCHED_GPU;                    /* prefill / MTP-verify batch */
    if(gpu_resident && weight_bytes >= sched_minkb() * 1024)   /* big resident S=1 op */
        return SCHED_GPU;
    return SCHED_CPU;                                          /* the S=1 decode tail */
}
#endif
