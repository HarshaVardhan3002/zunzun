/* Unit test for the placement policy (scheduler.h). Pure logic, no hardware. */
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "scheduler.h"

int main(void){
    const int64_t KB = 1024, MB = 1024*1024;
    /* no GPU available -> always CPU, whatever the size */
    assert(sched_place(SCHED_MATMUL, 64, 64*MB, 1, 0) == SCHED_CPU);
    /* batched (S >= SCHED_SGPU=2) -> GPU */
    assert(sched_place(SCHED_MATMUL, 2,   1*KB, 0, 1) == SCHED_GPU);
    assert(sched_place(SCHED_MATMUL, 64,  1*KB, 0, 1) == SCHED_GPU);
    /* S=1 small -> CPU (dispatch overhead beats a tiny matvec) */
    assert(sched_place(SCHED_MATMUL, 1,   1*KB, 1, 1) == SCHED_CPU);
    /* S=1 big AND resident -> GPU (weight already there, worth it) */
    assert(sched_place(SCHED_MATMUL, 1,   8*MB, 1, 1) == SCHED_GPU);
    /* S=1 big but NOT resident -> CPU (upload cost not amortized at S=1) */
    assert(sched_place(SCHED_MATMUL, 1,   8*MB, 0, 1) == SCHED_CPU);
    printf("test_sched: PASS (no-gpu->CPU, batch->GPU, S=1 tail->CPU, big+resident->GPU)\n");
    return 0;
}
