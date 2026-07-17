/* Unit test for the KV-resume watermark (kvdisk.h): a .coli_kv tail written by
 * an NGEN-truncated turn must not be resumed (the "dangling half answer" bug).
 * The EOS token is never stored in hist, so completeness is recorded at append
 * time as a watermark (header word h[7]); resume trims to it. Pure logic, no I/O. */
#include <stdio.h>
#include <assert.h>
#include "kvdisk.h"

int main(void){
    /* 1. append bookkeeping: a complete turn advances the watermark to the new
     *    length; a truncated one leaves it at the last complete boundary */
    assert(kv_wm_append( 0, 120, 1) == 120);   /* first turn, ended via EOS */
    assert(kv_wm_append(120, 200, 0) == 120);  /* NGEN-truncated: wm stays */
    assert(kv_wm_append(120, 260, 1) == 260);  /* MORE finished the answer */
    assert(kv_wm_append( 0,  80, 0) == 0);     /* very first turn truncated */

    /* 2. truncate (API prefix mismatch / reset): wm can only shrink with it */
    assert(kv_wm_truncate(120, 200) == 120);   /* cut above wm: wm untouched */
    assert(kv_wm_truncate(120,  50) ==  50);   /* cut below wm: wm follows */
    assert(kv_wm_truncate(120,   0) ==   0);   /* reset */
    assert(kv_wm_truncate(120,  -3) ==   0);   /* garbage nrec clamps to 0 */

    /* 3. resume length: trim the dangling tail, tolerate corrupt headers */
    assert(kv_resume_len(200, 120) == 120);    /* half-turn tail -> trimmed */
    assert(kv_resume_len(200, 200) == 200);    /* clean file -> full resume */
    assert(kv_resume_len(200,   0) ==   0);    /* nothing complete -> fresh */
    assert(kv_resume_len(200, 999) == 200);    /* wm beyond nrec: clamp */
    assert(kv_resume_len(200,  -1) ==   0);    /* negative wm: clamp */
    assert(kv_resume_len( -5, 120) ==   0);    /* negative nrec: clamp */

    printf("test_kvdisk: PASS (append watermark, truncate, resume trim)\n");
    return 0;
}
