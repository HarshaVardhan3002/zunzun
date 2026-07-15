/* Unit test for intent profiles (profile.h): name validation, blend math,
 * blended record building, top-k pin-set diff. Pure logic, no I/O. */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include "profile.h"

int main(void){
    /* 1. name validation: [A-Za-z0-9_-], 1..32 chars */
    assert( prof_valid_name("coding"));
    assert( prof_valid_name("c-1_X"));
    assert(!prof_valid_name(""));
    assert(!prof_valid_name(NULL));
    assert(!prof_valid_name("has space"));
    assert(!prof_valid_name("dot.dot"));
    assert(!prof_valid_name("../evil"));
    assert(!prof_valid_name("123456789012345678901234567890123"));  /* 33 chars */
    assert( prof_valid_name("12345678901234567890123456789012"));   /* 32 chars */

    /* 2. blend math: g + w*p + s, saturating at UINT32_MAX */
    assert(prof_blend(10, 3, 2, 4) == 10 + 4*3 + 2);
    assert(prof_blend(0, 0, 0, 4) == 0);
    assert(prof_blend(0xffffffffu, 0xffffffffu, 5, 4) == 0xffffffffu);   /* saturates */

    /* 3. blended records from per-layer arrays (rows may be NULL; pbase/sess may be NULL) */
    uint32_t g0[4]={5,0,7,0}, p0[4]={0,2,1,0};
    uint32_t *gb[2]={g0,NULL}, *pb[2]={p0,NULL};
    PRec out[8];
    int n = prof_blend_recs(gb, pb, NULL, 2, 4, 4, out);
    assert(n==3);                                     /* e0:5  e1:8  e2:11 */
    assert(out[0].l==0 && out[0].e==0 && out[0].c==5);
    assert(out[1].l==0 && out[1].e==1 && out[1].c==8);      /* 0 + 4*2 + 0 */
    assert(out[2].l==0 && out[2].e==2 && out[2].c==11);     /* 7 + 4*1 + 0 */
    n = prof_blend_recs(gb, NULL, NULL, 2, 4, 4, out);      /* no profile: global only */
    assert(n==2 && out[0].c==5 && out[1].c==7);

    /* 4. top-k diff: top-2 of a={A,B,...} vs top-2 of b={A,C,...} -> 50% differ */
    PRec a[3]={{0,0,100},{0,1,90},{0,2,1}};
    PRec b[3]={{0,0,80},{0,3,70},{0,1,2}};
    assert(prof_topk_diff_pct(a,3,b,3,2)==50);   /* (0,1) in a's top-2, not in b's */
    PRec c1[2]={{0,0,9},{0,1,8}}, c2[2]={{0,1,9},{0,0,8}};
    assert(prof_topk_diff_pct(c1,2,c2,2,2)==0);  /* same set, different order */
    assert(prof_topk_diff_pct(c1,2,c2,2,0)==0);  /* k=0: no pins, no diff */

    printf("test_profile: PASS (names, blend+saturation, records, top-k diff)\n");
    return 0;
}
