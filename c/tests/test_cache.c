/* Unit test for the ExpertCache facade (cache.h). Uses minimal stand-in ESlot
 * and Model with only the fields the facade touches — a separate TU, so it does
 * not need the full glm.c type context. Verifies the policy, not the I/O. */
#include <stdio.h>
#include <stdint.h>
#include <assert.h>

typedef struct { int eid; uint64_t used; } ESlot;      /* subset the facade reads */
typedef struct {
    ESlot **pin;  int *npin;
    ESlot **ecache; int *ecn;
    ESlot ws[64];
    int ecap;
    uint64_t hits, eclock;
} Model;

#include "cache.h"

int main(void){
    ESlot pin0[1]  = {{ .eid=5 }};
    ESlot lru0[2]  = {{ .eid=1, .used=10 }, { .eid=2, .used=20 }};
    ESlot *pinp[1] = { pin0 };   int npin[1] = { 1 };
    ESlot *lrup[1] = { lru0 };   int ecn[1]  = { 2 };
    Model m = {0};
    m.pin=pinp; m.npin=npin; m.ecache=lrup; m.ecn=ecn; m.ecap=2;

    ec_hit how;
    /* 1. pinned hit */
    assert(ec_lookup(&m,0,5,&how)==&pin0[0] && how==EC_HIT_PIN && m.hits==1);
    /* 2. LRU hit bumps recency */
    ESlot *r = ec_lookup(&m,0,1,&how);
    assert(r==&lru0[0] && how==EC_HIT_LRU && r->used==1 && m.eclock==1 && m.hits==2);
    /* 3. miss */
    assert(ec_lookup(&m,0,99,&how)==NULL && how==EC_MISS);
    /* 4. residency */
    assert(ec_resident(&m,0,5)==1 && ec_resident(&m,0,2)==1 && ec_resident(&m,0,99)==0);
    /* 5. promote into a full cache evicts min-`used` (eid=1, used=1) and swaps ws out */
    m.ws[0].eid=7;
    ec_promote(&m,0,1);
    assert(m.ecache[0][0].eid==7);      /* freshly promoted */
    assert(m.ws[0].eid==1);             /* evicted slot swapped back into ws (buffer reuse) */
    assert(m.ecn[0]==2);                /* still full, no growth */

    printf("test_cache: PASS (pin>LRU, recency bump, min-used eviction, swap-not-free)\n");
    return 0;
}
