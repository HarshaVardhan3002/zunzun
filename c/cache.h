/* ExpertCache — ARCHITECTURE.md subsystem 2 (unified-memory expert pool).
 *
 * The per-layer resident set of streamed experts, on top of NVMe. R1 is a thin
 * FACADE over the pinned hot-store + LRU + working-set already carried on Model
 * (fields pin/npin, ecache/ecn, ws, ecap; heat lives in tier.h). Behaviour is
 * byte-identical to the pre-R1 inline lookup in moe(); this only gives the pool
 * a named seam so R2 can add a GPU view (dev_ptr / VRAM mirror) and R3 can move
 * the batch fetch+placement in here.
 *
 * CPU-only, host residency. Include AFTER the ESlot and Model typedefs. */
#ifndef CACHE_H
#define CACHE_H

typedef enum { EC_MISS=0, EC_HIT_PIN=1, EC_HIT_LRU=2 } ec_hit;

/* acquire (hit path): pinned hot-store first, then per-layer LRU. Returns the
 * resident slot or NULL on miss. Bumps LRU recency + hit counter, like before.
 * Miss FETCH stays batched in the caller for now (parallel expert_load); it
 * folds in here at R3. `how` may be NULL. */
static inline ESlot *ec_lookup(Model *m, int layer, int eid, ec_hit *how){
    ESlot *P=m->pin[layer];
    for(int z=0;z<m->npin[layer];z++)
        if(P[z].eid==eid){ m->hits++; if(how)*how=EC_HIT_PIN; return &P[z]; }
    ESlot *Sl=m->ecache[layer]; int nn=m->ecn[layer];
    for(int z=0;z<nn;z++)
        if(Sl[z].eid==eid){ m->hits++; Sl[z].used=++m->eclock; if(how)*how=EC_HIT_LRU; return &Sl[z]; }
    if(how)*how=EC_MISS;
    return NULL;
}

/* residency predicate (pin or LRU). Drives the next-block prefetch gate. */
static inline int ec_resident(Model *m, int layer, int eid){
    ESlot *P=m->pin[layer];
    for(int z=0;z<m->npin[layer];z++) if(P[z].eid==eid) return 1;
    ESlot *Sl=m->ecache[layer];
    for(int z=0;z<m->ecn[layer];z++) if(Sl[z].eid==eid) return 1;
    return 0;
}

/* promote: swap freshly-loaded working-set slots ws[0..nmiss) into the LRU,
 * evicting least-recently-used (min `used`). Swap (not free) keeps the slab
 * buffers alive for reuse. Identical to the old inline promotion block. */
static inline void ec_promote(Model *m, int layer, int nmiss){
    ESlot *Sl=m->ecache[layer]; int *nn=&m->ecn[layer];
    int promo = nmiss<m->ecap ? nmiss : m->ecap;
    for(int a=0;a<promo;a++){ int q=nmiss-1-a; ESlot *dst;
        if(*nn<m->ecap) dst=&Sl[(*nn)++];
        else { int lru=0; for(int z=1;z<*nn;z++) if(Sl[z].used<Sl[lru].used) lru=z; dst=&Sl[lru]; }
        ESlot tmp=*dst; *dst=m->ws[q]; m->ws[q]=tmp; dst->used=++m->eclock; }
}

#endif
