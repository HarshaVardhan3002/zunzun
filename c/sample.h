/* sample.h — token sampling + lossless speculative verification (Leviathan).
 * Pure logic, no I/O in the sampling paths (smp_init prints OOM and exits on
 * allocation failure), no model deps. Extracted verbatim from glm.c so the
 * accept/ban-resample math is unit-testable (tests/test_sample.c).
 * RNG is BORROWED by pointer: the engine keeps one global sequence so that
 * SEED=n runs stay byte-reproducible across this refactor. *rng must be
 * NONZERO: xorshift64 has an absorbing zero state (0 maps to 0 forever). */
#ifndef COLI_SAMPLE_H
#define COLI_SAMPLE_H
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    uint64_t *rng;      /* borrowed xorshift state */
    float *p;           /* target distribution buffer [V] */
    int   *idx;         /* nucleus sort scratch [V] */
    int    V;
} Smp;

static float *smp__p;   /* qsort ctx (no qsort_r on MinGW); decode is single-thread */
static int smp__cmp(const void *a,const void *b){
    float pa=smp__p[*(const int*)a], pb=smp__p[*(const int*)b];
    return pa<pb ? 1 : pa>pb ? -1 : 0; }

static inline void smp_init(Smp *s,int V,uint64_t *rng){
    s->V=V; s->rng=rng;
    s->p=malloc((size_t)V*sizeof(float));
    s->idx=malloc((size_t)V*sizeof(int));
    if(!s->p||!s->idx){fprintf(stderr,"OOM\n");exit(1);}
}
static inline void smp_free(Smp *s){ free(s->p); free(s->idx); s->p=NULL; s->idx=NULL; }

static inline double smp_rndu(Smp *s){
    uint64_t r=*s->rng; r^=r<<13; r^=r>>7; r^=r<<17; *s->rng=r;
    return (double)(r>>11)*(1.0/9007199254740992.0); }

static inline int smp_argmax(const float *lo,int V){
    int b=0; float bv=lo[0]; for(int i=1;i<V;i++) if(lo[i]>bv){bv=lo[i];b=i;} return b; }

/* target distribution: softmax(lo/temp) truncated to top-p nuc, renormalized */
static inline void smp_dist(Smp *s,const float *lo,float temp,float nuc){
    int V=s->V;
    float mx=lo[0]; for(int i=1;i<V;i++) if(lo[i]>mx) mx=lo[i];
    double sum=0; float invt=1.f/(temp>1e-4f?temp:1e-4f);
    for(int i=0;i<V;i++){ s->p[i]=expf((lo[i]-mx)*invt); sum+=s->p[i]; }
    for(int i=0;i<V;i++) s->p[i]/=(float)sum;
    if(nuc>0 && nuc<1.f){
        for(int i=0;i<V;i++) s->idx[i]=i;
        smp__p=s->p; qsort(s->idx,V,sizeof(int),smp__cmp);
        double cum=0; int keep=V;
        for(int i=0;i<V;i++){ cum+=s->p[s->idx[i]]; if(cum>=nuc){ keep=i+1; break; } }
        double s2=0; for(int i=keep;i<V;i++) s->p[s->idx[i]]=0;
        for(int i=0;i<keep;i++) s2+=s->p[s->idx[i]];
        for(int i=0;i<keep;i++) s->p[s->idx[i]]/=(float)s2;
    }
}
/* sample from s->p; ban>=0 -> excluded, renormalizing on the fly */
static inline int smp_sample(Smp *s,int ban){
    int V=s->V;
    double z = 1.0 - (ban>=0 ? s->p[ban] : 0.0); if(z<=1e-12) z=1e-12;
    double u = smp_rndu(s)*z, cum=0;
    for(int i=0;i<V;i++){ if(i==ban) continue; cum+=s->p[i]; if(cum>=u) return i; }
    for(int i=V-1;i>=0;i--) if(i!=ban && s->p[i]>0) return i;
    return 0;
}
/* next token from logits: greedy if temp<=0, else sample (ban = refused draft) */
static inline int smp_pick(Smp *s,const float *lo,float temp,float nuc,int ban){
    if(temp<=0) return smp_argmax(lo,s->V);
    smp_dist(s,lo,temp,nuc);
    return smp_sample(s,ban);
}
/* Leviathan accept for a DETERMINISTIC draft (q = point mass): accept w.p. p[draft].
 * Caller must have called smp_dist for THIS position first. */
static inline int smp_accept_draft(Smp *s,int draft){
    return smp_rndu(s) < s->p[draft];
}
#endif
