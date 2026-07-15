/* INTENT PROFILES: storie d'uso per dominio (.coli_usage.<nome>) fuse con la storia
 * globale al load; il pin score diventa globale + W*profilo + sessione.
 * EN: per-domain usage histories blended with the global one at load; pure logic only
 * (validation, blend, ranking diff) — file I/O stays in glm.c. */
#ifndef COLI_PROFILE_H
#define COLI_PROFILE_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* nome profilo valido: [A-Za-z0-9_-], 1..32 caratteri (diventa un suffisso di path:
 * niente separatori, niente dot). EN: valid profile name — becomes a path suffix. */
static inline int prof_valid_name(const char *s){
    if(!s || !*s) return 0;
    size_t n = strlen(s); if(n > 32) return 0;
    for(size_t i=0;i<n;i++){ char c=s[i];
        if(!((c>='A'&&c<='Z')||(c>='a'&&c<='z')||(c>='0'&&c<='9')||c=='_'||c=='-'))
            return 0;
    }
    return 1;
}

typedef struct { int l,e; uint32_t c; } PRec;   /* (layer, expert, blended count) */

/* punteggio fuso per un expert: globale + W*profilo + sessione, saturando a u32.
 * EN: blended score, saturating add. */
static inline uint32_t prof_blend(uint32_t g, uint32_t p, uint32_t s, uint32_t w){
    uint64_t v = (uint64_t)g + (uint64_t)w*p + s;
    return v > 0xffffffffu ? 0xffffffffu : (uint32_t)v;
}

/* costruisce la lista dei record fusi dalle matrici per-layer (righe NULL ammesse;
 * pbase/sess interi possono essere NULL). Ritorna quanti record scritti in out
 * (capienza richiesta: nrows*nexp). EN: build blended records; NULL rows/arrays ok. */
static inline int prof_blend_recs(uint32_t **gbase, uint32_t **pbase, uint32_t **sess,
                                  int nrows, int nexp, uint32_t w, PRec *out){
    int n=0;
    for(int l=0;l<nrows;l++){
        for(int e=0;e<nexp;e++){
            uint32_t g = (gbase && gbase[l]) ? gbase[l][e] : 0;
            uint32_t p = (pbase && pbase[l]) ? pbase[l][e] : 0;
            uint32_t s = (sess  && sess[l])  ? sess[l][e]  : 0;
            uint32_t b = prof_blend(g,p,s,w);
            if(b) out[n++] = (PRec){l,e,b};
        }
    }
    return n;
}

static int prof_rec_cmp_desc(const void *A, const void *B){
    const PRec *a=A, *b=B;
    return (a->c < b->c) - (a->c > b->c);
}

/* percentuale (0..100) dei top-k (l,e) di `a` ASSENTI dai top-k di `b`.
 * Ordina a e b in-place (decrescente). Serve alla riga "[PROFILE] pin set: N% differs".
 * EN: % of a's top-k pairs missing from b's top-k; sorts both in place. */
static inline int prof_topk_diff_pct(PRec *a, int na, PRec *b, int nb, int k){
    if(k<=0 || na<=0 || nb<=0) return 0;
    qsort(a,(size_t)na,sizeof(PRec),prof_rec_cmp_desc);
    qsort(b,(size_t)nb,sizeof(PRec),prof_rec_cmp_desc);
    if(k>na) k=na; if(k>nb) k=nb;
    int diff=0;
    for(int i=0;i<k;i++){
        int found=0;
        for(int j=0;j<k;j++) if(a[i].l==b[j].l && a[i].e==b[j].e){ found=1; break; }
        if(!found) diff++;
    }
    return diff*100/k;
}
#endif
