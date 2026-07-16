/* test_sample: sampling + lossless speculative verification (Leviathan) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "../sample.h"

static void test_greedy(void){
    float lo[5]={1.f,9.f,3.f,-2.f,8.9f};
    uint64_t rng=42; Smp s; smp_init(&s,5,&rng);
    assert(smp_pick(&s,lo,0.0f,0.9f,-1)==1);      /* temp<=0 -> argmax */
    assert(smp_pick(&s,lo,0.0f,0.9f,1)==1);       /* ban ignored under greedy */
    smp_free(&s);
}

static void test_nucleus(void){
    /* softmax of {0,0,0,10} at temp 1: ~all mass on 3; nucleus .9 keeps only it */
    float lo[4]={0.f,0.f,0.f,10.f};
    uint64_t rng=42; Smp s; smp_init(&s,4,&rng);
    smp_dist(&s,lo,1.0f,0.9f);
    assert(s.p[3]>0.999f);
    assert(s.p[0]==0.f && s.p[1]==0.f && s.p[2]==0.f);
    smp_free(&s);
}

static void test_ban_renorm(void){
    float lo[3]={5.f,5.f,-30.f};                  /* p ~= {.5,.5,0} */
    uint64_t rng=7; Smp s; smp_init(&s,3,&rng);
    smp_dist(&s,lo,1.0f,1.0f);
    for(int i=0;i<1000;i++) assert(smp_sample(&s,0)!=0);   /* banned never sampled */
    smp_free(&s);
}

/* THE LEVIATHAN PROPERTY: for any draft token d,
 *   emit d with prob p[d]; otherwise resample from p with d banned (renormalized)
 * must reproduce exactly p. Empirical check, fixed seed, 3-sigma tolerance. */
static void test_leviathan_lossless(void){
    enum { V=6, N=400000 };
    float lo[V]={2.1f,0.3f,-1.0f,1.4f,0.0f,-0.5f};
    uint64_t rng=1234567; Smp s; smp_init(&s,V,&rng);
    smp_dist(&s,lo,0.7f,0.90f);                   /* engine defaults: temp .7, nucleus .90 */
    float target[V]; memcpy(target,s.p,sizeof target);
    for(int d=0;d<V;d++){                         /* every possible draft token */
        long cnt[V]={0};
        for(int i=0;i<N;i++){
            smp_dist(&s,lo,0.7f,0.90f);
            int tok = smp_accept_draft(&s,d) ? d : smp_sample(&s,d);
            cnt[tok]++;
        }
        for(int v=0;v<V;v++){
            double emp=(double)cnt[v]/N;
            double sig=sqrt(target[v]*(1.0-target[v])/N);
            double tol=3.0*sig+1e-4;
            if(fabs(emp-target[v])>tol){
                fprintf(stderr,"LEVIATHAN VIOLATION draft=%d tok=%d emp=%.5f target=%.5f tol=%.5f\n",
                        d,v,emp,target[v],tol);
                exit(1);
            }
        }
    }
    smp_free(&s);
}

int main(void){
    test_greedy(); test_nucleus(); test_ban_renorm(); test_leviathan_lossless();
    printf("test_sample: PASS (greedy, nucleus, ban-renorm, leviathan-lossless)\n");
    return 0;
}
