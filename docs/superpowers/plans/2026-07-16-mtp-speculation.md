# MTP Speculation (Bug Gate → Acceptance Recovery) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Root-cause the dropped-token quality bug, then recover MTP draft acceptance from 28% toward ≥60% so tokens/forward rises from 1.85 toward ≥3 — doubling warm throughput with zero extra bytes read.

**Architecture:** Extract the token-sampling + Leviathan rejection logic from `glm.c` into a new pure-logic header `c/sample.h` (repo pattern: `json.h`/`cache.h`/`profile.h` + paired test) so the speculation-verification math is unit-testable; add a per-depth acceptance histogram inside `spec_decode` (model-agnostic); drive all experiments through a probe script (`bench/spec_probe.sh`) that runs the engine's text mode with fixed prompt/seed. Wiring experiments use the existing `MTP_PRENORM`/`MTP_SWAP` env seams.

**Tech Stack:** C (single-header libs, MinGW/UCRT64 gcc), MSYS2 bash for builds, engine text mode (`PROMPT=` env) for on-box runs, GLM-5.2 744B int4 snapshot at `C:/Users/Von/Downloads/model`.

**Spec:** `docs/superpowers/specs/2026-07-16-mtp-speculation-design.md`

---

## Ground Truth (read before starting)

All line numbers are for `c/glm.c` at commit `835448c` (drift ±5 lines is normal — match on content).

- `spec_decode` (~1786–1845): the draft→verify loop. Greedy accept = exact argmax match (line ~1826); temp>0 accept = `rndu() < g_pbuf[draft[k]]` (Leviathan, ~1828); on reject `carry_ban=draft[k]`, resample next iteration via `pick_tok(logit,V,carry_ban)` (~1792).
- Sampling block (~1727–1765): `g_rng` xorshift + `rndu()`, `dist_build` (softmax/temp + nucleus top-p into `g_pbuf`), `dist_sample(V,ban)` (ban-token renormalized out), `pick_tok`.
- `mtp_draft` (~1597–1631) / `mtp_absorb` (~1635–1654): the MTP head chain. Env seams: `MTP_PRENORM` (skip the `final_norm` on h before `hnorm`), `MTP_SWAP` (concat `[h;emb]` instead of `[emb;h]`), `MTP_DEBUG=2` (per-draft stderr diagnostics).
- Defaults resolved in `main` (~2686–2744): `DRAFT=-1` → auto **3** when MTP present; `TEMP=-1` → auto **0.7** in text/chat; `NUCLEUS` default **0.90**; `SEED=<n>` makes sampling reproducible (`g_rng = n*0x9E3779B97F4A7C15ULL+1`), otherwise time-seeded.
- Text mode: `SNAP=<dir> PROMPT="..." NGEN=<n> ./glm.exe [cap]` — prints final stats incl. `speculation: X tokens/forward ... MTP acceptance Y%` (~1986).
- MTP auto-off guard (~1805): drafts disable below 10% acceptance after 24 proposals.
- Stats counters on `Model`: `n_emit`, `n_fw`, `mtp_prop`, `mtp_acc`.
- The MTP head is deliberately stored int8 (`zun` ~515: at int4 "i draft sbagliano quasi sempre") — head quantization sensitivity is a known effect.

### Build recipe (Windows, PowerShell tool)

```powershell
$env:TMP="$env:USERPROFILE\AppData\Local\Temp"; $env:MSYSTEM="UCRT64"
C:\msys64\usr\bin\bash.exe -lc "cd /c/Users/Von/Desktop/colibri/c && make glm && make test-c"
```

If `glm.exe` is locked ("Permission denied" at link), a chat/serve engine is still running — do NOT kill it; check with `Get-Process glm` and wait, or link to `glm_check.exe` (`make glm EXE_OUT=...` is not supported — instead run `gcc` line manually with `-o glm_check.exe`) just to typecheck.

### Run recipe (Bash tool — Git Bash runs the exe fine; builds do NOT work here)

```bash
cd /c/Users/Von/Desktop/colibri/c
SNAP="C:/Users/Von/Downloads/model" PIPE=1 SEED=1 NGEN=192 \
PROMPT='[gMASK]<sop><|user|>Write a Python function that finds the shortest path in a weighted graph using Dijkstra with heapq, then a small JS class Particle with update() and a scoring function. Output only code.<|assistant|><think></think>' \
./glm.exe 2>run.err | tee run.out
```

CPU-only (no `COLI_HIP`) for all probe/bisect runs — deterministic and comparable. Expect ~5–10 min per 192-token run warm. The GPU config is used only for the final real-chat confirmation (Task 11).

---

### Task 0: Branch

**Files:** none (git only)

- [ ] **Step 0.1:** `cd /c/Users/Von/Desktop/colibri && git checkout -b feature/mtp-speculation`
- [ ] **Step 0.2:** `mkdir -p bench/spec` (artifacts of every measured run live here, committed).

### Task 1: Phase 0 baselines — reproduce the bug, capture goldens

Four on-box runs with the **same prompt and SEED=1**. The two greedy runs are also the "golden" outputs used later to prove the `sample.h` refactor changed nothing.

**Files:**
- Create: `bench/spec/BISECT.md` (verdict log)
- Create: `bench/spec/run_A.out/.err`, `run_B.out/.err`, `run_D.out/.err`, `run_E.out/.err`

- [ ] **Step 1.1: Run A — as-shipped sampled (`TEMP=0.7 DRAFT=3`, i.e. pure defaults):** use the Run recipe verbatim, `tee bench/spec/run_A.out`, stderr to `run_A.err`.
- [ ] **Step 1.2: Run B — sampled, no speculation:** same command + `DRAFT=0`, outputs to `run_B.*`.
- [ ] **Step 1.3: Run D — greedy with drafts:** same + `TEMP=0` (keep `DRAFT` default 3), outputs to `run_D.*`.
- [ ] **Step 1.4: Run E — greedy, no drafts:** same + `TEMP=0 DRAFT=0`, outputs to `run_E.*`.
- [ ] **Step 1.4b: Run C — sampled, n-gram drafts only (run ONLY if A garbles and B is clean):** same command as A + `MTP=0` (drafts still on, but proposed by n-gram lookup instead of the MTP head), outputs to `run_C.*`. C clean → the MTP head's drafts are implicated specifically; C garbled → the verify/accept machinery is implicated regardless of draft source.
- [ ] **Step 1.5: Mechanical check — lossless greedy guarantee:**

```bash
md5sum bench/spec/run_D.out bench/spec/run_E.out
```

Expected: **identical hashes** (speculation is invisible under greedy). If they differ → the lossless invariant itself is broken; diff the outputs, find the first divergent token, and record it in `BISECT.md` — this becomes the primary lead for Task 4.

- [ ] **Step 1.6: Inspect A and B for garbling.** Look for the known pattern: dropped identifiers/operands (`this.x -= ;`, `score = ;`, `get(, ...)`). Note: A and B legitimately produce *different* text (rejection sampling consumes the RNG differently); the question per run is only "does its code have dropped tokens?", not "do they match?". Record a verdict table in `bench/spec/BISECT.md`:

```markdown
| run | config            | garbled? | evidence (quoted lines) |
|-----|-------------------|----------|-------------------------|
| A   | temp .7, draft 3  |          |                         |
| B   | temp .7, draft 0  |          |                         |
| C   | temp .7, MTP=0    | (only if A garbles, B clean) |     |
| D   | greedy, draft 3   |          |                         |
| E   | greedy, draft 0   |          |                         |
```

Interpretation guide: A garbles & B clean → speculation×sampling interaction guilty (Leviathan path prime suspect). A & B both garble → speculation innocent; it's sampling quality (temp/nucleus/int4) or emit. D/E garble → not sampling at all (emit/detok or model quality).

- [ ] **Step 1.7:** Also record from `run_A.out` stats: tokens/forward, MTP acceptance % (baseline numbers for Phase 2 comparison).
- [ ] **Step 1.8: Commit** `bench/spec/` artifacts: `git add bench/spec && git commit -m "bench(spec): phase-0 bisect baselines + golden greedy outputs"`.

### Task 2: `sample.h` — extract sampling into a tested header (TDD)

The statistical test below verifies the Leviathan property mechanically: speculative accept/reject+ban-resample must reproduce the target distribution exactly. If the test fails with logic copied verbatim from `glm.c`, **the Phase-0 bug is found here**.

**Files:**
- Create: `c/sample.h`
- Create: `c/tests/test_sample.c`
- Modify: `c/Makefile:56` (TEST_BINS) and add rule after `tests/test_profile$(EXE)` (~line 134)

- [ ] **Step 2.1: Write the failing test** `c/tests/test_sample.c`:

```c
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
```

- [ ] **Step 2.2: Makefile wiring.** In `c/Makefile`: append `tests/test_sample$(EXE)` to `TEST_BINS` (line 56) and add after the `test_profile` rule:

```make
tests/test_sample$(EXE): tests/test_sample.c sample.h
	$(CC) $(CFLAGS) -I. $< -o $@ $(LDFLAGS)
```

(Match the exact recipe style of the `tests/test_profile$(EXE)` rule above it — copy its compiler/flags line verbatim, changing only names.)

- [ ] **Step 2.3: Run to verify it fails:** build recipe → expected: `sample.h: No such file or directory`.
- [ ] **Step 2.4: Write `c/sample.h`** — logic copied **verbatim** from `glm.c` (same xorshift, same softmax/nucleus order, same ban renormalization), only re-shaped into a struct so tests can instantiate it:

```c
/* sample.h — token sampling + lossless speculative verification (Leviathan).
 * Pure logic, no I/O, no model deps. Extracted verbatim from glm.c so the
 * accept/ban-resample math is unit-testable (tests/test_sample.c).
 * RNG is BORROWED by pointer: the engine keeps one global sequence so that
 * SEED=n runs stay byte-reproducible across this refactor. */
#ifndef COLI_SAMPLE_H
#define COLI_SAMPLE_H
#include <stdint.h>
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
```

- [ ] **Step 2.5: Run the test:** build recipe → expected: `test_sample: PASS (...)`. **If `test_leviathan_lossless` fails**: the verbatim engine logic is mathematically wrong — this is the Phase-0 root cause. Stop, record the violation printout in `bench/spec/BISECT.md`, and carry it into Task 4.
- [ ] **Step 2.6: Commit:** `git add c/sample.h c/tests/test_sample.c c/Makefile && git commit -m "feat(sample): extract sampling+Leviathan verify into tested header"`.

### Task 3: Integrate `sample.h` into `glm.c` (byte-identical refactor)

**Files:**
- Modify: `c/glm.c` — sampling block ~1727–1765, accept site ~1826–1829, seed site ~2706, `#include` list (~line 39, after `grammar.h`)
- Modify: `c/Makefile` — add `sample.h` to the `glm$(EXE)` dependency list (same line that lists `profile.h`)

- [ ] **Step 3.1:** Add `#include "sample.h"` next to the other header-lib includes. Delete `rndu`, `g_pbuf`, `g_pidx`, `cmp_pdesc`, `dist_build`, `dist_sample` bodies; keep `g_rng` (seeded in `main`, unchanged) and add one engine-wide sampler:

```c
static uint64_t g_rng=0x9E3779B97F4A7C15ULL;      /* kept: seeded in main via SEED */
static Smp g_smp={0};
static void smp_ready(int V){ if(!g_smp.p) smp_init(&g_smp,V,&g_rng); }
```

- [ ] **Step 3.2:** Rewrite the two call sites (keep `pick_tok`'s signature so nothing else changes):

```c
static int pick_tok(const float *lo, int V, int ban){
    smp_ready(V);
    return smp_pick(&g_smp,lo,g_temp,g_nuc,ban);
}
```

and in `spec_decode`'s verify loop (~1826):

```c
int accept;
if(g_temp<=0) accept = (argmax_v(lo+(int64_t)k*V,V)==draft[k]);
else { smp_ready(V); smp_dist(&g_smp,lo+(int64_t)k*V,g_temp,g_nuc);
       accept = smp_accept_draft(&g_smp,draft[k]); }
```

`argmax_v` stays (used elsewhere); it may simply forward to `smp_argmax`.

- [ ] **Step 3.3: Build + unit tests:** build recipe → all tests PASS, `glm.exe` links.
- [ ] **Step 3.4: Byte-identical proof against Task-1 goldens.** Re-run Run A and Run D commands exactly (same `SEED=1`); then:

```bash
md5sum bench/spec/run_A.out run_A2.out bench/spec/run_D.out run_D2.out
```

Expected: A==A2 and D==D2 (same RNG sequence, same math ⇒ same bytes). If A2 differs, the refactor altered the RNG call order — find and fix before proceeding (diff stderr side by side).

- [ ] **Step 3.5: Commit:** `git commit -am "refactor(sample): glm.c uses sample.h; byte-identical vs golden runs"`.

### Task 4: Phase 0 verdict — root-cause and fix (decision task)

**Files:** depends on branch; verdict always recorded in `bench/spec/BISECT.md`.

- [ ] **Step 4.1:** Invoke the **superpowers:systematic-debugging** skill with the evidence from Tasks 1–3.
- [ ] **Step 4.2:** Follow the branch the evidence selected:
  - **Leviathan test failed (Task 2.5):** fix the math in `sample.h` minimally; the statistical test is the regression test. Re-run Task 3.4 (goldens will legitimately change for sampled runs — regenerate and note it).
  - **Greedy D≠E (Task 1.5):** lossless invariant broken outside sampling — instrument `spec_decode` around the first divergent position (`MTP_DEBUG=1`, print accepted `k`, `kv`, and `hlast` position ~1838) and inspect `mtp_absorb`'s position bookkeeping (~1836: `all+kv+1`, `pos_base=kv`). Fix; regression = greedy D/E MD5 equality added as a step in `bench/spec_probe.sh --check`.
    **AMENDMENT (Task 4 verdict, commit 16d8f72):** this branch was followed and the root cause is batched-vs-single forward FP numerics (I4S kernel S-gate + MoE batch-union accumulation order), NOT a bookkeeping bug — see `bench/spec/BISECT.md` "Phase 0 verdict". Bit-exact greedy D/E equality is therefore **unattainable by design** on this engine; do NOT add a greedy MD5-equality check to `spec_probe.sh --check`. Losslessness holds in distribution (guaranteed by `test_leviathan_lossless`); greedy divergences must be near-tie flips consistent with the verdict.
  - **A & B both garble (speculation innocent):** the bug is sampling quality or emit. Check emit first: `emit_stream` (~1852) decodes one token at a time into a 63-byte buffer — verify no token in the garbled region decodes to >63 bytes and that multi-byte UTF-8 sequences split across tokens survive `fputs`. If emit is clean, the verdict is *generation quality* (temp 0.7 / nucleus 0.90 / int4 noise): document, and test lower `NUCLEUS=0.85` as mitigation data only.
  - **Nothing reproduces:** garbling was environmental (e.g. the earlier serve-pipe CRLF class of bug already fixed). Document in `BISECT.md` with the four clean outputs as evidence; gate passes.
- [ ] **Step 4.3: Commit** fix + regression evidence: `git commit -am "fix(spec): <root cause> — phase-0 gate closed"`. **Do not start Task 5 until this task's verdict is written.**

### Task 5: Per-depth acceptance histogram (Phase 1 instrumentation)

Which draft position dies tells us *where* the head goes wrong (depth-1 failure = head wiring/quality; deep-tail failure = error compounding, expected).

**Files:**
- Modify: `c/glm.c` — file-scope counters near `g_stop` (~1769), verify loop (~1824–1835), stats print in `run_text` (~1986)

- [ ] **Step 5.1:** Add counters + report helper:

```c
/* acceptance per draft depth: [j] = drafts proposed/accepted at position j+1.
 * Model-agnostic: counts only the verify loop, no GLM specifics. */
static uint64_t g_dpos_prop[64], g_dpos_acc[64];
static void spec_depth_report(FILE *f){
    int last=-1; for(int j=0;j<64;j++) if(g_dpos_prop[j]) last=j;
    if(last<0) return;
    fprintf(f,"acceptance by depth:");
    for(int j=0;j<=last;j++)
        fprintf(f," d%d %.0f%% (%llu/%llu)", j+1,
            g_dpos_prop[j]?100.0*g_dpos_acc[j]/g_dpos_prop[j]:0.0,
            (unsigned long long)g_dpos_acc[j],(unsigned long long)g_dpos_prop[j]);
    fprintf(f,"\n");
}
```

- [ ] **Step 5.2:** In `spec_decode`, after the verify `while` loop (right before the `gsrc==1` bookkeeping at ~1834), count only genuinely examined positions:

```c
for(int j=0;j<k && j<64;j++){ g_dpos_prop[j]++; g_dpos_acc[j]++; }
if(k<g && k<64 && !done && emitted<n_new) g_dpos_prop[k]++;   /* the rejected one */
```

- [ ] **Step 5.3:** Call `spec_depth_report(stdout)` right after the `speculation: ...` printf in `run_text` (~1988) and `spec_depth_report(stderr)` at the same point in `run_serve`'s end-of-run stats (search for the matching `speculation:` print in `run_serve`; if none exists, add only the run_text one — YAGNI).
- [ ] **Step 5.4:** Build; smoke-run the Run recipe with `NGEN=64 TEMP=0`; expected new line like `acceptance by depth: d1 41% (18/44) d2 24% (...) d3 ...`.
- [ ] **Step 5.5: Commit:** `git commit -am "feat(spec): per-depth draft acceptance histogram in decode stats"`.

### Task 6: Probe script + baseline acceptance table

**Files:**
- Create: `bench/spec_probe.sh`
- Create: `bench/spec/PROBES.md` (running log of every probe result)

- [ ] **Step 6.1:** Write `bench/spec_probe.sh`:

```bash
#!/usr/bin/env bash
# Acceptance probe: fixed prompt, fixed seed, greedy, CPU-only. ~3 min warm.
#   ./spec_probe.sh <tag> [ENV=VAL ...]   e.g.: ./spec_probe.sh swap MTP_SWAP=1
# Appends the stats block to bench/spec/PROBES.md under the tag.
set -e
cd "$(dirname "$0")/../c"
TAG="${1:?usage: spec_probe.sh <tag> [ENV=VAL ...]}"; shift || true
OUT="../bench/spec/probe_${TAG}.out"; ERR="../bench/spec/probe_${TAG}.err"
env SNAP="${SNAP:-C:/Users/Von/Downloads/model}" PIPE=1 SEED=1 TEMP=0 NGEN=96 \
  PROMPT='[gMASK]<sop><|user|>Write a Python function that finds the shortest path in a weighted graph using Dijkstra with heapq. Output only code.<|assistant|><think></think>' \
  "$@" ./glm.exe >"$OUT" 2>"$ERR"
{ echo "## $TAG  ($(date +%F\ %T))  [$*]";
  grep -E "tokens in|speculation:|acceptance by depth" "$OUT"; echo; } >> ../bench/spec/PROBES.md
tail -n 4 "$OUT"
```

- [ ] **Step 6.2:** `chmod +x bench/spec_probe.sh`; run `./bench/spec_probe.sh base` → expect a `## base` block in `PROBES.md` with tokens/forward, MTP acceptance %, and the depth histogram. This is the Phase-2 reference row.
- [ ] **Step 6.3: Commit:** `git add bench/spec_probe.sh bench/spec && git commit -m "bench(spec): acceptance probe script + baseline table"`.

### Task 7: HF reference audit of the MTP head wiring

**Files:**
- Create: `bench/spec/WIRING-AUDIT.md`

- [ ] **Step 7.1:** Fetch the reference implementation. In order of preference: (a) the GLM-5.2 HF repo's modeling file (`hf_fs`/WebFetch on the model card's files, look for `modeling_*.py` MTP/`nextn` section), (b) vLLM's GLM/DeepSeek MTP module (`vllm/model_executor/models/deepseek_mtp.py` on GitHub), (c) DeepSeek-V3's HF modeling file. The comments in `glm.c` already cite "vLLM: h POST model.norm" — verify that claim against source.
- [ ] **Step 7.2:** Write `WIRING-AUDIT.md` as a two-column comparison (reference line ↔ `glm.c` line) covering exactly these seams, quoting real source both sides:
  1. Is `h` fed to the head **pre** or **post** the final `model.norm`? (glm.c: post, unless `MTP_PRENORM`; lines ~1610, ~1645)
  2. Concat order `[enorm(emb); hnorm(h)]` vs swapped (glm.c default emb-first; `MTP_SWAP` seam, ~1612, ~1647)
  3. Norm-weight assignment: `enorm` on embedding, `hnorm` on hidden — confirm the checkpoint tensor names map the same way (`model_init` ~816–841).
  4. **Chained drafting** (depth>1): reference behavior for feeding h' back (glm.c: `h=hx` raw pre-norm output, then `hnorm` next iteration but NOT `final_norm` for g≥1 — line ~1610's `g==0` condition. Is that right per reference?)
  5. Position/rope indices for the head's KV (`pos=p+g`, `kv_start` reset semantics, ~1599–1607).
  6. Verify-time absorb (`mtp_absorb`) uses the same formulation as draft-time (~1640–1649).
- [ ] **Step 7.3:** End the audit with a verdict list: `CONFIRMED-CORRECT / DISCREPANCY(fix=...) / UNKNOWN` per seam. Any DISCREPANCY becomes a candidate config for Task 8's sweep (via the env seams, or a small patch if not expressible).
- [ ] **Step 7.4: Commit:** `git add bench/spec/WIRING-AUDIT.md && git commit -m "docs(spec): MTP head wiring audit vs reference implementation"`.

### Task 8: Wiring sweep

**Files:**
- Modify: `bench/spec/PROBES.md` (results); possible small `c/glm.c` patch if the audit found a non-env-expressible fix

- [ ] **Step 8.1:** Run the 2×2 grid plus any audit candidates:

```bash
./bench/spec_probe.sh base2                      # repeat baseline (run-to-run sanity)
./bench/spec_probe.sh prenorm MTP_PRENORM=1
./bench/spec_probe.sh swap    MTP_SWAP=1
./bench/spec_probe.sh both    MTP_PRENORM=1 MTP_SWAP=1
# + one per audit DISCREPANCY, tagged audit1, audit2, ...
```

- [ ] **Step 8.2:** If every acceptance lands within noise of baseline, run one `MTP_DEBUG=2` probe (`./bench/spec_probe.sh dbg2 MTP_DEBUG=2`) and analyze the `[mtp2]` lines: `pre_blk` (head input already predicts token?) vs `post_blk` (after the MTP layer) vs verified token — this separates "eh_proj/concat wrong" (pre_blk garbage) from "MTP layer/KV wrong" (pre reasonable, post bad).
- [ ] **Step 8.3:** Record the winner in `PROBES.md` with a one-paragraph interpretation. Commit.

### Task 9: Adopt the winning wiring — or document the ceiling

**Files:**
- Modify: `c/glm.c` (`mtp_draft` ~1605–1613, `mtp_absorb` ~1640–1648) — only if a winner exists
- Create: `bench/spec/CEILING.md` — only if not

- [ ] **Step 9.1 (winner exists):** Make the winning formulation the *default code path* (delete the losing branch; keep `MTP_PRENORM`/`MTP_SWAP` seams only if they still select something meaningful, otherwise remove the getenv calls entirely — dead experiment flags are debt). Both `mtp_draft` and `mtp_absorb` must change identically.
- [ ] **Step 9.2 (winner exists):** Re-run `./bench/spec_probe.sh adopted` → acceptance must match the sweep's winning row. Run full `make test-c` → PASS.
- [ ] **Step 9.3 (no winner — acceptance still <60% everywhere):** Write `bench/spec/CEILING.md`: the sweep table, the `[mtp2]` pre/post analysis, and the int8-head quantization argument (`zun` ~515 already documents int4 head failure; extrapolate honestly). State the conclusion: single-chain MTP on this checkpoint tops out at X% — and per spec, tree speculation is dead too. Phase 3 then shrinks to Task 11's re-verification only.
- [ ] **Step 9.4: Commit** either way: `git commit -am "feat(mtp): adopt corrected head wiring (acceptance X%->Y%)"` or `"docs(spec): acceptance ceiling documented; wiring confirmed correct"`.

### Task 10: Draft depth tuning (only if Task 9 adopted a winner)

**Files:**
- Modify: `bench/spec/PROBES.md`

- [ ] **Step 10.1:**

```bash
./bench/spec_probe.sh d2 DRAFT=2
./bench/spec_probe.sh d4 DRAFT=4
./bench/spec_probe.sh d6 DRAFT=6
./bench/spec_probe.sh d8 DRAFT=8
```

Compare **tok/s** (not just acceptance — deeper drafts enlarge the batch-union expert set; the optimum is where marginal depth-k acceptance × 1 token no longer beats its extra expert loads). The depth histogram from Task 5 shows exactly where the chain dies.

- [ ] **Step 10.2:** Set the winner as the auto default at `glm.c` ~2744: `if(g_draft<0) g_draft = m.has_mtp ? <winner> : 0;` and update the comment at ~2689. Commit with the measured table in the message body.

### Task 11: Ship gates — lossless proof + real-session confirmation

**Files:**
- Modify: `run_glm52.sh` (comment line only, if defaults changed)
- Modify: `bench/spec/PROBES.md` (final numbers)

- [ ] **Step 11.1: Lossless greedy A/B (the invariant that must never break):**

```bash
cd /c/Users/Von/Desktop/colibri/c
SNAP="C:/Users/Von/Downloads/model" PIPE=1 TEMP=0 NGEN=128 SEED=1 DRAFT=0 PROMPT='[gMASK]<sop><|user|>Explain how a hash map handles collisions.<|assistant|><think></think>' ./glm.exe > ../bench/spec/final_nodraft.out 2>/dev/null
SNAP="C:/Users/Von/Downloads/model" PIPE=1 TEMP=0 NGEN=128 SEED=1 PROMPT='[gMASK]<sop><|user|>Explain how a hash map handles collisions.<|assistant|><think></think>' ./glm.exe > ../bench/spec/final_draft.out 2>/dev/null
md5sum ../bench/spec/final_nodraft.out ../bench/spec/final_draft.out
```

Expected (**amended per Task 4 verdict, commit 16d8f72**): bit-exact equality is NOT the gate — it is unattainable on this engine (batched-vs-single forward numerics; see BISECT.md "Phase 0 verdict"). The gate is: (a) both outputs are clean, coherent code/text with no dropped-token garbling; (b) if the text-region hashes (`tail -n +4`, stop at `---`) differ, re-run the draft config with `MTP_DEBUG=1` and confirm the first divergent position is a near-tie flip (`[gap]` line, small gap, same top-2 set) consistent with the verdict; record the outcome and gap numbers in PROBES.md.

- [ ] **Step 11.2: Real-session confirmation with the full GPU config:** one `./run_glm52.sh` chat turn (~300 tokens, a coding prompt), record tok/s, tok/forward, acceptance from the stderr heartbeat + endstats vs the 2026-07-15 baseline (0.49 tok/s, 1.85 tok/fw, 28%). Also eyeball the generated code for the Phase-0 garbling pattern — it must be gone (or match the Task-4 verdict).
- [ ] **Step 11.3:** `make test-c` PASS; update `run_glm52.sh`'s comment block if the `DRAFT` default changed. Commit.

### Task 12: Close out

- [ ] **Step 12.1:** Update `bench/spec/PROBES.md` with a final summary block: baseline → final acceptance, tok/forward, tok/s.
- [ ] **Step 12.2:** Update the project memory file (`strix-halo-target-hardware.md`): speculation project shipped/parked, headline numbers, pointer to `bench/spec/`.
- [ ] **Step 12.3:** Invoke **superpowers:finishing-a-development-branch** (merge to main locally, per this repo's convention; origin push remains a separate user decision).

---

## Verification Summary (what proves what)

| Claim | Proof |
|---|---|
| Rejection sampling is mathematically lossless | `test_sample.c::test_leviathan_lossless` (statistical, fixed seed) |
| `sample.h` refactor changed nothing | MD5 of re-runs vs Task-1 goldens (same SEED) |
| Speculation lossless in distribution (greedy bit-equality unattainable per Task 4 verdict) | `test_leviathan_lossless` + Task 11.1 amended gate (clean text; divergences = documented near-tie flips) |
| Bug root-caused | `bench/spec/BISECT.md` "Phase 0 verdict" (16d8f72): I4S kernel S-gate + MoE union order; H1 exonerated |
| Acceptance improvement is real | `bench/spec/PROBES.md` — same prompt/seed/config rows |
| Depth default is optimal | Task 10 tok/s table |
| End-to-end win | Task 11.2 real session vs 2026-07-15 baseline |
