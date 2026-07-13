# Intent Profiles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `PROFILE=coding` loads a per-domain expert-usage history alongside the global one, pins the blended hot set (`global + 4×profile`), and saves session selections into both files.

**Architecture:** All engine logic sits at the existing AUTOPIN seam near the end of `main` in `c/glm.c`. `m->eusage` changes meaning from "loaded history + live counters" to **session deltas only**; two new baseline arrays (`gbase`, `pbase`) hold what was loaded from disk, so each file can be saved as `baseline + deltas` without double-counting. Pure logic (name validation, blending, top-k overlap) lives in a new single-header module `c/profile.h` with its own test TU, following the `cache.h`/`test_cache.c` pattern.

**Tech Stack:** C99 single-header modules, MinGW/MSYS2 `make` on Windows, Python `coli` CLI (argparse).

**Spec:** `docs/superpowers/specs/2026-07-13-intent-profiles-design.md`

**Deliberately untouched:** `c/glm_baseline.c` is the frozen A/B baseline — do not mirror any of these changes into it.

---

## File structure

| File | Change | Responsibility |
|---|---|---|
| `c/profile.h` | Create | Pure logic: profile-name validation, blend math, blended record list, top-k pin-set diff |
| `c/tests/test_profile.c` | Create | Unit tests for everything in `profile.h` |
| `c/Makefile` | Modify | Build rule for the test; add `profile.h` to `glm$(EXE)` deps; extend `TEST_BINS` |
| `c/glm.c` | Modify | Model fields `gbase`/`pbase`; `usage_load_into`; dual-destination `usage_save`; `pin_load` split; `PROFILE` wiring in the AUTOPIN block |
| `c/coli` | Modify | `--profile` flag → `PROFILE` env; pass `[PROFILE]` stderr lines through to the UI |
| `run_glm52.sh` | Modify | One comment line documenting `PROFILE=name` |

---

### Task 1: `profile.h` pure-logic module (TDD)

**Files:**
- Create: `c/profile.h`
- Test: `c/tests/test_profile.c`
- Modify: `c/Makefile:56` (TEST_BINS), after `c/Makefile:132` (build rule)

- [ ] **Step 1: Write the failing test**

Create `c/tests/test_profile.c`:

```c
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
```

- [ ] **Step 2: Add Makefile entries**

In `c/Makefile` line 56, extend `TEST_BINS`:

```make
TEST_BINS = tests/test_json$(EXE) tests/test_st$(EXE) tests/test_tier$(EXE) tests/test_grammar$(EXE) tests/test_cache$(EXE) tests/test_sched$(EXE) tests/test_profile$(EXE)
```

After the `tests/test_sched$(EXE)` rule (line 131-132), add:

```make
tests/test_profile$(EXE): tests/test_profile.c profile.h
	$(CC) $(CFLAGS) -I. $< -o $@ $(LDFLAGS)
```

Also add `profile.h` to the `glm$(EXE)` dependency list (line 90), so engine rebuilds pick up header changes:

```make
glm$(EXE): glm.c st.h json.h tok.h tok_unicode.h compat.h grammar.h model.h cache.h sched.h backend.h profile.h $(CUDA_OBJ) $(HIP_OBJ)
```

- [ ] **Step 3: Run test to verify it fails**

Run (from `c/`): `make tests/test_profile.exe`
Expected: FAIL — `profile.h: No such file or directory`

- [ ] **Step 4: Write the implementation**

Create `c/profile.h`:

```c
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
```

- [ ] **Step 5: Run test to verify it passes**

Run (from `c/`): `make tests/test_profile.exe && ./tests/test_profile.exe`
Expected: `test_profile: PASS (names, blend+saturation, records, top-k diff)`

- [ ] **Step 6: Run the whole C suite to check nothing broke**

Run (from `c/`): `make test-c`
Expected: every test prints PASS, exit 0.

- [ ] **Step 7: Commit**

```bash
git add c/profile.h c/tests/test_profile.c c/Makefile
git commit -m "feat(profile): pure-logic module for intent profiles (blend, validation, top-k diff)"
```

---

### Task 2: `glm.c` — baseline arrays and dual-destination saves

The behavioral contract of this task: **with `PROFILE` unset, behavior is identical to today.** `eusage` becomes session-deltas; the global file is saved as `gbase + eusage`, which equals exactly what the old code saved (it loaded history *into* `eusage` and dumped it back).

**Files:**
- Modify: `c/glm.c:130` (Model struct), `c/glm.c:762` (model_init), `c/glm.c:2345-2370` (dump/load/save), `c/glm.c:2724-2743` (AUTOPIN block, load call only — the rest changes in Task 4)

- [ ] **Step 1: Include the header and extend Model**

Near the other local includes at the top of `glm.c` (where `cache.h` is included), add:

```c
#include "profile.h"
```

At `c/glm.c:130`, replace:

```c
    uint32_t **eusage;                           /* contatori persistenti (per STATS/PIN) */
```

with:

```c
    uint32_t **eusage;                           /* SOLO delta di sessione (per STATS/PIN) / session deltas only */
    uint32_t **gbase, **pbase;                   /* storia caricata: globale e profilo / loaded history: global, profile */
```

- [ ] **Step 2: Allocate the row-pointer arrays in model_init**

At `c/glm.c:762`, extend:

```c
    m->eusage=calloc(NR,sizeof(uint32_t*)); m->eheat=calloc(NR,sizeof(uint32_t*));
```

to:

```c
    m->eusage=calloc(NR,sizeof(uint32_t*)); m->eheat=calloc(NR,sizeof(uint32_t*));
    m->gbase=calloc(NR,sizeof(uint32_t*));  m->pbase=calloc(NR,sizeof(uint32_t*));
```

(`Model m;` at glm.c:2710 is uninitialized stack memory — these fields MUST be assigned here, like every other field.)

- [ ] **Step 3: Generalize the dump to "baselines + deltas"**

Replace `stats_dump_q` (glm.c:2348-2356) with a two-baseline version and keep the old entry points as thin wrappers:

```c
/* scarica su file l'istogramma: righe "layer eid count" con count = b1 + b2 + delta
 * di sessione (saturando a u32). b1/b2 possono essere NULL. Scrittura atomica.
 * EN: dump histogram as baseline(s) + session deltas, saturating; atomic write. */
static void stats_dump2(Model *m, const char *path, uint32_t **b1, uint32_t **b2, int quiet){
    char tmp[2100]; snprintf(tmp,sizeof(tmp),"%s.tmp",path);
    FILE *f=fopen(tmp,"w"); if(!f){ if(!quiet) perror(tmp); return; }
    Cfg *c=&m->c; int64_t tot=0, nz=0;
    for(int i=0;i<=c->n_layers;i++){ if(!m->eusage[i]) continue;
        for(int e=0;e<c->n_experts;e++){
            uint32_t v = prof_blend(b1&&b1[i]?b1[i][e]:0, b2&&b2[i]?b2[i][e]:0,
                                    m->eusage[i][e], 1);
            if(v){ fprintf(f,"%d %d %u\n",i,e,v); tot+=v; nz++; }
        } }
    fclose(f); rename(tmp,path);
    if(!quiet) fprintf(stderr,"[STATS] %lld selections across %lld distinct experts -> %s\n",(long long)tot,(long long)nz,path);
}
/* STATS diagnostico: la vista fusa completa (globale + profilo + sessione) */
static void stats_dump(Model *m, const char *path){ stats_dump2(m,path,m->gbase,m->pbase,0); }
```

(Note: `prof_blend(g,p,s,1)` is exactly `g + p + s` saturating — reused here so the file can never wrap a u32.)

- [ ] **Step 4: Load into a baseline instead of into eusage**

Replace `usage_load` (glm.c:2363-2369) with:

```c
static char g_usage_path[2100]="", g_profile_path[2100]="";
static uint32_t g_profile_w=4;                   /* peso del profilo nel blend / profile weight */
/* carica una storia in una matrice baseline (righe allocate al bisogno, solo dove
 * esiste la riga eusage). EN: load one history file into a baseline array. */
static int64_t usage_load_into(Model *m, const char *path, uint32_t **dst){
    FILE *f=fopen(path,"r"); if(!f) return 0;
    Cfg *c=&m->c; int l,e; uint32_t cnt; int64_t tot=0;
    while(fscanf(f,"%d %d %u",&l,&e,&cnt)==3)
        if(l>=0&&l<=c->n_layers&&e>=0&&e<c->n_experts&&m->eusage[l]){
            if(!dst[l]) dst[l]=calloc(c->n_experts,sizeof(uint32_t));
            dst[l][e]+=cnt; tot+=cnt;
        }
    fclose(f); return tot;
}
```

(The old `g_usage_path` declaration at glm.c:2362 is replaced by the two-path line above.)

- [ ] **Step 5: Save to both destinations**

Replace `usage_save` (glm.c:2370) with:

```c
/* salva a ogni turno: globale = gbase+delta, profilo = pbase+delta. Idempotente:
 * le baseline non si ri-sommano mai a se stesse. EN: per-turn dual save, idempotent. */
static void usage_save(Model *m){
    if(g_usage_path[0])   stats_dump2(m, g_usage_path,   m->gbase, NULL, 1);
    if(g_profile_path[0]) stats_dump2(m, g_profile_path, m->pbase, NULL, 1);
}
```

- [ ] **Step 6: Update the AUTOPIN block's load call (minimal, no PROFILE yet)**

At glm.c:2730, replace:

```c
      int64_t hist = usage_load(&m,g_usage_path);
```

with:

```c
      int64_t hist = usage_load_into(&m,g_usage_path,m.gbase);
```

**Temporary regression in this task, fixed in Task 4:** the `pin_load(&m, g_usage_path, pin_gb)` call three lines down still reads the global *file*, which now equals `gbase` (deltas are zero at startup) — so AUTOPIN behavior is unchanged and the build stays green between tasks.

- [ ] **Step 7: Build and run the suite**

Run (from `c/`): `make glm.exe HIP=1 && make test-c`
Expected: clean build (warnings-as-usual), all tests PASS.
If `hipcc` is unavailable in this shell, `make glm.exe` (CPU build) is an acceptable compile check — the touched code has no GPU dependency.

- [ ] **Step 8: Commit**

```bash
git add c/glm.c
git commit -m "refactor(usage): split loaded history (gbase/pbase) from session deltas; dual-destination saves"
```

---

### Task 3: `glm.c` — split `pin_load` into parse + `pin_load_recs`

Pure refactor: the file-parsing front half stays in `pin_load`; the ranking/loading/GPU-upload/wire back half becomes `pin_load_recs`, callable with in-memory blended records. No behavior change.

**Files:**
- Modify: `c/glm.c:2420-2509` (`pin_load`)

- [ ] **Step 1: Restructure pin_load**

Replace the beginning of `pin_load` (glm.c:2420-2439, up to and including the `if(npin<1)` line) with:

```c
/* meta' "ranking+load" del pin: prende record (layer,eid,count) GIA' validati,
 * ordina per frequenza, carica i top entro il budget. NON libera r (il chiamante
 * lo riusa per la diff del pin set). Ritorna npin. EN: ranking+loading half of the
 * pin; does NOT free r; returns how many experts were pinned. */
static int pin_load_recs(Model *m, PRec *r, int n, double gb, const char *src){
    Cfg *c=&m->c;
    for(int a=0;a<n;a++){ int best=a;                       /* selection sort parziale, poi taglio */
        for(int b=a+1;b<n;b++) if(r[b].c>r[best].c) best=b;
        PRec t=r[a]; r[a]=r[best]; r[best]=t;
        if(a>4095) break;                                    /* bastano i top ~4k */
    }
    int64_t eb=expert_bytes_probe(m,m->ebits);
    int npin=(int)(gb*1e9/eb); if(npin>n) npin=n; if(npin>4096) npin=4096;
    if(npin<1) return 0;
```

The rest of the old body (from `int *cnt_l=calloc(...)` at glm.c:2440 through `pin_wire(m);` at glm.c:2507) stays inside `pin_load_recs` **unchanged**, except:
- every reference to the old local `Rec` type is now `PRec` (same fields `l`, `e`, `c` — the `#ifdef COLI_GPU` block at glm.c:2454-2506 already only uses `r[a].l` and `r[a].e`, so it compiles as-is),
- the final line `free(r); free(cnt_l);` becomes `free(cnt_l); return npin;`,
- the `[PIN]` message keeps its format, printing `src` where it printed `statspath`.

Then the file-parsing wrapper (replacing the old parse front half) becomes:

```c
static void pin_load(Model *m, const char *statspath, double gb){
    FILE *f=fopen(statspath,"r"); if(!f){ perror(statspath); return; }
    Cfg *c=&m->c; int cap=(c->n_layers+1)*c->n_experts;
    PRec *r=malloc((size_t)cap*sizeof(PRec)); int n=0;
    int l,e; uint32_t cnt;
    while(n<cap && fscanf(f,"%d %d %u",&l,&e,&cnt)==3){
        int ok = l>=0 && e>=0 && e<c->n_experts &&
                 ((l<c->n_layers && m->L[l].sparse) || (l==c->n_layers && m->has_mtp));
        if(ok) r[n++]=(PRec){l,e,cnt};
    }
    fclose(f);
    pin_load_recs(m,r,n,gb,statspath);
    free(r);
}
```

(Also delete the old local `typedef struct { int l,e; uint32_t c; } Rec;` — `PRec` from profile.h replaces it.)

- [ ] **Step 2: Build and run the suite**

Run (from `c/`): `make glm.exe HIP=1 && make test-c`
Expected: clean build, all tests PASS.

- [ ] **Step 3: Commit**

```bash
git add c/glm.c
git commit -m "refactor(pin): split pin_load into file parse + pin_load_recs (in-memory records)"
```

---

### Task 4: `glm.c` — PROFILE wiring in the AUTOPIN block

**Files:**
- Modify: `c/glm.c:2724-2743` (the AUTOPIN block in `main`)

- [ ] **Step 1: Replace the AUTOPIN block**

Replace the whole block from the comment at glm.c:2724 (`/* CACHE CHE IMPARA: ... */`) through the closing `cap_for_ram(&m, ram_env, ebits, est_ctx); }` at glm.c:2743 with:

```c
    /* CACHE CHE IMPARA: l'uso degli expert si accumula in <SNAP>/.coli_usage tra le sessioni;
     * all'avvio i piu' usati vengono auto-pinnati in RAM (meta' del budget expert: il pin
     * conosce la TUA storia, la LRU si adatta alla sessione). AUTOPIN=0 disattiva.
     * INTENT PROFILES: PROFILE=<nome> fonde <SNAP>/.coli_usage.<nome> con la storia globale
     * (score = globale + W*profilo, PROFILE_W=4); le selezioni nuove nutrono ENTRAMBI i file.
     * EN: PROFILE=<name> blends a per-domain history into the pin ranking; new selections
     * feed both files. */
    { double ram_env = getenv("RAM_GB")?atof(getenv("RAM_GB")):0.0;
      int est_ctx = getenv("CTX")?atoi(getenv("CTX")):4096;   /* stesso default di run_serve */
      snprintf(g_usage_path,sizeof(g_usage_path),"%s/.coli_usage",snap);
      int64_t ghist = usage_load_into(&m,g_usage_path,m.gbase), phist = 0;
      const char *prof = getenv("PROFILE");
      if(prof && !*prof) prof=NULL;
      if(prof && !prof_valid_name(prof)){
          fprintf(stderr,"[PROFILE] invalid name '%s' ([A-Za-z0-9_-], max 32 chars); running without a profile\n",prof);
          prof=NULL;
      }
      if(prof){
          if(getenv("PROFILE_W")){ int w=atoi(getenv("PROFILE_W")); g_profile_w = w<1?1:(uint32_t)w; }
          snprintf(g_profile_path,sizeof(g_profile_path),"%s/.coli_usage.%s",snap,prof);
          phist = usage_load_into(&m,g_profile_path,m.pbase);
          if(phist>0)
              fprintf(stderr,"[PROFILE] %s: %lld selections (+ %lld global, weight %ux)\n",
                      prof,(long long)phist,(long long)ghist,g_profile_w);
          else
              fprintf(stderr,"[PROFILE] %s: new profile (pin set seeded from %lld global selections)\n",
                      prof,(long long)ghist);
      }
      int64_t hist = ghist + phist;
      if(ghist>0) fprintf(stderr,"[USAGE] expert history: %lld selections (%s)\n",(long long)ghist,g_usage_path);
      int autopin = getenv("AUTOPIN")?atoi(getenv("AUTOPIN")):1;
      if(!getenv("PIN") && autopin && hist>=5000){
          /* quota pin proporzionale alla FIDUCIA nella storia: con pochi dati il pin
           * sbaglia expert e ruba slot alla LRU adattiva; a regime (>=200k selezioni,
           * qualche ora di chat) arriva a meta' del budget expert. */
          double conf = (double)hist/200000.0; if(conf>1) conf=1;
          double pin_gb = expert_avail(&m,ram_env,ebits,est_ctx)*0.5*conf/1e9;
          if(pin_gb>=0.5){
              Cfg *c=&m.c; int cap_r=(c->n_layers+1)*c->n_experts;
              PRec *rb=malloc((size_t)cap_r*sizeof(PRec));
              int nb=prof_blend_recs(m.gbase,m.pbase,m.eusage,c->n_layers+1,c->n_experts,g_profile_w,rb);
              int npin=pin_load_recs(&m,rb,nb,pin_gb,prof?prof:g_usage_path);
              if(prof && npin>0){
                  /* quanto il profilo cambia davvero il pin set rispetto al solo globale */
                  PRec *rg=malloc((size_t)cap_r*sizeof(PRec));
                  int ng=prof_blend_recs(m.gbase,NULL,NULL,c->n_layers+1,c->n_experts,1,rg);
                  fprintf(stderr,"[PROFILE] pin set: %d%% differs from global-only\n",
                          prof_topk_diff_pct(rb,nb,rg,ng,npin));
                  free(rg);
              }
              free(rb);
          }
      }
      /* SEMPRE: senza clamp la LRU cresce fino a cap*76 layer = decine di GB -> OOM-kill.
       * RAM_GB assente o <=0 = budget automatico da MemAvailable. */
      cap_for_ram(&m, ram_env, ebits, est_ctx); }
```

Notes for the implementer:
- `m.eusage` is passed to `prof_blend_recs` for correctness-by-construction (it is all zeros at startup, so it contributes nothing here).
- Dense layers have no `eusage` row, so `usage_load_into` never allocates baseline rows for them and `prof_blend_recs` skips them — the validity filter the old file-parse applied (`L[l].sparse || MTP`) is preserved structurally.
- `pin_load_recs` partially sorts `rb` in place; `prof_topk_diff_pct` fully re-sorts both arrays with `qsort`, so passing the already-touched `rb` is fine.

- [ ] **Step 2: Build and run the suite**

Run (from `c/`): `make glm.exe HIP=1 && make test-c`
Expected: clean build, all tests PASS.

- [ ] **Step 3: Commit**

```bash
git add c/glm.c
git commit -m "feat(profile): PROFILE=<name> blends per-domain usage history into the pin set"
```

---

### Task 5: CLI and launcher surface

**Files:**
- Modify: `c/coli:19` (help text), `c/coli:141` (env_for), `c/coli:387` (stderr filter), `c/coli:525` (argparse)
- Modify: `run_glm52.sh:14`

- [ ] **Step 1: coli — flag, env, help, stderr passthrough**

At `c/coli:525`, next to the `--repin` line, add:

```python
    common.add_argument("--profile", default=None, help="intent profile: per-domain expert history, e.g. coding")
```

At `c/coli:141`, next to `if a.repin:`, add:

```python
    if a.profile: e["PROFILE"]=a.profile
```

In the usage/help text near `c/coli:19` (`--repin N ...`), add the matching line:

```
  --profile NAME     intent profile: per-domain expert history (e.g. coding)
```

At `c/coli:387`, add `"[PROFILE]"` to the startswith tuple so the engine's profile lines reach the user:

```python
            if l.startswith(("[RAM_GB","[PIN]","[MTP]","[USAGE]","[DSA]","[KV]","[PROFILE]")):
```

- [ ] **Step 2: run_glm52.sh — document the env passthrough**

After the `export PIPE=...` line (run_glm52.sh:14), add:

```bash
# PROFILE=coding ./run_glm52.sh — per-domain expert history: pre-pins YOUR coding hot set (any [A-Za-z0-9_-] name)
```

(No export needed — `PROFILE` flows through the environment to `coli` and on to `glm.exe`.)

- [ ] **Step 3: Sanity-check the CLI parses**

Run (from `c/`): `python coli chat --help`
Expected: help prints, `--profile` listed, exit 0.

- [ ] **Step 4: Commit**

```bash
git add c/coli run_glm52.sh
git commit -m "feat(cli): --profile flag and PROFILE passthrough for intent profiles"
```

---

### Task 6: Integration verification on the box (real model)

No code — evidence gathering. Model dir: `C:/Users/Von/Downloads/model`. Run from `c/` in an MSYS2/bash shell. These runs load ~10+ GB and take minutes; run them sequentially, never in parallel.

- [ ] **Step 1: Baseline unchanged without PROFILE**

```bash
SNAP=C:/Users/Von/Downloads/model TEMP=0 NGEN=24 PROMPT="Write a haiku about caches." ./glm.exe 2>base.err >base.out
grep -E '\[USAGE\]|\[PIN\]' base.err
```

Expected: `[USAGE] expert history: <N> selections`, `[PIN] hot store: ...` — same startup shape as before this feature; no `[PROFILE]` line.

- [ ] **Step 2: Byte-identical greedy output with a profile**

```bash
SNAP=C:/Users/Von/Downloads/model TEMP=0 NGEN=24 PROFILE=coding PROMPT="Write a haiku about caches." ./glm.exe 2>prof.err >prof.out
diff base.out prof.out && echo IDENTICAL
grep '\[PROFILE\]' prof.err
```

Expected: `IDENTICAL` (pinning changes only where weights are read from, never the result); `[PROFILE] coding: new profile (pin set seeded from <N> global selections)`. Being a new profile, there is no `pin set differs` line yet (blend == global when pbase is empty).

- [ ] **Step 3: Both histories grow, no double-counting**

```bash
ls -la C:/Users/Von/Downloads/model/.coli_usage*
awk '{t+=$3} END {print t}' C:/Users/Von/Downloads/model/.coli_usage.coding
awk '{t+=$3} END {print t}' C:/Users/Von/Downloads/model/.coli_usage
```

Expected: `.coli_usage.coding` exists and its total equals (roughly) the selections of the one profiled run (≈ 616 × tokens processed — small, thousands not tens of thousands); the global total grew by about the two runs' selections, NOT doubled. Run Step 2's command once more and re-check: the coding total should increase by about one run's worth again (linear growth = no baseline re-adding).

- [ ] **Step 4: Second profiled start announces and diffs the pin set**

```bash
SNAP=C:/Users/Von/Downloads/model TEMP=0 NGEN=8 PROFILE=coding PROMPT="hi" ./glm.exe 2>prof2.err >/dev/null
grep '\[PROFILE\]' prof2.err
```

Expected: `[PROFILE] coding: <n> selections (+ <N> global, weight 4x)` and, if AUTOPIN pinned (global history is well past 5000), `[PROFILE] pin set: <p>% differs from global-only`. With a young profile `<p>` may be small — the line existing is the check, not its value.

- [ ] **Step 5: Invalid name falls back cleanly**

```bash
SNAP=C:/Users/Von/Downloads/model TEMP=0 NGEN=8 PROFILE="../evil" PROMPT="hi" ./glm.exe 2>bad.err >/dev/null
grep '\[PROFILE\]' bad.err
ls C:/Users/Von/Downloads/model/.coli_usage..* 2>/dev/null || echo NO_STRAY_FILES
```

Expected: `[PROFILE] invalid name '../evil' ...; running without a profile`, then `NO_STRAY_FILES`.

- [ ] **Step 6: Clean up scratch outputs and commit nothing**

```bash
rm -f base.err base.out prof.err prof.out prof2.err bad.err
```

This task produces no commit; it gates the feature as done. If any step fails, use superpowers:systematic-debugging before touching code.

---

## Self-review notes (done at plan-writing time)

- **Spec coverage:** data model (Task 2), blend + W (Tasks 1, 4), pin from memory (Task 3), dual save (Task 2), UX env+flag (Task 5), `[PROFILE]` startup + pin-diff lines (Task 4), STATS blended view (Task 2 Step 3), error handling (Task 1 validation, Task 4 fallback, Task 6 Step 5), tests (Tasks 1, 6). Follow-up (prefetch) is spec-recorded, deliberately no task.
- **Placeholders:** none; every code step shows the code.
- **Type consistency:** `PRec{l,e,c}` defined once in profile.h, used by Tasks 1, 3, 4; `pin_load_recs(Model*,PRec*,int,double,const char*) -> int` consistent between Tasks 3 and 4; `stats_dump2(Model*,const char*,uint32_t**,uint32_t**,int)` consistent between Task 2 steps.
