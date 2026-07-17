# NPU Router / Concierge — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Qwen3-1.7B concierge in front of GLM: instant flagged answers to trivial queries, stateless JSON tasks to GLM over KV slots, expert prefetch on intent, and intent→expert-usage logging.

**Architecture:** `zun chat` gains a router layer (new `c/router.py`): a deterministic RouteGate, a Lemonade-hosted concierge (OpenAI-compatible HTTP), and a task-slot manager that maps `task_id` → engine KV slot using the existing `\x02PROMPT` prefix-matching. The engine gains two serve commands: `\x02PREFETCH <profile>` (stage experts from `.coli_usage.<profile>` into free LRU slots) and `\x02USAGE` (per-turn expert-usage delta for the intent log).

**Tech Stack:** Python 3 stdlib only (urllib, unittest — no new pip deps in zun), C99 + the repo's header/test pattern, Lemonade server (external process) hosting Qwen3-1.7B GGUF.

**Spec:** `docs/superpowers/specs/2026-07-17-npu-router-design.md`

**Build/test commands (Windows, PowerShell tool only):**
- C: `$env:TMP="$env:USERPROFILE\AppData\Local\Temp"; $env:MSYSTEM="UCRT64"; C:\msys64\usr\bin\bash.exe -lc "cd /c/Users/Von/Desktop/colibri/c && make tests/test_prefetch.exe && ./tests/test_prefetch.exe"`
- Engine rebuild (HIP, do NOT run plain `make glm` — it clobbers the GPU binary): `... bash.exe -lc "cd /c/Users/Von/Desktop/colibri/c && touch backend_hip.o && make HIP=1 glm"`
- Python: `... bash.exe -lc "cd /c/Users/Von/Desktop/colibri/c && python -m unittest tests.test_router -v"`
- Git: Git Bash tool only.

---

### Task 1: Lemonade + Qwen3-1.7B setup (environment, no code)

**Files:** none (environment only).

- [ ] **Step 1: Install Lemonade server**

Follow https://lemonade-server.ai install for Windows (or `pip install lemonade-sdk` into the global Python 3.12 — the same env that now hosts the ROCm wheel). Do not guess CLI flags; use `lemonade-server --help` after install.

- [ ] **Step 2: Pull and serve Qwen3-1.7B (GGUF int4) on CPU**

Start the server, then verify the model list endpoint:

```powershell
curl.exe http://127.0.0.1:8000/api/v1/models
```

Expected: JSON list including a Qwen3-1.7B variant. Note its exact `id` — it becomes `ZUN_ROUTER_MODEL`.

- [ ] **Step 3: Smoke-test a chat completion**

```powershell
curl.exe -s http://127.0.0.1:8000/api/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"<id from step 2>","messages":[{"role":"user","content":"say OK"}],"max_tokens":8}'
```

Expected: JSON with `choices[0].message.content`. Record the base URL; if the port differs from 8000, all later tasks use `ZUN_ROUTER_URL` to point at it. If Lemonade cannot serve this model class on this box, fall back to any OpenAI-compatible local server (llama.cpp `llama-server -m qwen3-1.7b-*.gguf --port 8000`) — the router only needs the API shape.

---

### Task 2: RouteGate (pure Python, TDD)

**Files:**
- Create: `c/router.py`
- Test: `c/tests/test_router.py`

- [ ] **Step 1: Write the failing tests**

```python
# c/tests/test_router.py
"""Router layer unit tests: RouteGate decisions, envelope rendering,
task-slot LRU, concierge JSON parsing, intent log."""
import json, os, tempfile, unittest, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import router

class TestRouteGate(unittest.TestCase):
    def test_trivial_goes_local(self):
        for m in ["hi", "Hello!", "hey", "thanks", "thank you", "ok", "bye",
                  "good morning", "yep", "lol"]:
            self.assertEqual(router.route_gate(m), "local", m)
    def test_hard_keywords_go_glm(self):
        for m in ["write code for dijkstra", "think about this problem",
                  "reason through the tradeoffs", "review my coding style",
                  "here is a snippet ```py\nx=1\n```", "fix main.py please"]:
            self.assertEqual(router.route_gate(m), "glm", m)
    def test_long_messages_go_glm(self):
        self.assertEqual(router.route_gate("explain " + "x " * 200), "glm")
    def test_middle_asks_concierge(self):
        for m in ["explain this error", "what's the difference between TCP and UDP",
                  "summarize our discussion"]:
            self.assertEqual(router.route_gate(m), "ask", m)
    def test_local_only_when_short(self):
        # a greeting buried in a long message is not trivial
        self.assertNotEqual(router.route_gate("hi " + "and also " * 20), "local")

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `python -m unittest tests.test_router -v` (from `c/`)
Expected: ERROR — `No module named 'router'`. Create an empty `c/router.py`, re-run, expect FAIL with `AttributeError: route_gate` — that is the correct red.

- [ ] **Step 3: Implement RouteGate**

```python
# c/router.py
"""Router/concierge layer (NPU router spec, phase 1).
Pure logic lives here so tests never need Lemonade or the engine."""
import json, re, time, urllib.request, urllib.error

_LOCAL = re.compile(
    r"^(hi|hiya|hello|hey|yo|howdy|sup|good\s?(morning|afternoon|evening|night)"
    r"|bye|goodbye|see\s?you|later|thanks?|thank\s?you|thx|ty|ok|okay|k|cool"
    r"|nice|great|lol|haha|yes|yep|no|nope)[\s!.,?]*$", re.I)
_GLM = re.compile(r"\b(code|coding|think|thinking|reason|reasoning)\b|```", re.I)
_PATH = re.compile(
    r"\b[\w./\\-]+\.(py|c|h|cpp|hpp|js|ts|rs|go|java|sh|ps1|md|json|yaml|yml|toml)\b", re.I)

def route_gate(msg):
    """Deterministic pre-filter, runs before any model.
    'local' = trivial, concierge answers; 'glm' = always the big model;
    'ask' = concierge judges (biased to escalate — spec: default to GLM)."""
    s = msg.strip()
    if len(s) <= 40 and _LOCAL.match(s): return "local"
    if _GLM.search(s) or _PATH.search(s) or len(s) > 240: return "glm"
    return "ask"
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `python -m unittest tests.test_router -v` — Expected: all `TestRouteGate` PASS.

- [ ] **Step 5: Commit**

```bash
git add c/router.py c/tests/test_router.py
git commit -m "feat(router): RouteGate deterministic pre-filter (TDD)"
```

---

### Task 3: Task envelope + prompt rendering (TDD)

**Files:**
- Modify: `c/router.py`
- Modify: `c/tests/test_router.py`

- [ ] **Step 1: Add failing tests**

```python
class TestEnvelope(unittest.TestCase):
    def test_valid_intent(self):
        self.assertTrue(router.valid_intent("coding"))
        self.assertFalse(router.valid_intent("../evil"))
        self.assertFalse(router.valid_intent(""))
        self.assertFalse(router.valid_intent("x" * 33))
    def test_render_new_task_nothink(self):
        env = {"task_id": "t1", "action": "new", "intent": "coding",
               "think": False, "instructions": "write dijkstra", "context": ""}
        p = router.render_task_prompt([(env, None)])
        self.assertTrue(p.startswith("[gMASK]<sop><|user|>"))
        self.assertIn('"instructions": "write dijkstra"', p)
        self.assertTrue(p.endswith("<|assistant|><think></think>"))
    def test_render_think_omits_empty_think(self):
        env = {"task_id": "t1", "action": "new", "intent": "math",
               "think": True, "instructions": "prove it", "context": ""}
        p = router.render_task_prompt([(env, None)])
        self.assertTrue(p.endswith("<|assistant|>"))
    def test_render_continuation_appends(self):
        e1 = {"task_id": "t1", "action": "new", "intent": "coding",
              "think": False, "instructions": "write dijkstra", "context": ""}
        e2 = dict(e1, action="continue", instructions="add type hints")
        p = router.render_task_prompt([(e1, "def dijkstra(): ..."), (e2, None)])
        # first turn (envelope + reply) is a byte-prefix of the two-turn prompt
        p1 = router.render_task_prompt([(e1, "def dijkstra(): ...")])
        self.assertTrue(p.startswith(p1))
        self.assertIn('"add type hints"', p)
```

- [ ] **Step 2: Run, verify FAIL** (`AttributeError: valid_intent`)

- [ ] **Step 3: Implement**

```python
_INTENT = re.compile(r"^[A-Za-z0-9_-]{1,32}$")   # same rule as profile.h

def valid_intent(name):
    return bool(name) and bool(_INTENT.match(name))

def render_task_prompt(turns):
    """turns: [(envelope_dict, raw_reply_or_None)]. Renders the FULL task
    transcript every time — the engine's \\x02PROMPT prefix matching turns the
    unchanged prefix into a KV hit, which is how 'continue' stays cheap.
    Keys are sorted so the same envelope always renders byte-identically."""
    out = ["[gMASK]<sop>"]
    for env, reply in turns:
        think = "" if env.get("think") else "<think></think>"
        out.append("<|user|>" + json.dumps(env, sort_keys=True, ensure_ascii=False)
                   + "<|assistant|>" + think)
        if reply is not None: out.append(reply)
    return "".join(out)
```

- [ ] **Step 4: Run tests, verify PASS.** Note: the prefix-property test is the load-bearing one — if rendering ever breaks byte-stability, task continuation silently degrades to full re-prefill.

- [ ] **Step 5: Commit** — `git commit -m "feat(router): task envelope rendering with prefix-stable transcripts"`

---

### Task 4: TaskSlots LRU (TDD)

**Files:**
- Modify: `c/router.py`
- Modify: `c/tests/test_router.py`

- [ ] **Step 1: Add failing tests**

```python
class TestTaskSlots(unittest.TestCase):
    def test_assign_and_reuse(self):
        ts = router.TaskSlots(n=2)
        s1, ev = ts.slot_for("a"); self.assertEqual((s1, ev), (0, None))
        s2, ev = ts.slot_for("b"); self.assertEqual((s2, ev), (1, None))
        s1b, ev = ts.slot_for("a"); self.assertEqual((s1b, ev), (0, None))
    def test_lru_eviction(self):
        ts = router.TaskSlots(n=2)
        ts.slot_for("a"); ts.slot_for("b"); ts.slot_for("a")   # b is LRU now
        s, ev = ts.slot_for("c")
        self.assertEqual(ev, "b"); self.assertEqual(s, 1)
        self.assertIsNone(ts.turns.get("b"))                   # transcript dropped
    def test_turns_survive_reuse(self):
        ts = router.TaskSlots(n=2)
        ts.slot_for("a"); ts.turns["a"].append(({"task_id": "a"}, "reply"))
        ts.slot_for("a")
        self.assertEqual(len(ts.turns["a"]), 1)
```

- [ ] **Step 2: Run, verify FAIL** (`AttributeError: TaskSlots`)

- [ ] **Step 3: Implement**

```python
class TaskSlots:
    """task_id -> engine KV slot. Eviction = LRU; the evicted task's next use
    starts a fresh slot transcript (the engine truncates on divergent prefix,
    so no explicit RESET is needed)."""
    def __init__(self, n=8):
        self.n = n; self.map = {}; self.turns = {}; self.clock = 0; self.stamp = {}
    def slot_for(self, tid):
        self.clock += 1
        if tid in self.map:
            self.stamp[tid] = self.clock; return self.map[tid], None
        if len(self.map) < self.n:
            slot = len(self.map); evicted = None
        else:
            evicted = min(self.stamp, key=self.stamp.get)
            slot = self.map.pop(evicted)
            self.stamp.pop(evicted); self.turns.pop(evicted, None)
        self.map[tid] = slot; self.stamp[tid] = self.clock
        self.turns.setdefault(tid, [])
        return slot, evicted
```

- [ ] **Step 4: Run tests, verify PASS.**

- [ ] **Step 5: Commit** — `git commit -m "feat(router): TaskSlots LRU mapping task_id -> KV slot"`

---

### Task 5: Concierge JSON parsing + system prompt (TDD)

**Files:**
- Modify: `c/router.py`
- Modify: `c/tests/test_router.py`

- [ ] **Step 1: Add failing tests**

```python
class TestParseConcierge(unittest.TestCase):
    def test_local_reply(self):
        d = router.parse_concierge('{"route":"local","reply":"Hello!"}')
        self.assertEqual(d, {"route": "local", "reply": "Hello!"})
    def test_glm_task(self):
        raw = ('{"route":"glm","action":"new","intent":"coding","think":false,'
               '"instructions":"write dijkstra","context":""}')
        d = router.parse_concierge(raw)
        self.assertEqual(d["route"], "glm"); self.assertEqual(d["intent"], "coding")
    def test_json_inside_prose_is_extracted(self):
        d = router.parse_concierge('Sure!\n{"route":"local","reply":"hi"}\nDone.')
        self.assertEqual(d["route"], "local")
    def test_garbage_returns_none(self):
        self.assertIsNone(router.parse_concierge("I think you should..."))
        self.assertIsNone(router.parse_concierge('{"route":"glm"}'))          # missing fields
        self.assertIsNone(router.parse_concierge('{"route":"glm","action":"new",'
            '"intent":"../evil","think":false,"instructions":"x","context":""}'))
```

- [ ] **Step 2: Run, verify FAIL.**

- [ ] **Step 3: Implement parser + the system prompt constant**

```python
SYSTEM_PROMPT = """You are zun-mini, the front desk for a 744B model called GLM. Reply with ONE JSON object only — no prose, no markdown fences.
If the user's message is small talk (greeting, thanks, goodbye, a quick meta question about this session), reply:
  {"route":"local","reply":"<your short answer>"}
Otherwise hand it to GLM:
  {"route":"glm","action":"new"|"continue","intent":"<one word: coding|writing|math|analysis|general>","think":true|false,"instructions":"<what GLM must do, self-contained>","context":"<verbatim code/data from the conversation that GLM needs, else empty string>"}
Rules:
- When in doubt, route to GLM. You answer ONLY what a receptionist could.
- "continue" ONLY if this message refines the task GLM just worked on; a new goal is "new".
- think=true when the user asks to think, reason, prove, or debug something subtle.
- GLM sees nothing except your JSON. Put everything it needs in instructions+context."""

_JSON_RE = re.compile(r"\{.*\}", re.S)

def parse_concierge(text):
    """Extract and validate the concierge's JSON. None = unusable (caller
    retries once, then falls back to raw passthrough — never a dead end)."""
    m = _JSON_RE.search(text or "")
    if not m: return None
    try: d = json.loads(m.group(0))
    except json.JSONDecodeError: return None
    if d.get("route") == "local":
        return d if isinstance(d.get("reply"), str) else None
    if d.get("route") == "glm":
        ok = (d.get("action") in ("new", "continue")
              and valid_intent(d.get("intent", ""))
              and isinstance(d.get("think"), bool)
              and isinstance(d.get("instructions"), str) and d["instructions"]
              and isinstance(d.get("context"), str))
        return d if ok else None
    return None
```

- [ ] **Step 4: Run tests, verify PASS.**

- [ ] **Step 5: Commit** — `git commit -m "feat(router): concierge system prompt + strict JSON parsing"`

---

### Task 6: Concierge HTTP client + intent log (TDD where pure)

**Files:**
- Modify: `c/router.py`
- Modify: `c/tests/test_router.py`

- [ ] **Step 1: Add failing tests** (log is pure; the client is tested for its fallback behavior only — no live server in unit tests)

```python
class TestIntentLog(unittest.TestCase):
    def test_appends_jsonl(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, ".zun_intent_log")
            router.log_intent(path, {"task_id": "t1", "intent": "coding", "tok": 5})
            router.log_intent(path, {"task_id": "t1", "intent": "coding", "tok": 9})
            lines = open(path, encoding="utf-8").read().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[1])["tok"], 9)

class TestConciergeClient(unittest.TestCase):
    def test_unreachable_returns_none(self):
        c = router.Concierge("http://127.0.0.1:1", "any-model", timeout=0.3)
        self.assertIsNone(c.chat([{"role": "user", "content": "hi"}]))
```

- [ ] **Step 2: Run, verify FAIL.**

- [ ] **Step 3: Implement**

```python
def log_intent(path, record):
    """Append one JSONL record. The corpus for the phase-3 recommender —
    best-effort, never raises into the chat loop."""
    try:
        record = dict(record, ts=round(time.time(), 3))
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass

class Concierge:
    """Minimal OpenAI-compatible chat client (Lemonade). None on any failure:
    the caller treats None as 'router unavailable' and passes through to GLM."""
    def __init__(self, base_url, model, timeout=5.0):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model; self.timeout = timeout
    def chat(self, messages, max_tokens=512):
        body = json.dumps({"model": self.model, "messages": messages,
                           "max_tokens": max_tokens, "temperature": 0.2}).encode()
        req = urllib.request.Request(self.url, data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                d = json.loads(r.read().decode("utf-8", "replace"))
            return d["choices"][0]["message"]["content"]
        except Exception:
            return None
```

- [ ] **Step 4: Run the whole file:** `python -m unittest tests.test_router -v` — all PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat(router): Lemonade client (fail-open) + intent JSONL log"`

---

### Task 7: `prefetch.h` staging selection (C, TDD)

**Files:**
- Create: `c/prefetch.h`
- Create: `c/tests/test_prefetch.c`
- Modify: `c/Makefile` (TEST_BINS + rule)

- [ ] **Step 1: Write the failing test**

```c
/* Unit test for prefetch selection (prefetch.h): pick highest-scored
 * non-resident experts into FREE LRU slots only — never evict, bounded total.
 * Pure logic, no I/O. PRec comes from profile.h. */
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "profile.h"
#include "prefetch.h"

int main(void){
    /* layer 0 has 2 free slots, layer 1 has 0, layer 2 has 1 */
    int freec[3] = {2, 0, 1};
    PRec r[] = {{1, 9, 100},   /* best score but layer 1 is full -> skipped */
                {0, 3, 90}, {2, 7, 80}, {0, 5, 70}, {0, 6, 60},  /* L0 full after 2 */
                {0, 3, 50}};   /* duplicate of (0,3) -> skipped */
    PRec out[8];
    int n = pf_select(r, 6, freec, 3, 8, out);
    assert(n == 3);
    assert(out[0].l == 0 && out[0].e == 3);
    assert(out[1].l == 2 && out[1].e == 7);
    assert(out[2].l == 0 && out[2].e == 5);
    /* max_total caps the batch (bounded staging latency) */
    int freec2[3] = {9, 9, 9};
    n = pf_select(r, 6, freec2, 3, 2, out);
    assert(n == 2);
    /* rows out of range are ignored, empty input is fine */
    PRec bad[] = {{-1, 0, 5}, {3, 0, 5}};
    assert(pf_select(bad, 2, freec2, 3, 8, out) == 0);
    assert(pf_select(NULL, 0, freec2, 3, 8, out) == 0);
    printf("test_prefetch: PASS (score order, free-slot cap, dedupe, bounds)\n");
    return 0;
}
```

Makefile additions (mirror the `test_kvdisk` pattern):

```make
TEST_BINS = ... tests/test_prefetch$(EXE)      # append to the existing list

tests/test_prefetch$(EXE): tests/test_prefetch.c prefetch.h profile.h
	$(CC) $(CFLAGS) -I. $< -o $@ $(LDFLAGS)
```

- [ ] **Step 2: Create `c/prefetch.h` with a stub returning -1; build and run**

Run: `make tests/test_prefetch.exe && ./tests/test_prefetch.exe`
Expected: `Assertion failed: n == 3` — correct red.

- [ ] **Step 3: Implement**

```c
/* Prefetch selection — pure, testable (test_prefetch). Include AFTER profile.h.
 *
 * \x02PREFETCH <profile> stages the profile's hottest NON-resident experts into
 * FREE LRU slots while the user reads the router's acknowledgment. Free slots
 * only: prefetch is a hint and must never evict what the session already
 * earned (pins are untouchable by design; LRU residents by choice). */
#ifndef COLIBRI_PREFETCH_H
#define COLIBRI_PREFETCH_H

static int pf_cmp_desc(const void *a, const void *b){
    uint32_t ca=((const PRec*)a)->c, cb=((const PRec*)b)->c;
    return ca<cb ? 1 : ca>cb ? -1 : 0;
}

/* r[n]: candidates (caller already excluded resident experts); freec[nrows]:
 * free LRU slots per layer row. Picks by score desc, at most max_total.
 * Returns the number selected into out (caller sizes out >= max_total). */
static int pf_select(PRec *r, int n, const int *freec, int nrows,
                     int max_total, PRec *out){
    if(!r || n<=0 || max_total<=0) return 0;
    qsort(r, (size_t)n, sizeof(PRec), pf_cmp_desc);
    int *left = malloc((size_t)nrows*sizeof(int));
    memcpy(left, freec, (size_t)nrows*sizeof(int));
    int taken=0;
    for(int i=0;i<n && taken<max_total;i++){
        int l=r[i].l, e=r[i].e;
        if(l<0 || l>=nrows || left[l]<=0) continue;
        int dup=0;
        for(int j=0;j<taken;j++) if(out[j].l==l && out[j].e==e){ dup=1; break; }
        if(dup) continue;
        out[taken++]=r[i]; left[l]--;
    }
    free(left);
    return taken;
}

#endif
```

- [ ] **Step 4: Build + run: expect PASS; then `make test-c` — all suites PASS.**

- [ ] **Step 5: Commit** — `git commit -m "feat(prefetch): pf_select staging logic, free-slots-only (TDD)"`

---

### Task 8: Engine `\x02PREFETCH` + `\x02USAGE` serve commands

**Files:**
- Modify: `c/glm.c` (include block ~line 155, Model struct ~line 150, run_serve command dispatch — the `\x02RESET`/`\x02MORE` block ~line 2270)

No pure logic here (all covered by Task 7); verified by build + the piped serve session in Step 4.

- [ ] **Step 1: Include header and extend Model**

After `#include "kvdisk.h"` add:

```c
#include "prefetch.h"                                /* \x02PREFETCH: selezione staging pura */
```

In the Model struct, next to `uint32_t **eusage;` add:

```c
    uint32_t **euse_snap;                        /* snapshot per \x02USAGE: delta per-turno al router */
```

- [ ] **Step 2: Add `prefetch_stage` above `run_serve`**

```c
/* \x02PREFETCH <profile>: carica gli expert piu' caldi del profilo negli slot
 * LRU LIBERI (mai evict) mentre l'utente legge l'ack del router. Best-effort:
 * profilo assente/invalido = 0, nessun errore. PREFETCH_MAX limita il batch
 * (default 64 expert ~ 1.2 GB ~ <1 s: il costo resta sotto il prefill). */
static int prefetch_stage(Model *m, const char *snap, const char *prof){
    if(!prof_valid_name(prof)) return 0;
    char path[2048]; snprintf(path,sizeof(path),"%s/.coli_usage.%s",snap,prof);
    FILE *f=fopen(path,"r"); if(!f) return 0;
    Cfg *c=&m->c; int NR=c->n_layers+1, cap=NR*c->n_experts;
    PRec *r=malloc((size_t)cap*sizeof(PRec)); int n=0; int l,e; uint32_t cnt;
    while(n<cap && fscanf(f,"%d %d %u",&l,&e,&cnt)==3){
        int ok = l>=0 && e>=0 && e<c->n_experts &&
                 ((l<c->n_layers && m->L[l].sparse) || (l==c->n_layers && m->has_mtp));
        if(ok && !ec_resident(m,l,e)) r[n++]=(PRec){l,e,cnt};
    }
    fclose(f);
    int *freec=malloc((size_t)NR*sizeof(int));
    for(int i=0;i<NR;i++) freec[i]= m->ecache[i] ? m->ecap-m->ecn[i] : 0;
    int maxn = getenv("PREFETCH_MAX")?atoi(getenv("PREFETCH_MAX")):64;
    if(maxn<1) maxn=1;
    PRec *sel=malloc((size_t)maxn*sizeof(PRec));
    int ns=pf_select(r,n,freec,NR,maxn,sel);
    double t0=now_s();
    #pragma omp parallel for schedule(dynamic,1)
    for(int a=0;a<ns;a++){
        int li=sel[a].l, slot;
        #pragma omp critical
        { slot=m->ecn[li]++; m->eclock++; }
        expert_load(m,li,sel[a].e,&m->ecache[li][slot]);
        m->ecache[li][slot].used=(uint32_t)m->eclock;
    }
    if(ns) fprintf(stderr,"[PREFETCH] %s: %d experts staged in %.1fs\n",prof,ns,now_s()-t0);
    free(r); free(freec); free(sel);
    return ns;
}
```

- [ ] **Step 3: Wire both commands into the serve dispatch**

In `run_serve`, immediately after the `\x02MORE` handler's `continue; }`, add:

```c
        if(!strncmp(line,"\x02PREFETCH ",10)){         /* staging-hint dal router */
            int ns=prefetch_stage(m,snap,line+10);
            printf("PREFETCH %d\n",ns);
            printf("\x01\x01" "END" "\x01\x01\n"); printf("STAT 0 0.00 0.0 %.2f\n", rss_gb());
            fflush(stdout); continue; }
        if(!strcmp(line,"\x02USAGE")){                 /* delta uso expert dall'ultima chiamata:
            righe "layer eid n" per il log intent->expert del router */
            Cfg *cu=&m->c; int NRu=cu->n_layers+1;
            if(!m->euse_snap) m->euse_snap=calloc(NRu,sizeof(uint32_t*));
            for(int i=0;i<NRu;i++) if(m->eusage[i]){
                if(!m->euse_snap[i]) m->euse_snap[i]=calloc(cu->n_experts,sizeof(uint32_t));
                for(int j=0;j<cu->n_experts;j++){
                    uint32_t d=m->eusage[i][j]-m->euse_snap[i][j];
                    if(d) printf("%d %d %u\n",i,j,d);
                    m->euse_snap[i][j]=m->eusage[i][j];
                }
            }
            printf("\x01\x01" "END" "\x01\x01\n"); printf("STAT 0 0.00 0.0 %.2f\n", rss_gb());
            fflush(stdout); continue; }
```

- [ ] **Step 4: Rebuild and verify with a piped serve session**

Rebuild: `touch backend_hip.o && make HIP=1 glm` (HIP build — never plain `make glm`). Then, in PowerShell (background, this loads the model):

```powershell
Set-Location C:\Users\Von\Desktop\colibri\c
$env:SNAP="C:/Users/Von/Downloads/model"; $env:SERVE="1"; $env:AUTOPIN="0"
"`u{2}PREFETCH coding`n`u{2}USAGE" | .\glm.exe 2>prefetch_check.err | Select-String "PREFETCH"
```

Expected stdout: `PREFETCH <n>` with n>0 (the `coding` profile exists on this box) and stderr containing `[PREFETCH] coding: <n> experts staged`. `\x02USAGE` right after boot prints no delta lines (nothing generated) — just the sentinels. Also run `make test-c`: all PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat(serve): \\x02PREFETCH profile staging + \\x02USAGE per-turn expert delta"`

---

### Task 9: Router wiring in `zun chat`

**Files:**
- Modify: `c/zun` (`cmd_chat`, ~lines 372–473; `env_for` untouched)
- Modify: `c/router.py` (one orchestration function)

This is the integration layer — covered by Task 10's live checklist rather than unit tests; everything decision-shaped already lives in tested `router.py` functions.

- [ ] **Step 1: Add the orchestration entry point to `c/router.py`**

```python
def decide(msg, concierge, force_continue=False):
    """One user message -> an action dict for zun:
      {"kind":"local","reply":str,"src":"gate"|"mini"}
      {"kind":"glm","task":dict}        # validated concierge task (no task_id yet)
      {"kind":"passthrough"}            # router unavailable/unparseable: raw to GLM
    """
    if msg.startswith("!glm "):
        return {"kind": "glm", "task": {"action": "continue" if force_continue else "new",
                "intent": "general", "think": False,
                "instructions": msg[5:].strip(), "context": ""}}
    route = route_gate(msg)
    if route == "local":
        reply = None
        if concierge:
            raw = concierge.chat([{"role": "system", "content": SYSTEM_PROMPT},
                                  {"role": "user", "content": msg}])
            d = parse_concierge(raw) if raw else None
            if d and d.get("route") == "local": reply = d["reply"]
        return {"kind": "local", "reply": reply or "Hi! (zun-mini)", "src": "mini"}
    if concierge is None:
        return {"kind": "passthrough"}
    raw = concierge.chat([{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": msg}])
    d = parse_concierge(raw) if raw else None
    if d is None and raw is not None:      # one retry with the error shown
        raw = concierge.chat([{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": msg},
                              {"role": "assistant", "content": raw},
                              {"role": "user", "content": "Invalid JSON. Reply with ONE valid JSON object only."}])
        d = parse_concierge(raw) if raw else None
    if d is None: return {"kind": "passthrough"}
    if d["route"] == "local" and route == "ask":
        return {"kind": "local", "reply": d["reply"], "src": "mini"}
    if d["route"] == "glm":
        if force_continue: d["action"] = "continue"
        return {"kind": "glm", "task": {k: d[k] for k in
                ("action", "intent", "think", "instructions", "context")}}
    return {"kind": "passthrough"}
```

Add a matching unit test to `c/tests/test_router.py` (concierge=None paths are pure):

```python
class TestDecide(unittest.TestCase):
    def test_bang_glm_forces_task(self):
        d = router.decide("!glm do the thing", None)
        self.assertEqual(d["kind"], "glm")
        self.assertEqual(d["task"]["instructions"], "do the thing")
    def test_no_concierge_middle_passthrough(self):
        self.assertEqual(router.decide("explain this error", None)["kind"], "passthrough")
    def test_no_concierge_trivial_still_local(self):
        d = router.decide("hi", None)
        self.assertEqual(d["kind"], "local")
```

Run `python -m unittest tests.test_router -v` (red first — `decide` missing — then green after implementing).

- [ ] **Step 2: Wire into `cmd_chat`**

At the top of `cmd_chat` (after `need_model`), initialize the router (kill switch honored):

```python
    import router as R
    router_on = os.environ.get("ZUN_ROUTER", "1") != "0"
    concierge = None
    if router_on:
        url = os.environ.get("ZUN_ROUTER_URL", "http://127.0.0.1:8000/api/v1")
        rmodel = os.environ.get("ZUN_ROUTER_MODEL", "Qwen3-1.7B-GGUF")
        c0 = R.Concierge(url, rmodel, timeout=float(os.environ.get("ZUN_ROUTER_TIMEOUT", "5")))
        if c0.chat([{"role": "user", "content": "ping"}], max_tokens=4) is not None:
            concierge = c0
        else:
            print(f"  {C.yel}router: Lemonade unreachable at {url} — direct GLM mode{C.r}")
    tasks = R.TaskSlots(n=8); cur_task = [0]; force_cont = [False]
    ilog = os.path.join(a.model, ".zun_intent_log")
```

In the env setup line `e=env_for(a); e["SERVE"]="1"`, add `e["KV_SLOTS"]="8"` when `concierge` is not None.

Inside the message loop, replace the plain `else: p.stdin.write((msg...))` branch with routing (keep `:reset`/`:more`/quit handling as-is; add `:new` and `:continue` beside them):

```python
            if msg == ":new":
                cur_task[0] += 1; force_cont[0] = False
                print(f"  {C.dim}✦ next query starts a fresh GLM task{C.r}\n"); continue
            if msg == ":continue":
                force_cont[0] = True
                print(f"  {C.dim}✦ next query continues the current GLM task{C.r}\n"); continue
            act = R.decide(msg, concierge, force_cont[0]) if concierge or R.route_gate(msg) == "local" \
                  else {"kind": "passthrough"}
            force_cont[0] = False
            if act["kind"] == "local":
                print(f"\n  {C.org}◆ [mini]{C.r} {act['reply']}\n"); continue
            if act["kind"] == "glm":
                t = act["task"]
                if t["action"] == "new": cur_task[0] += 1
                tid = f"t{cur_task[0]}"
                slot, evicted = tasks.slot_for(tid)
                if evicted: print(f"  {C.dim}✦ task {evicted} evicted from slot {slot}{C.r}")
                # fire prefetch while composing (best-effort; response is "PREFETCH n")
                p.stdin.write(f"\x02PREFETCH {t['intent']}\n".encode()); p.stdin.flush()
                stream_turn(p, END, lambda b: None)
                env = dict(t, task_id=tid)
                tasks.turns[tid].append((env, None))
                prompt = R.render_task_prompt(tasks.turns[tid])
                pb = prompt.encode()
                hdr = f"\x02PROMPT {len(pb)} {a.ngen} {a.temp if a.temp is not None else 1.0} 0.95 {slot}\n"
                p.stdin.write(hdr.encode() + pb + b"\n"); p.stdin.flush()
            else:
                p.stdin.write((msg.replace("\n", " ") + "\n").encode()); p.stdin.flush()
```

After the existing `st=stream_turn(p, END, echo)` completes for a `glm` task, record the raw reply and log intent (the raw text is accumulated by extending `echo`: append each decoded chunk to a `parts` list alongside the existing rendering). Then:

```python
            if act.get("kind") == "glm":
                tasks.turns[tid][-1] = (env, "".join(parts))
                R.log_intent(ilog, {"task_id": tid, "intent": env["intent"],
                                    "action": env["action"], "tok": st.get("tok", 0),
                                    "tps": st.get("tps", 0), "hit": st.get("hit", 0)})
```

- [ ] **Step 3: Sanity-run the unit suites** — `python -m unittest discover -s tests -p 'test_*.py'` and `make test-c`: all PASS (`cmd_chat` changes are exercised in Task 10).

- [ ] **Step 4: Commit** — `git commit -m "feat(zun): router layer in chat — [mini] flagging, task slots, prefetch, intent log"`

---

### Task 10: Live end-to-end checklist (real model + Lemonade)

**Files:** none (verification; fixes loop back into Tasks 2–9).

- [ ] **Step 1: Router-off regression** — `ZUN_ROUTER=0 ./run_glm52.sh`: chat behaves exactly as before (this is the "never worse than today" gate).
- [ ] **Step 2: Lemonade down** — stop Lemonade, run with router on: warning printed once, queries go straight to GLM.
- [ ] **Step 3: Trivial query** — with Lemonade up: `hi` → `[mini]`-flagged answer well under 1 s, engine stderr shows NO prefill.
- [ ] **Step 4: Hard query** — `write code for dijkstra` → stderr shows `[PREFETCH] coding: n experts staged`, GLM streams an unflagged answer, `.zun_intent_log` gains a record.
- [ ] **Step 5: Continuation** — `now add type hints` → same task id in the log (`action":"continue`), engine stderr `[API] KV slot ... prefix` shows a non-zero prefix (KV reuse worked).
- [ ] **Step 6: Overrides** — `:new` then a follow-up starts `t<n+1>`; `!glm hi` forces a GLM task for a trivial message.
- [ ] **Step 7: Commit any fixes** discovered, one commit per fix.

---

### Task 11: Prefetch benefit probe

**Files:**
- Create: `bench/router/probe_prefetch.sh`

- [ ] **Step 1: Write the probe** (engine-only, no Lemonade needed — measures the mechanism)

```bash
#!/usr/bin/env bash
# A/B: same coding task cold, with vs without \x02PREFETCH coding first.
# Usage: ./probe_prefetch.sh   (run each arm after a fresh engine start)
set -e
cd "$(dirname "$0")/../../c"
TASK='{"action":"new","context":"","instructions":"write a python function that finds the shortest path in a weighted graph using dijkstra with heapq. output only code.","intent":"coding","task_id":"t1","think":false}'
PROMPT="[gMASK]<sop><|user|>${TASK}<|assistant|><think></think>"
run_arm(){  # $1 = tag, $2 = optional PREFETCH line
  { [ -n "$2" ] && printf '\x02PREFETCH coding\n'
    printf '\x02PROMPT %d 96 0 0 0\n%s\n' "${#PROMPT}" "$PROMPT"
  } | env SNAP="${SNAP:-C:/Users/Von/Downloads/model}" SERVE=1 SEED=1 TEMP=0 \
      ./glm.exe > "../bench/router/${1}.out" 2> "../bench/router/${1}.err"
  grep -E "^STAT" "../bench/router/${1}.out" | tail -1
  grep -E "PREFETCH|hit" "../bench/router/${1}.err" || true
}
case "${1:-both}" in
  cold)     run_arm cold ;;
  prefetch) run_arm prefetch yes ;;
  both)     run_arm cold; run_arm prefetch yes ;;
esac
```

- [ ] **Step 2: Run both arms** (each is a full engine start; minutes each): `mkdir -p bench/router && ./bench/router/probe_prefetch.sh both`

- [ ] **Step 3: Record results** in `bench/router/RESULTS.md`: STAT tps + hit% per arm, `[PREFETCH]` staging count/time. Success criterion from the spec: measurable hit-rate improvement on the prefetch arm.

- [ ] **Step 4: Commit** — `git commit -m "bench(router): prefetch A/B probe + first results"`

---

### Task 12: Docs + memory close-out

**Files:**
- Modify: `docs/superpowers/specs/2026-07-17-npu-router-design.md` (status header → implemented, phase 1)
- Modify: `c/zun` docstring (add the router env vars to the configuration list: `ZUN_ROUTER=0`, `ZUN_ROUTER_URL`, `ZUN_ROUTER_MODEL`)

- [ ] **Step 1: Update both files** (spec status line; docstring gains three lines).
- [ ] **Step 2: Full suite:** `make test-c` + `python -m unittest discover -s tests -p 'test_*.py'` — all PASS.
- [ ] **Step 3: Commit** — `git commit -m "docs(router): phase-1 close-out"`

---

## Self-review notes (done at plan time)

- **Spec coverage:** RouteGate (T2), concierge+flagging (T5/T9), envelope+statelessness (T3), slots/eviction (T4), `\x02PREFETCH` (T7/T8), `\x02USAGE`+intent log (T6/T8/T9), overrides+kill switch (T9), error table (T6 fail-open client, T9 passthrough, T10 gates), integration probes (T10/T11). Phase 2/3 explicitly out of scope.
- **Known risk, accepted:** tokenizer round-trip on continuation (stored reply text may re-tokenize slightly differently) degrades gracefully to a shorter prefix match — correctness unaffected, cost is extra prefill. Checked live in T10 step 5.
- **Type consistency check:** `PRec{l,e,c}` from `profile.h` used in T7/T8; `pf_select(r,n,freec,nrows,max_total,out)` signature identical in test and impl; `TaskSlots.slot_for` returns `(slot, evicted)` in T4 and T9.
