#!/usr/bin/env python3
"""colibrì router — OpenAI-compatible gateway over multiple local engines.

Classifies each request, arbitrates the single-owner iGPU, lazy-spawns the
backend (llama-server or `coli serve`), proxies streaming, reaps idle engines.
Stdlib only. Config: registry.yaml (+ ../engines/llamacpp/presets.yaml).

  python router.py                 # serve on :8080
  COLI_ROUTER_PORT=9000 python router.py
"""
import os, sys, json, time, threading, subprocess, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ---------- minimal YAML (subset: nested maps, list-of-maps, inline {..} flow) ----------
def _scalar(v):
    v = v.strip()
    if v.startswith(("'", '"')) and v[-1:] == v[:1]: return v[1:-1]
    if v.startswith("{"):                                   # inline flow map
        out = {}
        body = v[1:v.rindex("}")].strip()
        for pair in filter(None, (p.strip() for p in body.split(","))):
            k, _, val = pair.partition(":"); out[k.strip()] = _scalar(val)
        return out
    if v in ("true", "false"): return v == "true"
    for cast in (int, float):
        try: return cast(v)
        except ValueError: pass
    return v

def yaml_load(path):
    """Handles the 2-space-indented subset used by our config files."""
    root, stack = {}, [(-1, {})]                            # (indent, container)
    stack[0] = (-1, root)
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].rstrip()
            if not line.strip(): continue
            indent = len(line) - len(line.lstrip())
            while stack and indent <= stack[-1][0]: stack.pop()
            parent = stack[-1][1]
            body = line.strip()
            if body.startswith("- "):                       # list item (map)
                item = {}
                if not isinstance(parent, list):            # first item: replace key's value
                    continue
                parent.append(item)
                k, _, v = body[2:].partition(":")
                item[k.strip()] = _scalar(v) if v.strip() else {}
                stack.append((indent, item))
                continue
            key, _, val = body.partition(":")
            key = key.strip()
            if val.strip():
                parent[key] = _scalar(val)
            else:                                           # opens a block: map or list
                nxt = _peek_is_list(path, indent)
                parent[key] = [] if nxt else {}
                stack.append((indent, parent[key]))
    return root

def _peek_is_list(path, indent):
    return _PEEK.get((path, indent), False)

def _prescan(path):
    """Record whether each 'key:' block is a list, by peeking the next non-blank line."""
    lines = [l.split("#",1)[0].rstrip() for l in open(path, encoding="utf-8")]
    lines = [l for l in lines if l.strip()]
    for i, l in enumerate(lines):
        ind = len(l) - len(l.lstrip())
        if l.strip().endswith(":"):
            for nxt in lines[i+1:]:
                nind = len(nxt) - len(nxt.lstrip())
                if nind <= ind: break
                _PEEK[(path, ind)] = nxt.strip().startswith("- "); break
_PEEK = {}

def load_yaml(path):
    _prescan(path); return yaml_load(path)

# ---------- config ----------
CFG = load_yaml(os.path.join(HERE, "registry.yaml"))
PRESETS = load_yaml(os.path.join(ROOT, "engines", "llamacpp", "presets.yaml"))
ENGINES = {e["name"]: e for e in CFG["engines"]}
GPU_BUDGET = float(CFG.get("gpu_vram_gb", 96))
IDLE = float(CFG.get("idle_timeout_s", 300))
DEFAULT = CFG.get("default") or next(iter(ENGINES))

# ---------- backend process management ----------
class Backend:
    def __init__(self, spec): self.spec = spec; self.proc = None; self.last = time.time()
    @property
    def base(self): return f"http://127.0.0.1:{self.spec['port']}"
    def cmd(self):
        s = self.spec
        if s["engine"] == "llamacpp":
            d = PRESETS["defaults"]; cls = PRESETS["classes"].get(s.get("class"), {})
            ctx = cls.get("ctx", 8192)
            flags = d["flags"].format(port=s["port"], model=s["model"], ctx=ctx)
            binp = os.path.join(ROOT, "engines", "llamacpp", d["bin"].lstrip("./").replace("/", os.sep))
            return [binp, *flags.split(), *str(cls.get("extra","")).split()]
        # colibrì: OpenAI server built into coli
        coli = os.path.join(ROOT, "c", "coli")
        return [sys.executable, coli, "serve", "--model", s["model"],
                "--host", "127.0.0.1", "--port", str(s["port"])]
    def env(self):
        e = dict(os.environ); e.update({k: str(v) for k, v in (self.spec.get("env") or {}).items()}); return e
    def start(self):
        if self.proc and self.proc.poll() is None: return
        log = open(os.path.join(HERE, f"engine_{self.spec['name']}.log"), "w")
        self.proc = subprocess.Popen(self.cmd(), env=self.env(), stdout=log, stderr=subprocess.STDOUT)
    def ready(self, timeout=180):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None: raise RuntimeError(f"{self.spec['name']} exited; see engine log")
            try:
                urllib.request.urlopen(self.base + "/v1/models", timeout=2); return True
            except (urllib.error.URLError, OSError): time.sleep(1)
        raise TimeoutError(f"{self.spec['name']} not ready in {timeout}s")
    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try: self.proc.wait(10)
            except subprocess.TimeoutExpired: self.proc.kill()

class Arbiter:
    """Ensures at most one iGPU engine runs; CPU engines coexist."""
    def __init__(self): self.live = {}; self.lock = threading.RLock()
    def ensure(self, name):
        with self.lock:
            spec = ENGINES[name]
            if spec.get("device") == "igpu":                 # evict any other igpu owner (LRU)
                for n, b in list(self.live.items()):
                    if ENGINES[n].get("device") == "igpu" and n != name:
                        b.stop(); del self.live[n]
            b = self.live.get(name)
            if not b:
                b = self.live[name] = Backend(spec); b.start(); b.ready()
            b.last = time.time(); return b
    def reap(self):
        while True:
            time.sleep(30)
            with self.lock:
                for n, b in list(self.live.items()):
                    if time.time() - b.last > IDLE:
                        b.stop(); del self.live[n]

ARB = Arbiter()

# ---------- request classification ----------
def classify(body):
    """Explicit registry model id wins; else 'auto' picks by prompt size/task."""
    m = body.get("model", "auto")
    if m in ENGINES: return m
    msgs = body.get("messages") or []
    chars = sum(len(x.get("content","")) for x in msgs if isinstance(x.get("content"), str))
    txt = " ".join(x.get("content","") for x in msgs if isinstance(x.get("content"), str)).lower()
    want_ctx = body.get("max_tokens", 0) or 0
    # escalate to the streamed 744B only when the caller explicitly asks for depth
    if any(k in txt for k in ("prove", "derive", "reason step by step", "744b", "frontier")):
        big = [n for n,e in ENGINES.items() if e.get("class")=="xlarge"]
        if big: return big[0]
    if chars > 24000 or want_ctx > 8000:
        big = [n for n,e in ENGINES.items() if e.get("class") in ("large","xlarge")]
        if big: return big[0]
    return DEFAULT

# ---------- HTTP gateway ----------
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def _json(self, code, obj):
        b = json.dumps(obj).encode(); self.send_response(code)
        self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(b)))
        self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path.rstrip("/") == "/health": return self._json(200, {"status":"ok","live":list(ARB.live)})
        if self.path == "/v1/models":
            return self._json(200, {"object":"list","data":[{"id":n,"object":"model"} for n in ENGINES]})
        self._json(404, {"error":"not found"})
    def do_POST(self):
        if not self.path.startswith("/v1/"): return self._json(404, {"error":"not found"})
        n = int(self.headers.get("Content-Length", 0)); raw = self.rfile.read(n)
        try: body = json.loads(raw or b"{}")
        except json.JSONDecodeError: return self._json(400, {"error":"bad json"})
        try:
            name = classify(body); back = ARB.ensure(name)
        except Exception as ex: return self._json(503, {"error":f"engine '{name}' unavailable: {ex}"})
        self._proxy(back.base + self.path, raw, body.get("stream", False), name)
    def _proxy(self, url, raw, stream, name):
        req = urllib.request.Request(url, data=raw, headers={"Content-Type":"application/json"})
        try: up = urllib.request.urlopen(req, timeout=600)
        except urllib.error.HTTPError as e: up = e
        except (urllib.error.URLError, OSError) as e: return self._json(502, {"error":str(e)})
        self.send_response(up.status if hasattr(up,"status") else 200)
        self.send_header("Content-Type", up.headers.get("Content-Type","application/json"))
        self.send_header("X-Coli-Engine", name)
        if stream: self.send_header("Transfer-Encoding","chunked")
        self.end_headers()
        try:
            while True:
                chunk = up.read(8192)
                if not chunk: break
                if stream: self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                else: self.wfile.write(chunk)
                self.wfile.flush()
            if stream: self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError): pass

def main():
    port = int(os.environ.get("COLI_ROUTER_PORT", 8080))
    threading.Thread(target=ARB.reap, daemon=True).start()
    print(f"colibrì router :{port}  engines={list(ENGINES)}  default={DEFAULT}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()

if __name__ == "__main__":
    main()
