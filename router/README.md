# colibrì router — the routing OS layer (R5)

One OpenAI-compatible endpoint in front of the colibrì engine. Classifies each
request, dispatches to a backend *profile*, lazy-spawns it via `coli serve`,
arbitrates the single-owner iGPU, proxies streaming, reaps idle backends.
Stdlib only, no dependencies.

## Run
```
python router/router.py            # serves on :8080 (COLI_ROUTER_PORT to change)
```
Point any OpenAI client at `http://127.0.0.1:8080/v1`.

## As-built profiles (registry.yaml)
- `colibri-gpu` — colibrì with the HIP iGPU tier (`COLI_HIP=1 CUDA_DENSE=1 CUDA_EXPERT_GB=N`), `device: igpu`.
- `colibri-cpu` — pure CPU streaming, `device: cpu`.
Both are the same model at different device tiers; the gateway picks per request.
llama.cpp profiles are commented examples for when you add resident small models.

## Routing
- `model: "colibri-gpu"` / `"colibri-cpu"` → that profile explicitly.
- `model: "auto"` → `default` (colibri-gpu), unless the prompt is very long or
  asks for depth (*prove/derive/reason*), which escalates to the largest profile.
- `GET /v1/models` lists profiles. `GET /health` shows what's live.

## iGPU arbiter
`device: igpu` profiles are single-owner: spawning one LRU-evicts any other igpu
profile (they'd fight over the same VRAM carveout). `device: cpu` profiles
coexist. Idle backends die after `idle_timeout_s`.

## Status / caveat
Gateway logic is verified end-to-end (classify, arbiter, spawn-cmd, streaming
proxy). It becomes useful once `coli serve` has a runnable model (a model dir
with `tokenizer.json`) — the tiny TF oracle has no tokenizer, and the 744B model
is 500 GB. With one runnable model the value is CPU-vs-GPU-profile routing; with
several it becomes true multi-model routing.
