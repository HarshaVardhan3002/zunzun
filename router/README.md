# colibrì router

OpenAI-compatible gateway. One endpoint, many local engines. Classifies each
request, arbitrates the single-owner iGPU, lazy-spawns backends, proxies
streaming, reaps idle engines. Stdlib only.

## Run
```
python router/router.py          # :8080  (COLI_ROUTER_PORT to change)
```
Point any OpenAI client at `http://127.0.0.1:8080/v1`.

## Routing
- `model: "<registry name>"` → forces that engine (e.g. `gptoss-120b`, `glm-5.2`).
- `model: "auto"` → `default` engine, unless: prompt > 24k chars or `max_tokens` > 8k → largest engine; keywords like *prove / derive / reason step by step* → the streamed 744B (`xlarge`).
- `GET /v1/models` lists engines. `GET /health` shows what's live.

## Device arbiter
`device: igpu` engines are single-owner: spawning one LRU-evicts any other igpu engine (they'd fight over the same GTT carveout). `device: cpu` engines (colibrì) coexist. Idle engines die after `idle_timeout_s`.

## Config — registry.yaml
Edit model paths. Each engine: `name, engine (llamacpp|colibri), class, model, device, vram_est_gb, port`, optional `env`. llama.cpp launch flags come from `../engines/llamacpp/presets.yaml` keyed by `class`.

## Limits (MVP)
- No true VRAM accounting yet; arbiter is one-igpu-at-a-time, not a byte budget. Phase 4 adds `vram_est_gb` summation + resource_plan APU awareness.
- Classifier is heuristic. Phase 4 swaps in a feedback-driven policy (tok/s, TTFT, hit-rate).
- YAML loader is a config subset (nested maps, list-of-maps, inline `{}`), not general YAML.
