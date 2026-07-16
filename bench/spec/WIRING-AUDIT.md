# MTP head wiring audit — glm.c vs reference implementations

**Date:** 2026-07-16 (plan Task 7)
**Local side:** `c/glm.c` @ 89d1ee9 — `mtp_draft` (~1616), `mtp_absorb` (~1654), `step()` prefill absorb (~1572), tensor loads (~835).
**References used:**
- `zai-org/GLM-5.2` HF repo ships **no modeling .py** (native transformers `glm_moe_dsa` class; transformers has no MTP generation path) → fallback per plan:
- **vLLM** `vllm/model_executor/models/deepseek_mtp.py` (DeepSeek-V3 lineage — the formulation GLM's MTP inherits) and `glm4_moe_mtp.py` (GLM MoE variant), plus the shared proposer `vllm/v1/spec_decode/llm_base_proposer.py` (EagleProposer pathway), all @ main 2026-07-16.
- **SGLang** `python/sglang/srt/models/glm4_moe_nextn.py` @ main 2026-07-16 — the only GLM-specific implementation that actually runs **multi-step** MTP in production.

---

## Seam 1 — h fed to the head: pre or post final `model.norm`?

| reference | glm.c |
|---|---|
| vLLM target model output hidden (what `propose()` receives as `target_hidden_states` and feeds as `previous_hidden_states` on the first draft step) is the model's **post-`model.norm`** output; DeepseekV2Model/Glm4MoeModel `forward` return `hidden_states, _ = self.norm(hidden_states, residual)`. | `mtp_draft` line ~1629: `if(g==0 && !prenorm) rmsnorm(h, h, m->final_norm, D, c->eps);  /* h vero: post model.norm */` — hlast is captured **pre-norm** in `step_all` (~1587), so applying `final_norm` here lands on post-norm. `mtp_absorb` ~1664 same (`rmsnorm(hf, x_i, final_norm)` then `hnorm`). |

**Verdict: CONFIRMED-CORRECT.** The existing `vLLM: h POST model.norm` comment claim checks out. `MTP_PRENORM=1` moves *away* from the reference — keep in the sweep only as a control row.

## Seam 2 — concat order into `eh_proj`

| reference | glm.c |
|---|---|
| vLLM deepseek_mtp.py ~133: `hidden_states = self.eh_proj(torch.cat([inputs_embeds, previous_hidden_states], dim=-1))` with `inputs_embeds = self.enorm(inputs_embeds)`, `previous_hidden_states = self.hnorm(...)`. Identical in glm4_moe_mtp.py ~118 and SGLang glm4_moe_nextn.py ~93: `torch.cat((self.enorm(hidden_states), self.hnorm(spec_info.hidden_states)))`. | `mtp_draft` ~1632 default: `memcpy(cat, x, ...); memcpy(cat+D, h, ...)` — **embedding first**, hidden second. Same in `mtp_absorb` ~1667. |

**Verdict: CONFIRMED-CORRECT.** All three references put the normed embedding first. `MTP_SWAP=1` is wrong — control row only.

## Seam 3 — norm-weight assignment (`enorm` ↔ embedding, `hnorm` ↔ hidden)

| reference | glm.c |
|---|---|
| vLLM/SGLang: `self.enorm(inputs_embeds)`, `self.hnorm(previous_hidden_states)`; logits head norm is `shared_head.norm`. | Loads (~836): `m->enorm=ld("enorm.weight")` applied to `x` (embedding, ~1628); `m->hnorm=ld("hnorm.weight")` applied to `h` (~1630); `m->mtp_norm=ld("shared_head.norm.weight")` applied before `lm_head` (~1641). |

**Verdict: CONFIRMED-CORRECT.** Checkpoint tensor names map identically.

## Seam 4 — chained drafting (depth ≥ 2): what h is fed back?

| reference | glm.c |
|---|---|
| vLLM deepseek_mtp.py ~149: <br>`# Recycle the post-final-norm hidden into the next draft step.`<br>`# compute_logits applies shared_head (== final norm) to the pre-norm`<br>`# element, so logits and the recycle each get exactly one final-norm.`<br>`# Matches SGLang's deepseek_nextn.`<br>`return hidden_states, self.shared_head(hidden_states)` — the proposer feeds the **second** element (post-`shared_head.norm`) back as the next step's `previous_hidden_states`. SGLang glm4_moe_nextn.py ~111 returns `self.shared_head.norm(hidden_states)` — same: the recycled hidden is **normed**. | `mtp_draft` ~1646: `draft[n++]=t2; tok=t2; memcpy(h, hx, D*sizeof(float));` — feeds back the **raw** block output `hx`; the next iteration applies only `hnorm` (the `g==0` condition skips `final_norm` for g≥1, and `mtp_norm` is applied only on the logits path `row`). |

**Verdict: DISCREPANCY (fix = feed `mtp_norm(hx)` back, i.e. `memcpy(h, row, ...)` — `row` already holds exactly that).** Two authoritative multi-step implementations agree; vLLM's glm4_moe_mtp.py returns the raw hidden but vLLM runs GLM MTP at `num_speculative_tokens=1` where the recycle path is dead code — weak counter-evidence. This discrepancy only bites at depth ≥ 2, which matches the symptom profile (28% aggregate acceptance at DRAFT=3; the Task-5 depth histogram will show whether d2/d3 collapse while d1 holds). **Sweep candidate: `MTP_CHAINNORM=1` (tag `audit1`).**

## Seam 5 — position / RoPE indices for the head's KV

| reference | glm.c |
|---|---|
| llm_base_proposer.py `set_inputs_first_pass` ~838: `# Simply rotate the input ids and leave the positions unchanged` … ~846: `# Shift the input ids by one token.` `self.input_ids[:num_tokens-1] = target_token_ids[1:]` … `self._set_positions(num_tokens, target_positions)` — slot i holds pair (emb(t_{i+1}), h_i) at **position i** (the h's position). Multi-step: positions incremented by 1 per draft step. | `mtp_draft` ~1626: `pos = p+g` with `p = kv-1` — the pair (h@p, emb(tok@p+1)) sits at **position p** (the h's position), +1 per chained step. `mtp_absorb` ~1671: pair i at `pos_base+i` = h's position. Draft-time and absorb-time agree. |

**Verdict: CONFIRMED-CORRECT.** Same convention (position of h, input ids shifted by one). `kv_start` reset semantics (`<0 || >pos` → set) only ever widens the window backward; no off-by-one vs reference.

## Seam 6 — head KV prefill / coverage

| reference | glm.c |
|---|---|
| llm_base_proposer.py ~607: `# The prefill forward pass above already ran to keep the drafter KV cache in sync` — the drafter (MTP layer) runs over **all** target tokens including the prompt. Also deepseek_mtp.py ~128: `# masking inputs at position 0, as not needed by MTP` — `inputs_embeds = torch.where(positions.unsqueeze(-1)==0, 0, inputs_embeds)`. | `step()` ~1572: `if(m->has_mtp && S>=2 && g_draft>0) mtp_absorb(m, ids+1, x, S-1, pos_base);` — the prompt **is** absorbed into the head KV at prefill (the doc comment at ~1612 "niente prefill" is stale — fixed in this commit). The S−1th pair of a chunk (h@last, next chunk/generated token) is written live by the next `mtp_draft`/absorb, so single-turn coverage is complete. |

**Verdict: CONFIRMED-CORRECT with two minor nuances.**
- (a) **Position-0 embedding mask missing:** reference zeroes the embedding half of the pair at absolute position 0; glm.c absorbs `(enorm(emb(ids[1])), hnorm(h0))` unmasked. One KV row affected. **Sweep candidate: `MTP_MASK0=1` (tag `audit2`)** — expected small/no effect, cheap to test.
- (b) **Multi-turn serve staleness:** at each turn boundary, position len−1's head-KV row keeps the last *speculative* pair written by `mtp_draft` (never corrected by an absorb, since the next turn's prefill absorb starts at `pos_base=len`). One possibly-wrong row per turn; does not affect `run_text` probes. Recorded, not swept (needs a serve-path fix, out of probe scope).

## Seam 7 — verify-time absorb formulation == draft-time

| reference | glm.c |
|---|---|
| One formulation in the reference (same module runs prefill/verify/draft). | `mtp_absorb` (~1660–1668) repeats `enorm(emb)` / `final_norm→hnorm(h)` / same concat order / same `eh_proj` / same layer at the same positions as `mtp_draft` g==0. The h inputs are true hiddens (`m->h_all`, pre-norm) so the chaining discrepancy of seam 4 does not apply here. |

**Verdict: CONFIRMED-CORRECT.**

---

## Verdict summary

| # | seam | verdict |
|---|------|---------|
| 1 | h post `model.norm` (step 1) | CONFIRMED-CORRECT |
| 2 | concat `[enorm(emb); hnorm(h)]` | CONFIRMED-CORRECT |
| 3 | norm-weight assignment | CONFIRMED-CORRECT |
| 4 | chained h for depth ≥ 2 | **DISCREPANCY** — fix = recycle `mtp_norm(hx)`; seam `MTP_CHAINNORM=1`, sweep tag `audit1` |
| 5 | position indices | CONFIRMED-CORRECT |
| 6 | head KV prefill | CONFIRMED-CORRECT (minor: pos-0 mask → `MTP_MASK0=1` tag `audit2`; serve turn-boundary staleness recorded) |
| 7 | absorb == draft | CONFIRMED-CORRECT |

**Interpretation for Phase 2:** the 2×2 `MTP_PRENORM`×`MTP_SWAP` grid is expected to *lose* everywhere (both flags move away from the reference — they remain useful as controls). The live candidate is `audit1` (`MTP_CHAINNORM=1`): it is the only wiring difference on the multi-step path, and its failure mode (un-normed hidden re-entering `hnorm` with a drifting scale) compounds with depth — consistent with 28% aggregate acceptance at DRAFT=3 while DeepSeek-class heads report ~85–90% at depth 1. If `audit1` moves d2/d3 but d1 stays low, the remaining gap is head quantization (int8) per `zun` ~515.
