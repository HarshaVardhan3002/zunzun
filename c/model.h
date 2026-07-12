/* ModelProvider — ARCHITECTURE.md subsystem 1 (storage / naming).
 *
 * The ONLY model-family-specific knowledge in the runtime: how a
 * (layer, expert, projection) triple maps to an on-disk tensor name, plus the
 * model's shape metadata. Everything downstream (ExpertCache, Scheduler, the
 * CPU/GPU backends) is family-agnostic and consumes this interface.
 *
 * R0: extracted verbatim from the strings glm.c used to build inline. GLM-5.2
 * adapter is mp_glm(); a DeepSeek/Qwen adapter (R4) is just another initializer.
 * Names produced here are byte-identical to the pre-R0 engine. */
#ifndef MODEL_H
#define MODEL_H
#include <stdio.h>
#include <stddef.h>
#include <string.h>

#define MP_NPROJ    3      /* gate / up / down */
#define MP_NAME_MAX 288
#define MP_QSNAME_MAX (MP_NAME_MAX+16)

typedef struct {
    /* naming (family-specific) */
    const char *expert_tmpl;            /* printf(layer %d, eid %d, proj %s) -> ".weight" name */
    const char *proj[MP_NPROJ];         /* projection stems, load/compute order */
    const char *qs_suffix;              /* per-row scale twin, e.g. ".qs" */
    /* metadata() — model shape, filled from config.json by the caller */
    int n_layers, experts_per_layer, top_k, first_dense, moe_inter, hidden;
} ModelProvider;

/* GLM-5.2 (glm_moe_dsa). Naming only; caller copies metadata from Cfg. */
static inline ModelProvider mp_glm(void){
    ModelProvider mp;
    mp.expert_tmpl = "model.layers.%d.mlp.experts.%d.%s.weight";
    mp.proj[0]="gate_proj"; mp.proj[1]="up_proj"; mp.proj[2]="down_proj";
    mp.qs_suffix = ".qs";
    mp.n_layers=mp.experts_per_layer=mp.top_k=mp.first_dense=mp.moe_inter=mp.hidden=0;
    return mp;
}

/* expert_ref name for projection k of (layer, eid). */
static inline void mp_expert_name(const ModelProvider *mp, int layer, int eid, int k,
                                  char *out, size_t n){
    snprintf(out, n, mp->expert_tmpl, layer, eid, mp->proj[k]);
}
/* all MP_NPROJ names at once (the coalesced-read set). */
static inline void mp_expert_names(const ModelProvider *mp, int layer, int eid,
                                   char out[MP_NPROJ][MP_NAME_MAX]){
    for(int k=0;k<MP_NPROJ;k++) mp_expert_name(mp, layer, eid, k, out[k], MP_NAME_MAX);
}
/* per-row scale twin name for a base tensor name. Bounded (memcpy, not snprintf)
 * so a variable-length suffix doesn't trip -Wformat-truncation. */
static inline void mp_qs_name(const ModelProvider *mp, const char *base, char *out, size_t n){
    if(!n) return;
    size_t bl=strlen(base), sl=strlen(mp->qs_suffix);
    if(bl>=n) bl=n-1;
    if(bl+sl>=n) sl=n-1-bl;
    memcpy(out, base, bl); memcpy(out+bl, mp->qs_suffix, sl); out[bl+sl]='\0';
}

#endif
