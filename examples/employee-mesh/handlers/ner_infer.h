/*
 * ner_infer.h — ONNX Runtime C API wrapper for the exported en_core_web_sm
 *               tok2vec model (models/ner.onnx).
 *
 * Usage:
 *   NerModel *m = ner_model_load("/path/to/ner.onnx");
 *   NerSeeds  s = ner_model_seeds(m);   // reads seeds from metadata_props
 *
 *   NerToken toks[NER_MAX_TOKENS];
 *   int n = spacy_tokenize(text, toks, NER_MAX_TOKENS);
 *
 *   uint64_t feats[NER_MAX_TOKENS * NER_N_FEATS];
 *   ner_build_feature_matrix(toks, n, &s, feats);
 *
 *   float vecs[NER_MAX_TOKENS * 64];
 *   ner_model_run(m, feats, n, vecs);   // (n, 64) token vectors
 *
 *   NerSpan spans[NER_MAX_TOKENS];
 *   int ns = ner_viterbi(vecs, n, spans);  // entity spans (placeholder)
 *
 *   ner_model_free(m);
 */
#pragma once
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <onnxruntime/onnxruntime_c_api.h>
#include "ner_features.h"

/* ── ORT error helper ─────────────────────────────────────────────────────── */

#define ORT_ABORT(api, status, msg) \
    do { \
        fprintf(stderr, "ner_infer: " msg ": %s\n", \
                (api)->GetErrorMessage(status)); \
        (api)->ReleaseStatus(status); \
        abort(); \
    } while (0)

#define ORT_CHECK(api, expr) \
    do { \
        OrtStatus *_s = (expr); \
        if (_s) ORT_ABORT(api, _s, #expr); \
    } while (0)

/* ── model handle ─────────────────────────────────────────────────────────── */

typedef struct {
    const OrtApi     *api;
    OrtEnv           *env;
    OrtSessionOptions *opts;
    OrtSession       *session;
    OrtMemoryInfo    *mem_info;
    NerSeeds          seeds;
} NerModel;

/* ── entity span (output) ─────────────────────────────────────────────────── */

typedef struct {
    int  start;   /* token index, inclusive */
    int  end;     /* token index, exclusive */
    char label[16];
} NerSpan;

/* ── load ─────────────────────────────────────────────────────────────────── */

/*
 * Parse "NAME:nV:seed" metadata_props values to extract per-column seeds.
 * Falls back to NER_DEFAULT_SEEDS if metadata is absent.
 */
static inline NerSeeds _ner_read_seeds(const OrtApi *api, OrtSession *sess)
{
    NerSeeds s = NER_DEFAULT_SEEDS;

    OrtModelMetadata *meta = NULL;
    OrtStatus *st = api->SessionGetModelMetadata(sess, &meta);
    if (st) { api->ReleaseStatus(st); return s; }

    OrtAllocator *alloc = NULL;
    ORT_CHECK(api, api->GetAllocatorWithDefaultOptions(&alloc));

    /* col_0..col_3 keys */
    uint32_t *dst[4] = { &s.norm_seed, &s.prefix_seed, &s.suffix_seed, &s.shape_seed };
    for (int col = 0; col < 4; col++) {
        char key[8];
        snprintf(key, sizeof(key), "col_%d", col);
        char *val = NULL;
        st = api->ModelMetadataLookupCustomMetadataMap(meta, alloc, key, &val);
        if (st) { api->ReleaseStatus(st); continue; }
        if (!val) continue;
        /* format: "NAME:nV:seed" — seed is the third colon-delimited field */
        char *p = val;
        int colon = 0;
        while (*p && colon < 2) { if (*p++ == ':') colon++; }
        if (colon == 2) *dst[col] = (uint32_t)atoi(p);
        alloc->Free(alloc, val);
    }

    api->ReleaseModelMetadata(meta);
    return s;
}

static inline NerModel *ner_model_load(const char *model_path)
{
    NerModel *m = (NerModel *)calloc(1, sizeof(NerModel));
    if (!m) { perror("ner_model_load: calloc"); abort(); }

    m->api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    ORT_CHECK(m->api, m->api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ner", &m->env));
    ORT_CHECK(m->api, m->api->CreateSessionOptions(&m->opts));
    ORT_CHECK(m->api, m->api->SetIntraOpNumThreads(m->opts, 1));
    /* ORT_ENABLE_EXTENDED triggers GatherSliceToSplitFusion, which tries to
     * rename fused Gather nodes but produces a collision when the exported
     * graph has no explicit node names (all name="").  BASIC covers constant
     * folding and common subexpression elimination — sufficient for this model. */
    ORT_CHECK(m->api, m->api->SetSessionGraphOptimizationLevel(m->opts, ORT_ENABLE_BASIC));
    ORT_CHECK(m->api, m->api->CreateSession(m->env, model_path, m->opts, &m->session));
    ORT_CHECK(m->api, m->api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &m->mem_info));

    m->seeds = _ner_read_seeds(m->api, m->session);
    return m;
}

static inline void ner_model_free(NerModel *m)
{
    if (!m) return;
    m->api->ReleaseMemoryInfo(m->mem_info);
    m->api->ReleaseSession(m->session);
    m->api->ReleaseSessionOptions(m->opts);
    m->api->ReleaseEnv(m->env);
    free(m);
}

/* ── inference ────────────────────────────────────────────────────────────── */

/*
 * Runs the ONNX model.
 * feats:    [n_toks * NER_N_FEATS] uint64  (from ner_build_feature_matrix)
 * n_toks:   sequence length
 * vecs_out: caller-allocated [n_toks * 64] float32  — token vectors
 */
static inline void ner_model_run(
    NerModel       *m,
    const uint64_t *feats,
    int             n_toks,
    float          *vecs_out)
{
    const OrtApi *api = m->api;

    /* input tensor: (n_toks, 16) uint64 */
    int64_t in_shape[2] = { n_toks, NER_N_FEATS };
    OrtValue *in_tensor = NULL;
    ORT_CHECK(api, api->CreateTensorWithDataAsOrtValue(
        m->mem_info,
        (void *)feats,
        (size_t)n_toks * NER_N_FEATS * sizeof(uint64_t),
        in_shape, 2,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
        &in_tensor));

    const char *in_names[]  = { "token_hashes" };
    const char *out_names[] = { NULL };

    /* discover output name from session metadata */
    OrtAllocator *alloc = NULL;
    ORT_CHECK(api, api->GetAllocatorWithDefaultOptions(&alloc));
    char *out_name = NULL;
    ORT_CHECK(api, api->SessionGetOutputName(m->session, 0, alloc, &out_name));
    out_names[0] = out_name;

    OrtValue *out_tensor = NULL;
    ORT_CHECK(api, api->Run(
        m->session, NULL,
        in_names,  (const OrtValue *const *)&in_tensor,  1,
        out_names, 1,
        &out_tensor));

    /* copy result */
    float *data = NULL;
    ORT_CHECK(api, api->GetTensorMutableData(out_tensor, (void **)&data));
    memcpy(vecs_out, data, (size_t)n_toks * 64 * sizeof(float));

    api->ReleaseValue(out_tensor);
    api->ReleaseValue(in_tensor);
    alloc->Free(alloc, out_name);
}

/* ── NER decode ───────────────────────────────────────────────────────────── */

/*
 * The ONNX model outputs (n, 64) token *vectors*, not class logits — it is the
 * tok2vec layer, not the full NER head.  A proper decode requires the
 * precomputable_affine + linear(74) layers (the NER head) which were not
 * exported.  This stub applies a simple cosine-nearest-neighbour to a small
 * set of hand-crafted label prototypes as a placeholder.
 *
 * TODO: export and wire the NER head (linear 74-class layer + Viterbi) so that
 * entity labels come from the real model weights rather than heuristics.
 */
static inline int ner_decode_spans(
    const float    *vecs,      /* (n, 64) */
    int             n_toks,
    const NerToken *toks,
    NerSpan        *spans_out, /* caller-allocated, NER_MAX_TOKENS is enough */
    float           threshold) /* minimum activation to call a token an entity */
{
    (void)vecs; (void)threshold;  /* suppress unused-param warnings until head is wired */

    /* placeholder: zero spans */
    (void)toks;
    (void)spans_out;
    return 0;
}
