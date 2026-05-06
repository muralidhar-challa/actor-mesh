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

/* ── decoder constants and weights ───────────────────────────────────────── */

#define NER_N_MOVES   74
#define NER_N_HIDDEN  64
#define NER_N_PIECES   2
#define NER_N_FEATS_PA 3

/*
 * Decoder weights embedded in ONNX metadata as base64 float32 blobs.
 * pa_b:     (64, 2)   bias added after gather-sum, before maxout
 * linear_W: (74, 64)  NER head weights
 * linear_b: (74,)     NER head bias
 * move_names: array of n_moves C strings (B-ORG, I-ORG, ..., O)
 */
typedef struct {
    float pa_b    [NER_N_HIDDEN * NER_N_PIECES]; /* (64,2) row-major */
    float linear_W[NER_N_MOVES  * NER_N_HIDDEN]; /* (74,64) row-major */
    float linear_b[NER_N_MOVES];
    char  move_names[NER_N_MOVES][24];           /* "B-ORG", "L-PERSON", "O", … */
} NerDecoder;

/* ── model handle ─────────────────────────────────────────────────────────── */

typedef struct {
    const OrtApi     *api;
    OrtEnv           *env;
    OrtSessionOptions *opts;
    OrtSession       *session;
    OrtMemoryInfo    *mem_info;
    NerSeeds          seeds;
    NerDecoder        dec;
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

/* ── base64 decoder (RFC 4648, no padding variants) ───────────────────────── */

static inline int _b64_decode(const char *src, float *dst, int max_floats)
{
    static const signed char T[256] = {
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,
        52,53,54,55,56,57,58,59,60,61,-1,-1,-1, 0,-1,-1,
        -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
        15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,
        -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
        41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    };
    unsigned char *out = (unsigned char *)dst;
    int n_bytes = 0, max_bytes = max_floats * (int)sizeof(float);
    uint32_t acc = 0; int bits = 0;
    for (; *src; src++) {
        int v = T[(unsigned char)*src];
        if (v < 0) continue;  /* skip padding '=' */
        acc = (acc << 6) | (uint32_t)v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            if (n_bytes < max_bytes) out[n_bytes++] = (acc >> bits) & 0xff;
        }
    }
    return n_bytes / (int)sizeof(float);
}

/* Reads pa_b, linear_W, linear_b, and move_names from ONNX metadata. */
static inline void _ner_read_decoder_weights(
    const OrtApi *api, OrtSession *sess, NerDecoder *dec)
{
    memset(dec, 0, sizeof(*dec));

    OrtModelMetadata *meta = NULL;
    OrtStatus *st = api->SessionGetModelMetadata(sess, &meta);
    if (st) { api->ReleaseStatus(st); return; }

    OrtAllocator *alloc = NULL;
    ORT_CHECK(api, api->GetAllocatorWithDefaultOptions(&alloc));

    struct { const char *key; float *dst; int max_floats; } blobs[] = {
        { "pa_b",     dec->pa_b,     NER_N_HIDDEN * NER_N_PIECES },
        { "linear_W", dec->linear_W, NER_N_MOVES  * NER_N_HIDDEN },
        { "linear_b", dec->linear_b, NER_N_MOVES  },
    };
    for (int i = 0; i < 3; i++) {
        char *val = NULL;
        st = api->ModelMetadataLookupCustomMetadataMap(meta, alloc, blobs[i].key, &val);
        if (st) { api->ReleaseStatus(st); continue; }
        if (val) { _b64_decode(val, blobs[i].dst, blobs[i].max_floats); alloc->Free(alloc, val); }
    }

    /* parse comma-separated move names */
    char *moves_val = NULL;
    st = api->ModelMetadataLookupCustomMetadataMap(meta, alloc, "moves", &moves_val);
    if (!st && moves_val) {
        int idx = 0;
        char *p = moves_val, *start = p;
        while (idx < NER_N_MOVES) {
            if (*p == ',' || *p == '\0') {
                int len = (int)(p - start);
                if (len >= (int)sizeof(dec->move_names[idx]))
                    len = (int)sizeof(dec->move_names[idx]) - 1;
                memcpy(dec->move_names[idx], start, len);
                dec->move_names[idx][len] = '\0';
                idx++;
                if (*p == '\0') break;
                start = p + 1;
            }
            p++;
        }
        alloc->Free(alloc, moves_val);
    }

    api->ReleaseModelMetadata(meta);
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
    _ner_read_decoder_weights(m->api, m->session, &m->dec);
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
 * feats:     [n_toks * NER_N_FEATS] uint64  (from ner_build_feature_matrix)
 * n_toks:    sequence length
 * table_out: caller-allocated [(n_toks+1) * NER_N_FEATS_PA * 64 * 2] float32
 *            Precomputed affine table: row 0 = pad, rows 1..n = per-token.
 *            Shape: (n+1, 3, 64, 2) in C order.
 */
static inline void ner_model_run(
    NerModel       *m,
    const uint64_t *feats,
    int             n_toks,
    float          *table_out)
{
    const OrtApi *api = m->api;

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

    float *data = NULL;
    ORT_CHECK(api, api->GetTensorMutableData(out_tensor, (void **)&data));
    /* table shape: (n_toks+1, 3, 64, 2) */
    memcpy(table_out, data,
           (size_t)(n_toks + 1) * NER_N_FEATS_PA * NER_N_HIDDEN * NER_N_PIECES
           * sizeof(float));

    api->ReleaseValue(out_tensor);
    api->ReleaseValue(in_tensor);
    alloc->Free(alloc, out_name);
}

/* ── NER greedy BILUO decoder ─────────────────────────────────────────────── */

/*
 * Feature-slot layout (StateC::set_context_tokens, n==3):
 *   ids[0] = B0          current buffer token
 *   ids[1] = E(0)        entity-start token if open, else -1  → pad row
 *   ids[2] = B0-1        previous token if inside entity, else -1 → pad row
 *
 * Table indexing: row 0 = pad, row t+1 = token t.
 * For slot f and row r: table[r * NF*NO*NP + f * NO*NP .. + NO*NP)
 */

/* dot product of two float vectors, length n */
static inline float _ner_dot(const float *a, const float *b, int n)
{
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

/*
 * Decode the pa table into entity spans using a greedy BILUO transition system.
 *
 * table:     (n_toks+1, 3, 64, 2) float32 from ner_model_run()
 * dec:       decoder weights from NerModel
 * n_toks:    number of tokens
 * spans_out: caller-allocated array, NER_MAX_TOKENS entries is sufficient
 * returns:   number of spans written
 */
static inline int ner_decode_spans(
    const float    *table,
    const NerDecoder *dec,
    int             n_toks,
    NerSpan        *spans_out)
{
    /*
     * Layout constants (row-major):
     *   row stride      = NF * NO * NP = 3 * 64 * 2 = 384 floats
     *   feature stride  = NO * NP      = 64 * 2     = 128 floats
     */
    const int NF = NER_N_FEATS_PA;
    const int NO = NER_N_HIDDEN;
    const int NP = NER_N_PIECES;
    const int row_stride  = NF * NO * NP;   /* 384 */
    const int feat_stride = NO * NP;        /* 128 */

    int n_spans      = 0;
    int state_outside = 1;
    int ent_start    = -1;
    int lbl_idx      = 0;

    float unmaxed[NER_N_HIDDEN * NER_N_PIECES]; /* (64, 2) gather-sum + bias */
    float hidden  [NER_N_HIDDEN];               /* maxout output (64,) */
    float scores  [NER_N_MOVES];               /* linear output (74,) */

    for (int t = 0; t < n_toks; t++) {
        /* token_ids for the three feature slots */
        int id0 = t;                                    /* B0 */
        int id1 = state_outside ? -1 : ent_start;      /* E(0) or pad */
        int id2 = state_outside ? -1 : (t - 1);        /* B0-1 or pad */

        /* row indices into table (row 0 = pad, row t+1 = token t) */
        int r0 = id0 + 1;
        int r1 = (id1 >= 0) ? (id1 + 1) : 0;
        int r2 = (id2 >= 0) ? (id2 + 1) : 0;

        /* gather-sum: unmaxed[o*NP+p] = sum_f table[rf, f, o, p] + pa_b[o,p] */
        for (int i = 0; i < NO * NP; i++) {
            unmaxed[i] =
                table[r0 * row_stride + 0 * feat_stride + i] +
                table[r1 * row_stride + 1 * feat_stride + i] +
                table[r2 * row_stride + 2 * feat_stride + i] +
                dec->pa_b[i];
        }

        /* maxout: hidden[o] = max(unmaxed[o*NP+0], unmaxed[o*NP+1]) */
        for (int o = 0; o < NO; o++) {
            float a = unmaxed[o * NP + 0];
            float b = unmaxed[o * NP + 1];
            hidden[o] = (a > b) ? a : b;
        }

        /* linear(74): scores[c] = dot(linear_W[c], hidden) + linear_b[c] */
        for (int c = 0; c < NER_N_MOVES; c++)
            scores[c] = _ner_dot(dec->linear_W + c * NO, hidden, NO) + dec->linear_b[c];

        /* valid-action mask and argmax */
        int best = -1;
        float best_score = -1e38f;
        if (state_outside) {
            /* B-* (0..17), U-* (54..71), O (73) */
            for (int c = 0; c < 18; c++)
                if (scores[c] > best_score) { best = c; best_score = scores[c]; }
            for (int c = 54; c < 72; c++)
                if (scores[c] > best_score) { best = c; best_score = scores[c]; }
            if (scores[73] > best_score) { best = 73; }
        } else {
            /* I-same (18+lbl), L-same (36+lbl) */
            int ci = 18 + lbl_idx, cl = 36 + lbl_idx;
            best = (scores[ci] > scores[cl]) ? ci : cl;
        }

        /* determine action and update state */
        int act;
        if      (best < 18)  act = 0; /* B */
        else if (best < 36)  act = 1; /* I */
        else if (best < 54)  act = 2; /* L */
        else if (best < 72)  act = 3; /* U */
        else                 act = 4; /* O or MISSING */

        if (act == 0) { /* B */
            ent_start = t;
            lbl_idx   = best % 18;
            state_outside = 0;
        } else if (act == 2) { /* L */
            if (n_spans < NER_MAX_TOKENS) {
                spans_out[n_spans].start = ent_start;
                spans_out[n_spans].end   = t + 1;
                /* label = move_names[best] without leading "L-" */
                strncpy(spans_out[n_spans].label,
                        dec->move_names[best] + 2,
                        sizeof(spans_out[n_spans].label) - 1);
                spans_out[n_spans].label[sizeof(spans_out[n_spans].label)-1] = '\0';
                n_spans++;
            }
            state_outside = 1; ent_start = -1;
        } else if (act == 3) { /* U */
            if (n_spans < NER_MAX_TOKENS) {
                spans_out[n_spans].start = t;
                spans_out[n_spans].end   = t + 1;
                strncpy(spans_out[n_spans].label,
                        dec->move_names[best] + 2,
                        sizeof(spans_out[n_spans].label) - 1);
                spans_out[n_spans].label[sizeof(spans_out[n_spans].label)-1] = '\0';
                n_spans++;
            }
            state_outside = 1; ent_start = -1;
        } else { /* I or O */
            if (act == 4) state_outside = 1; /* O resets; I stays inside */
        }
    }
    return n_spans;
}
