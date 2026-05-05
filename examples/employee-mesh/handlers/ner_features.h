/*
 * ner_features.h — tokenisation + spaCy/thinc feature extraction for the
 *                  exported en_core_web_sm NER ONNX model.
 *
 * Pipeline (matches export_ner_onnx.py smoke test exactly):
 *
 *   text
 *     → spacy_tokenize()           split on whitespace / punctuation
 *     → ner_feature_strings()      NORM / PREFIX / SUFFIX / SHAPE per token
 *     → spacy_hash_string()        MurmurHash64A(utf8, seed=0) → uint64 ID
 *     → thinc_hash4()              MurmurHash3_x86_32(id, seed+k) for k=0..3
 *     → (seq, 16) uint64           ready for OrtSession_Run
 *
 * Column order in the feature matrix matches model metadata_props:
 *   col_0=NORM, col_1=PREFIX, col_2=SUFFIX, col_3=SHAPE
 *   each occupies 4 consecutive columns (the 4 thinc hash keys).
 */
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include "../../../vendor/smhasher/MurmurHash2.h"

/* ── limits ───────────────────────────────────────────────────────────────── */

#define NER_MAX_TOKENS  512
#define NER_MAX_TOK_LEN 128
#define NER_N_COLS        4   /* NORM PREFIX SUFFIX SHAPE */
#define NER_N_KEYS        4   /* thinc hashembed keys per column */
#define NER_N_FEATS      16   /* NER_N_COLS * NER_N_KEYS */

/* ── token ────────────────────────────────────────────────────────────────── */

typedef struct {
    char text[NER_MAX_TOK_LEN];
    int  len;
    int  start;   /* byte offset in original text */
} NerToken;

/* ── hash wrappers ────────────────────────────────────────────────────────── */

/* spaCy StringStore: MurmurHash64A(utf8 bytes, seed=1) — matches strings.pyx hash_utf8 */
static inline uint64_t spacy_hash_string(const char *s, int len)
{
    return MurmurHash64A(s, len, 1);
}

/*
 * thinc NumpyOps.hash — verbatim port of MurmurHash3_x86_128_uint64 from
 * thinc/backends/numpy_ops.pyx.  One call per (id, column_seed) pair;
 * fills out[4] with four uint32 embedding keys.
 */
static inline void thinc_hash4(uint64_t val, uint32_t seed, uint64_t out[4])
{
    uint64_t h1 = val;
    h1 *= 0x87c37b91114253d5ULL;
    h1 = (h1 << 31) | (h1 >> 33);
    h1 *= 0x4cf5ad432745937fULL;
    h1 ^= (uint64_t)seed;
    h1 ^= 8;
    uint64_t h2 = (uint64_t)seed;
    h2 ^= 8;
    h1 += h2;
    h2 += h1;
    h1 ^= h1 >> 33; h1 *= 0xff51afd7ed558ccdULL; h1 ^= h1 >> 33;
    h1 *= 0xc4ceb9fe1a85ec53ULL; h1 ^= h1 >> 33;
    h2 ^= h2 >> 33; h2 *= 0xff51afd7ed558ccdULL; h2 ^= h2 >> 33;
    h2 *= 0xc4ceb9fe1a85ec53ULL; h2 ^= h2 >> 33;
    h1 += h2;
    h2 += h1;
    out[0] = h1 & 0xffffffffULL;
    out[1] = h1 >> 32;
    out[2] = h2 & 0xffffffffULL;
    out[3] = h2 >> 32;
}

/* ── shape string ─────────────────────────────────────────────────────────── */

/*
 * Replicates spaCy's SHAPE orthographic feature.
 * Rules: uppercase→'X', lowercase→'x', digit→'d', other→char as-is.
 * Runs of the same shape-char are capped at 4 (matches spaCy orth.py).
 */
static inline void ner_shape(const char *tok, int len, char *out, int *out_len)
{
    char last = '\0';
    int  seq  = 0;
    int  n    = 0;

    for (int i = 0; i < len && n < NER_MAX_TOK_LEN - 1; i++) {
        unsigned char c = (unsigned char)tok[i];
        char sc;
        if (isupper(c))      sc = 'X';
        else if (islower(c)) sc = 'x';
        else if (isdigit(c)) sc = 'd';
        else                  sc = (char)c;

        if (sc == last) { seq++; } else { seq = 0; last = sc; }
        if (seq < 4)
            out[n++] = sc;
    }
    out[n] = '\0';
    *out_len = n;
}

/* ── tokenizer ────────────────────────────────────────────────────────────── */

/*
 * Splits on whitespace and detaches leading/trailing punctuation as separate
 * tokens — a close approximation of spaCy's rule-based English tokenizer.
 *
 * Known gaps vs spaCy en_core_web_sm (cross-checked against test_ner_features.py):
 *
 *  1. ABBREVIATION EXCEPTIONS — spaCy has a built-in list of tokens that are
 *     never split even when they end with a period (Dr., Mr., Ms., Corp., Inc.,
 *     Ltd., Jan., Feb., …).  This tokenizer splits them: "Dr." → ["Dr", "."].
 *     Impact: the NORM/SUFFIX of the abbreviation token differs, so its
 *     embedding lookup is wrong.  For employee-name NER this matters for titles
 *     (Dr., Prof.) preceding names.
 *
 *  2. INFIX SPLITTING — spaCy splits on hyphens, slashes, and similar infixes
 *     when they appear between word characters: "2024-01-15" → ["2024","-","01",
 *     "-","15"], "state-of-the-art" → ["state","-","of","-","the","-","art"].
 *     This tokenizer keeps such chunks whole.
 *     Impact: dates and hyphenated names produce one large token instead of
 *     several small ones, with a different SUFFIX and SHAPE.
 *
 *  3. UTF-8 / NON-ASCII — isspace/ispunct/isupper/islower operate on bytes,
 *     not Unicode code points.  Accented names (Renée, Müller) will have their
 *     accent bytes misclassified.
 *     Impact: NORM (lowercased) and SHAPE may differ for names with diacritics.
 *
 *  4. SPECIAL CASES — spaCy has one-off rules (e.g. "n't" → ["n", "'t"],
 *     possessives "John's" → ["John", "'s"]).  Not implemented here.
 *
 * For the current use-case (masking PII in English employee text) gaps 2–4 have
 * low practical impact: the ONNX model will still produce entity vectors for the
 * tokens it receives; the worst case is that a name split differently gets a
 * slightly lower NER confidence, not that it is silently skipped.
 * Gap 1 (abbreviation exceptions) is the most likely to affect precision for
 * titles directly preceding names — add the exception table if this becomes a
 * problem in production.
 */
static inline int spacy_tokenize(const char *text, NerToken *toks, int max_toks)
{
    int n = 0;
    int i = 0;
    int tlen = (int)strlen(text);

    while (i < tlen && n < max_toks) {
        /* skip whitespace */
        while (i < tlen && isspace((unsigned char)text[i])) i++;
        if (i >= tlen) break;

        /* collect a whitespace-delimited chunk */
        int start = i;
        while (i < tlen && !isspace((unsigned char)text[i])) i++;
        int end = i;
        int clen = end - start;

        /* strip leading punctuation into separate tokens */
        int lp = 0;
        while (lp < clen && ispunct((unsigned char)text[start + lp])) lp++;

        /* strip trailing punctuation (only if some non-punct remains) */
        int rp = 0;
        if (lp < clen)
            while (rp < clen - lp && ispunct((unsigned char)text[end - 1 - rp])) rp++;

        /* emit leading punct tokens one char at a time */
        for (int k = 0; k < lp && n < max_toks; k++) {
            NerToken *t = &toks[n++];
            t->text[0] = text[start + k];
            t->text[1] = '\0';
            t->len     = 1;
            t->start   = start + k;
        }

        /* emit core token */
        int core_start = start + lp;
        int core_len   = clen - lp - rp;
        if (core_len > 0 && n < max_toks) {
            NerToken *t = &toks[n++];
            int cl = core_len < NER_MAX_TOK_LEN - 1 ? core_len : NER_MAX_TOK_LEN - 1;
            memcpy(t->text, text + core_start, cl);
            t->text[cl] = '\0';
            t->len   = cl;
            t->start = core_start;
        }

        /* emit trailing punct tokens one char at a time */
        for (int k = rp - 1; k >= 0 && n < max_toks; k--) {
            NerToken *t = &toks[n++];
            t->text[0] = text[end - 1 - k];
            t->text[1] = '\0';
            t->len     = 1;
            t->start   = end - 1 - k;
        }
    }
    return n;
}

/* ── feature matrix ───────────────────────────────────────────────────────── */

/*
 * Column seeds from the ONNX model metadata_props (col_N = "NAME:nV:seed").
 * Default values match en_core_web_sm; override by reading metadata at runtime.
 */
typedef struct {
    uint32_t norm_seed;
    uint32_t prefix_seed;
    uint32_t suffix_seed;
    uint32_t shape_seed;
} NerSeeds;

/* Seeds from en_core_web_sm hashembed attrs — read from ONNX metadata_props at runtime */
static const NerSeeds NER_DEFAULT_SEEDS = { 8, 9, 10, 11 };

/*
 * Fills feats[seq_len * 16] (row-major) with uint64 hash keys ready for ONNX.
 * feats must have room for n_toks * NER_N_FEATS uint64 values.
 */
static inline void ner_build_feature_matrix(
    const NerToken *toks, int n_toks,
    const NerSeeds *seeds,
    uint64_t       *feats)   /* out: [n_toks][16] */
{
    char  buf[NER_MAX_TOK_LEN];
    int   blen;

    for (int i = 0; i < n_toks; i++) {
        uint64_t *row = feats + i * NER_N_FEATS;
        const char *t = toks[i].text;
        int         l = toks[i].len;

        /* NORM: lowercased text */
        blen = 0;
        for (int j = 0; j < l && j < NER_MAX_TOK_LEN - 1; j++)
            buf[blen++] = (char)tolower((unsigned char)t[j]);
        buf[blen] = '\0';
        uint64_t norm_id = spacy_hash_string(buf, blen);
        thinc_hash4(norm_id, seeds->norm_seed, row + 0);

        /* PREFIX: first character */
        buf[0] = t[0]; buf[1] = '\0'; blen = 1;
        uint64_t pfx_id = spacy_hash_string(buf, blen);
        thinc_hash4(pfx_id, seeds->prefix_seed, row + 4);

        /* SUFFIX: last 3 characters (or fewer if token is short) */
        int sfx_start = l > 3 ? l - 3 : 0;
        blen = l - sfx_start;
        memcpy(buf, t + sfx_start, blen);
        buf[blen] = '\0';
        uint64_t sfx_id = spacy_hash_string(buf, blen);
        thinc_hash4(sfx_id, seeds->suffix_seed, row + 8);

        /* SHAPE */
        ner_shape(t, l, buf, &blen);
        uint64_t shape_id = spacy_hash_string(buf, blen);
        thinc_hash4(shape_id, seeds->shape_seed, row + 12);
    }
}
