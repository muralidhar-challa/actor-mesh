/*
 * handlers/pii-filter.c — PII mask / restore actor
 *
 * PII_MODE=mask    subscribe user_message,sql_result → emit *_masked
 * PII_MODE=restore subscribe sql_query_masked,agent_response_masked   → emit agent_response
 *
 * Env:
 *   PII_MODE              mask | restore
 *   NER_MODEL_PATH        path to models/ner.onnx  (mask mode only)
 *   TOKENMAP_PATH         LMDB directory (shared between mask and restore)
 *   ACTOR_CORRELATION_ID  session key
 *
 * Wire format (actor runtime):
 *   stdin  → raw mpack bytes
 *   stdout → "topic\n" + raw mpack bytes
 *
 * Build: see Makefile
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <lmdb.h>
#include "mpack.h"
#include "ner_infer.h"   /* NerModel, NerSpan, … */

/* ── sizing ──────────────────────────────────────────────────────────────── */

#define MAX_IN       (256 * 1024)
#define MAX_OUT      (256 * 1024)
#define MAX_STR      (32  * 1024)
#define MAX_MASKED   (64  * 1024)

static char g_in [MAX_IN];
static char g_out[MAX_OUT];

/* NER buffers — static to avoid huge stack VLAs (770KB table) */
static NerToken     g_ner_toks[NER_MAX_TOKENS];
static uint64_t     g_ner_feats[NER_MAX_TOKENS * NER_N_FEATS];
static float        g_ner_table[(NER_MAX_TOKENS + 1) * NER_N_FEATS_PA * NER_N_HIDDEN * NER_N_PIECES];

/* ── LMDB helpers ────────────────────────────────────────────────────────── */

static MDB_env *g_env;
static MDB_dbi  g_dbi;

static void lmdb_open(void)
{
    const char *path = getenv("TOKENMAP_PATH");
    if (!path) { fprintf(stderr, "pii-filter: TOKENMAP_PATH not set\n"); exit(1); }

    if (mdb_env_create(&g_env)) { perror("mdb_env_create"); exit(1); }
    mdb_env_set_mapsize(g_env, 64UL * 1024 * 1024);
    mdb_env_set_maxdbs(g_env, 2);
    if (mdb_env_open(g_env, path, 0, 0664)) {
        perror("mdb_env_open"); exit(1);
    }

    MDB_txn *txn;
    mdb_txn_begin(g_env, NULL, 0, &txn);
    mdb_dbi_open(txn, "tokenmap", MDB_CREATE, &g_dbi);
    mdb_txn_commit(txn);
}

/*
 * Look up or create a TOK_{LABEL}_{NNNN} placeholder for (entity_text, label).
 * Returns the placeholder in out_tok (caller must supply ≥ 32 bytes).
 */
static void get_or_create_tok(const char *corr, const char *text,
                               const char *label, char *out_tok)
{
    /* key: "{corr}:text:{entity_text}" */
    char lkey[512];
    snprintf(lkey, sizeof(lkey), "%s:text:%s", corr, text);

    MDB_txn *txn;
    mdb_txn_begin(g_env, NULL, 0, &txn);

    MDB_val k = { strlen(lkey), lkey };
    MDB_val v;

    if (mdb_get(txn, g_dbi, &k, &v) == 0) {
        /* already exists */
        int len = v.mv_size < 31 ? (int)v.mv_size : 31;
        memcpy(out_tok, v.mv_data, len);
        out_tok[len] = '\0';
        mdb_txn_abort(txn);
        return;
    }
    mdb_txn_abort(txn);

    /* allocate new token */
    mdb_txn_begin(g_env, NULL, 0, &txn);

    char ckey[256];
    snprintf(ckey, sizeof(ckey), "%s:__count__", corr);
    MDB_val ck = { strlen(ckey), ckey };
    MDB_val cv;
    int n = 0;
    if (mdb_get(txn, g_dbi, &ck, &cv) == 0) {
        char tmp[16]; int l = cv.mv_size < 15 ? (int)cv.mv_size : 15;
        memcpy(tmp, cv.mv_data, l); tmp[l] = '\0';
        n = atoi(tmp);
    }
    /* format: TOK_PERSON_0000, TOK_ORG_0001, etc. */
    snprintf(out_tok, 32, "TOK_%s_%04d", label, n);
    char ns[16]; snprintf(ns, sizeof(ns), "%d", n + 1);
    MDB_val nv = { strlen(ns), ns };
    mdb_put(txn, g_dbi, &ck, &nv, 0);

    /* forward: corr:text:entity → TOK */
    MDB_val fk = { strlen(lkey), lkey };
    MDB_val fv = { strlen(out_tok), out_tok };
    mdb_put(txn, g_dbi, &fk, &fv, 0);

    /* reverse: corr:tok:TOK → original text (store "label:text") */
    char rkey[512];
    snprintf(rkey, sizeof(rkey), "%s:tok:%s", corr, out_tok);
    char rval[MAX_STR + 32];
    snprintf(rval, sizeof(rval), "%s:%s", label, text);
    MDB_val rk = { strlen(rkey), rkey };
    MDB_val rv = { strlen(rval), rval };
    mdb_put(txn, g_dbi, &rk, &rv, 0);

    mdb_txn_commit(txn);
}

/* Lookup original text for a TOK_NNNN placeholder. Returns 0 on success. */
static int tok_lookup(const char *corr, const char *tok,
                      char *out_text, int out_max)
{
    char rkey[256];
    snprintf(rkey, sizeof(rkey), "%s:tok:%s", corr, tok);
    MDB_txn *txn;
    mdb_txn_begin(g_env, NULL, MDB_RDONLY, &txn);
    MDB_val k = { strlen(rkey), rkey };
    MDB_val v;
    int rc = mdb_get(txn, g_dbi, &k, &v);
    if (rc == 0) {
        /* stored as "label:text" — skip label prefix */
        const char *p = (const char *)v.mv_data;
        int sz = (int)v.mv_size;
        const char *colon = (const char *)memchr(p, ':', sz);
        if (colon) { p = colon + 1; sz -= (int)(p - (const char *)v.mv_data); }
        int len = sz < out_max - 1 ? sz : out_max - 1;
        memcpy(out_text, p, len);
        out_text[len] = '\0';
    }
    mdb_txn_abort(txn);
    return rc;
}

/* ── NER model (mask mode) ───────────────────────────────────────────────── */

static NerModel *g_ner;

static void ner_init(void)
{
    const char *path = getenv("NER_MODEL_PATH");
    if (!path) { fprintf(stderr, "pii-filter: NER_MODEL_PATH not set\n"); exit(1); }
    g_ner = ner_model_load(path);
}

/*
 * Mask all named entities in src, write result to dst (max dst_max bytes).
 * Uses LMDB to assign/recall TOK_NNNN placeholders.
 */
static void mask_string(const char *src, char *dst, int dst_max,
                        const char *corr)
{
    NerToken *toks = g_ner_toks;
    int n = spacy_tokenize(src, toks, NER_MAX_TOKENS);
    if (n <= 0) {
        strncpy(dst, src, dst_max - 1);
        dst[dst_max - 1] = '\0';
        return;
    }

    uint64_t *feats = g_ner_feats;
    ner_build_feature_matrix(toks, n, &g_ner->seeds, feats);

    float *table = g_ner_table;
    ner_model_run(g_ner, feats, n, table);

    NerSpan spans[NER_MAX_TOKENS];
    int ns = ner_decode_spans(table, &g_ner->dec, n, spans);

    if (ns == 0) {
        strncpy(dst, src, dst_max - 1);
        dst[dst_max - 1] = '\0';
        return;
    }

    /* reconstruct masked string using byte offsets from NerToken */
    int dpos = 0, last_end = 0;
    for (int i = 0; i < ns; i++) {
        int char_start = toks[spans[i].start].start;
        int char_end   = (spans[i].end <= n)
            ? toks[spans[i].end - 1].start + toks[spans[i].end - 1].len
            : (int)strlen(src);

        /* copy unchanged text before this entity */
        int pre = char_start - last_end;
        if (pre > 0 && dpos + pre < dst_max - 1) {
            memcpy(dst + dpos, src + last_end, pre);
            dpos += pre;
        }

        /* entity text */
        char ent[512] = "";
        int elen = char_end - char_start;
        if (elen > 0 && elen < (int)sizeof(ent)) {
            memcpy(ent, src + char_start, elen);
            ent[elen] = '\0';
        }

        /* get or create placeholder */
        char tok[10];
        get_or_create_tok(corr, ent, spans[i].label, tok);

        int tlen = (int)strlen(tok);
        if (dpos + tlen < dst_max - 1) {
            memcpy(dst + dpos, tok, tlen);
            dpos += tlen;
        }
        last_end = char_end;
    }
    /* tail */
    int tail = (int)strlen(src) - last_end;
    if (tail > 0 && dpos + tail < dst_max - 1) {
        memcpy(dst + dpos, src + last_end, tail);
        dpos += tail;
    }
    dst[dpos] = '\0';
}

/*
 * Scan src for TOK_NNNN patterns, replace with original text via LMDB.
 */
static void restore_string(const char *src, char *dst, int dst_max,
                            const char *corr)
{
    int si = 0, di = 0;
    int slen = (int)strlen(src);
    while (si < slen && di < dst_max - 1) {
        /*
         * Match TOK_{LABEL}_{NNNN}:
         *   "TOK_" + one or more uppercase letters + "_" + four digits
         */
        if (si + 9 <= slen &&
            src[si]=='T' && src[si+1]=='O' && src[si+2]=='K' && src[si+3]=='_' &&
            isupper((unsigned char)src[si+4]))
        {
            int j = si + 4;
            while (j < slen && isupper((unsigned char)src[j])) j++;
            if (j < slen && src[j] == '_' &&
                j+5 <= slen &&
                isdigit((unsigned char)src[j+1]) &&
                isdigit((unsigned char)src[j+2]) &&
                isdigit((unsigned char)src[j+3]) &&
                isdigit((unsigned char)src[j+4]))
            {
                int toklen = j + 5 - si;
                char tok[32];
                if (toklen < (int)sizeof(tok)) {
                    memcpy(tok, src + si, toklen); tok[toklen] = '\0';
                    char orig[MAX_STR];
                    if (tok_lookup(corr, tok, orig, sizeof(orig)) == 0) {
                        int ol = (int)strlen(orig);
                        if (di + ol < dst_max - 1) {
                            memcpy(dst + di, orig, ol);
                            di += ol;
                        }
                    } else {
                        memcpy(dst + di, tok, toklen);
                        di += toklen;
                    }
                    si += toklen;
                    continue;
                }
            }
        }
        dst[di++] = src[si++];
    }
    dst[di] = '\0';
}

/* ── mpack recursive walk ────────────────────────────────────────────────── */

typedef void (*str_fn)(const char *in, char *out, int out_max, const char *corr);

static void walk_and_rewrite(mpack_reader_t *r, mpack_writer_t *w,
                              str_fn fn, const char *corr)
{
    mpack_tag_t tag = mpack_read_tag(r);
    if (mpack_reader_error(r)) return;

    switch (tag.type) {
    case mpack_type_str: {
        uint32_t len = tag.v.l;
        char *buf = (char *)malloc(len + 1);
        mpack_read_bytes(r, buf, len);
        mpack_done_str(r);
        buf[len] = '\0';
        char out[MAX_MASKED];
        fn(buf, out, sizeof(out), corr);
        mpack_write_str(w, out, (uint32_t)strlen(out));
        free(buf);
        break;
    }
    case mpack_type_map: {
        uint32_t count = tag.v.n;
        mpack_start_map(w, count);
        for (uint32_t i = 0; i < count; i++) {
            walk_and_rewrite(r, w, fn, corr);   /* key */
            walk_and_rewrite(r, w, fn, corr);   /* value */
        }
        mpack_finish_map(w);
        break;
    }
    case mpack_type_array: {
        uint32_t count = tag.v.n;
        mpack_start_array(w, count);
        for (uint32_t i = 0; i < count; i++)
            walk_and_rewrite(r, w, fn, corr);
        mpack_finish_array(w);
        break;
    }
    case mpack_type_bool:
        mpack_write_bool(w, tag.v.b);
        break;
    case mpack_type_int:
        mpack_write_i64(w, tag.v.i);
        break;
    case mpack_type_uint:
        mpack_write_u64(w, tag.v.u);
        break;
    case mpack_type_float:
        mpack_write_float(w, tag.v.f);
        break;
    case mpack_type_double:
        mpack_write_double(w, tag.v.d);
        break;
    case mpack_type_nil:
        mpack_write_nil(w);
        break;
    default:
        mpack_write_nil(w);
        break;
    }
}

/* ── main ────────────────────────────────────────────────────────────────── */

int main(void)
{
    const char *mode  = getenv("PII_MODE");
    const char *corr  = getenv("ACTOR_CORRELATION_ID");
    if (!mode) mode   = "mask";
    if (!corr) corr   = "default";

    int is_mask = strcmp(mode, "mask") == 0;

    lmdb_open();
    if (is_mask) ner_init();

    /* read stdin */
    size_t nread = fread(g_in, 1, sizeof(g_in) - 1, stdin);
    g_in[nread] = '\0';

    /* parse message type for topic routing */
    mpack_reader_t probe;
    mpack_reader_init_data(&probe, g_in, nread);
    char msg_type[64] = "";
    mpack_tag_t top = mpack_read_tag(&probe);
    if (top.type == mpack_type_map) {
        for (uint32_t i = 0; i < top.v.n; i++) {
            mpack_tag_t kt = mpack_read_tag(&probe);
            if (kt.type == mpack_type_str) {
                char kbuf[64]; uint32_t kl = kt.v.l < 63 ? kt.v.l : 63;
                mpack_read_bytes(&probe, kbuf, kl); mpack_done_str(&probe);
                kbuf[kl] = '\0';
                mpack_tag_t vt = mpack_read_tag(&probe);
                if (strcmp(kbuf, "type") == 0 && vt.type == mpack_type_str) {
                    uint32_t vl = vt.v.l < 63 ? vt.v.l : 63;
                    mpack_read_bytes(&probe, msg_type, vl);
                    mpack_done_str(&probe);
                    msg_type[vl] = '\0';
                    break;
                } else if (vt.type == mpack_type_str) {
                    mpack_skip_bytes(&probe, vt.v.l); mpack_done_str(&probe);
                } else {
                    mpack_discard(&probe);
                }
            } else {
                mpack_discard(&probe); mpack_discard(&probe);
            }
        }
    }

    /* determine output topic */
    char topic[128];
    if (is_mask) {
        snprintf(topic, sizeof(topic), "%s_masked", msg_type);
    } else {
        /* restore: strip _masked suffix if present */
        size_t mlen = strlen(msg_type);
        if (mlen > 7 && strcmp(msg_type + mlen - 7, "_masked") == 0)
            snprintf(topic, sizeof(topic), "%.*s", (int)(mlen - 7), msg_type);
        else
            snprintf(topic, sizeof(topic), "%s", msg_type);
    }

    /* rewrite message */
    mpack_reader_t reader;
    mpack_reader_init_data(&reader, g_in, nread);

    mpack_writer_t writer;
    mpack_writer_init(&writer, g_out, sizeof(g_out));

    str_fn fn = is_mask ? mask_string : restore_string;
    walk_and_rewrite(&reader, &writer, fn, corr);

    size_t out_size = mpack_writer_buffer_used(&writer);

    /* emit topic\npayload */
    fwrite(topic, 1, strlen(topic), stdout);
    fputc('\n', stdout);
    fwrite(g_out, 1, out_size, stdout);
    fflush(stdout);

    if (is_mask) ner_model_free(g_ner);
    mdb_env_close(g_env);
    return 0;
}
