#define _POSIX_C_SOURCE 200809L
/*
 * handlers/llm-agent.c — agentic ReAct loop handler
 *
 * Receives: mpack user_message  {type, query}
 *           mpack sql_result    {type, rows:[{col:val,...},...]}
 *
 * Emits:    sql_query\n      mpack {type:"sql_query", sql:"..."}
 *        or agent_response\n mpack {type:"agent_response", answer:"..."}
 *
 * Memory: LMDB keyed by ACTOR_CORRELATION_ID
 *   Between turns: {"history":[{role,content},...]}
 *   During ReAct:  {"history":[...], "messages":[...], "round":N}
 *
 * The agent accumulates conversation history in LMDB across turns.
 * TUI sends simple {query} payload; correlation_id in the tuple header
 * identifies the session and keys the LMDB memory.
 *
 * Env: OLLAMA_URL, OLLAMA_MODEL, EMPLOYEE_DB, ACTOR_LMDB_PATH,
 *      ACTOR_CORRELATION_ID
 * Deps: libcurl, liblmdb, libsqlite3, mpack, cJSON
 */

#include "cJSON.h"
#include "mpack.h"
#include <curl/curl.h>
#include <lmdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_IN      (256 * 1024)
#define MAX_RESP    (2   * 1024 * 1024)
#define MAX_STATE   (512 * 1024)
#define MAX_SYS     (128 * 1024)
#define MAX_ROWS_J  (128 * 1024)
#define MAX_OUT     (1   * 1024 * 1024)
#define MAX_ROUNDS  10

static char   g_in[MAX_IN];
static char   g_resp[MAX_RESP];
static size_t g_resp_len;
static char   g_state_buf[MAX_STATE];
static char   g_sys[MAX_SYS];
static char   g_query[4096];
static char   g_rows_json[MAX_ROWS_J];
static char   g_out[MAX_OUT];

/* ── curl: write into static g_resp ─────────────────────────────────────── */

static size_t curl_cb(void* data, size_t sz, size_t nmemb, void* userp) {
    (void)userp;
    size_t add = sz * nmemb;
    if (g_resp_len + add + 1 > MAX_RESP) return 0;
    memcpy(g_resp + g_resp_len, data, add);
    g_resp_len += add;
    g_resp[g_resp_len] = '\0';
    return add;
}

/* ── LMDB helpers ────────────────────────────────────────────────────────── */

static MDB_env* lmdb_open(void) {
    const char* path = getenv("ACTOR_LMDB_PATH");
    if (!path) return NULL;
    MDB_env* env;
    if (mdb_env_create(&env)) return NULL;
    mdb_env_set_mapsize(env, 64UL * 1024 * 1024);
    mdb_env_set_maxdbs(env, 3);
    if (mdb_env_open(env, path, 0, 0664)) { mdb_env_close(env); return NULL; }
    return env;
}

static const char* state_load(MDB_env* env, const char* key) {
    MDB_txn* txn;
    if (mdb_txn_begin(env, NULL, 0, &txn)) return NULL;
    MDB_dbi dbi;
    if (mdb_dbi_open(txn, "state", 0, &dbi)) { mdb_txn_abort(txn); return NULL; }
    MDB_val k = { strlen(key), (void*)key };
    MDB_val v;
    const char* out = NULL;
    if (mdb_get(txn, dbi, &k, &v) == 0 && v.mv_size < MAX_STATE) {
        memcpy(g_state_buf, v.mv_data, v.mv_size);
        g_state_buf[v.mv_size] = '\0';
        out = g_state_buf;
    }
    mdb_txn_abort(txn);
    return out;
}

static void state_save(MDB_env* env, const char* key, const char* val) {
    MDB_txn* txn;
    if (mdb_txn_begin(env, NULL, 0, &txn)) return;
    MDB_dbi dbi;
    if (mdb_dbi_open(txn, "state", MDB_CREATE, &dbi)) { mdb_txn_abort(txn); return; }
    MDB_val k = { strlen(key), (void*)key };
    MDB_val v = { strlen(val), (void*)val };
    mdb_put(txn, dbi, &k, &v, 0);
    mdb_txn_commit(txn);
}

/* Returns 1 if this tuple_id was already seen (duplicate), 0 if fresh.
 * Uses MDB_NOOVERWRITE so concurrent instances also dedup correctly. */
static int dedup_check(MDB_env* env, const char* tuple_id) {
    if (!env || !tuple_id || !*tuple_id) return 0;
    MDB_txn* txn;
    if (mdb_txn_begin(env, NULL, 0, &txn)) return 0;
    MDB_dbi dbi;
    if (mdb_dbi_open(txn, "dedup", MDB_CREATE, &dbi)) {
        mdb_txn_abort(txn); return 0;
    }
    MDB_val k = { strlen(tuple_id), (void*)tuple_id };
    MDB_val v = { 1, "1" };
    int rc = mdb_put(txn, dbi, &k, &v, MDB_NOOVERWRITE);
    mdb_txn_commit(txn);
    return rc == MDB_KEYEXIST;
}

/* ── Ollama ──────────────────────────────────────────────────────────────── */

static const char* ollama_chat(const char* url, const char* model,
                               cJSON* messages, cJSON* tools) {
    cJSON* req = cJSON_CreateObject();
    cJSON_AddStringToObject(req, "model", model);
    cJSON_AddItemReferenceToObject(req, "messages", messages);
    if (tools) cJSON_AddItemReferenceToObject(req, "tools", tools);
    cJSON_AddFalseToObject(req, "stream");

    char* body = cJSON_PrintUnformatted(req);
    cJSON_Delete(req);
    if (!body) return NULL;

    g_resp_len = 0;
    g_resp[0]  = '\0';

    CURL* ch = curl_easy_init();
    char endpoint[256];
    snprintf(endpoint, sizeof(endpoint), "%s/api/chat", url);
    struct curl_slist* hdrs = curl_slist_append(NULL, "content-type: application/json");

    curl_easy_setopt(ch, CURLOPT_URL,           endpoint);
    curl_easy_setopt(ch, CURLOPT_HTTPHEADER,    hdrs);
    curl_easy_setopt(ch, CURLOPT_POSTFIELDS,    body);
    curl_easy_setopt(ch, CURLOPT_WRITEFUNCTION, curl_cb);
    curl_easy_setopt(ch, CURLOPT_WRITEDATA,     NULL);
    curl_easy_setopt(ch, CURLOPT_TIMEOUT,       120L);

    curl_easy_perform(ch);
    curl_slist_free_all(hdrs);
    curl_easy_cleanup(ch);
    free(body);

    return g_resp_len > 0 ? g_resp : NULL;
}

/* ── Schema ──────────────────────────────────────────────────────────────── */

static void make_system_prompt(void) {
    snprintf(g_sys, sizeof(g_sys),
        "You are a data analyst. You have access to a SQLite database via the query_db tool.\n"
        "ALWAYS call query_db — never answer from memory. "
        "Start by exploring the schema if needed (e.g. SELECT name FROM sqlite_master WHERE type='table'), "
        "then run whatever queries the question requires. "
        "Follow the data: if one query raises more questions, run another. "
        "Deliver a clear, evidence-based answer once you have enough data.");
}

/* ── Tool definition ─────────────────────────────────────────────────────── */

static cJSON* make_tools(void) {
    cJSON* tools = cJSON_CreateArray();
    cJSON* t     = cJSON_CreateObject();
    cJSON* fn    = cJSON_CreateObject();
    cJSON* params= cJSON_CreateObject();
    cJSON* props = cJSON_CreateObject();
    cJSON* sql_p = cJSON_CreateObject();
    cJSON* req   = cJSON_CreateArray();

    cJSON_AddStringToObject(sql_p, "type",        "string");
    cJSON_AddStringToObject(sql_p, "description", "SQLite query with LIMIT 20");
    cJSON_AddItemToObject(props, "sql", sql_p);
    cJSON_AddStringToObject(params, "type", "object");
    cJSON_AddItemToObject(params, "properties", props);
    cJSON_AddItemToArray(req, cJSON_CreateString("sql"));
    cJSON_AddItemToObject(params, "required", req);

    cJSON_AddStringToObject(fn, "name", "query_db");
    cJSON_AddStringToObject(fn, "description",
        "Execute a SQLite query against the employee database. Always include LIMIT 20.");
    cJSON_AddItemToObject(fn, "parameters", params);

    cJSON_AddStringToObject(t, "type", "function");
    cJSON_AddItemToObject(t, "function", fn);
    cJSON_AddItemToArray(tools, t);
    return tools;
}

/* ── mpack: decode sql_result rows into g_rows_json ─────────────────────── */

static void decode_rows_to_json(mpack_reader_t* r) {
    size_t len = 0;
    size_t cap = sizeof(g_rows_json);

#define JA(s) do { \
    size_t _l = strlen(s); \
    if (len + _l + 1 < cap) { memcpy(g_rows_json + len, s, _l); len += _l; } \
} while(0)

    uint32_t nrows = mpack_expect_array_max(r, 100);
    if (len + 1 < cap) g_rows_json[len++] = '[';

    for (uint32_t row = 0; row < nrows && mpack_reader_error(r) == mpack_ok; row++) {
        if (row > 0 && len + 1 < cap) g_rows_json[len++] = ',';
        if (len + 1 < cap) g_rows_json[len++] = '{';

        uint32_t ncols = mpack_expect_map_max(r, 64);
        for (uint32_t col = 0; col < ncols && mpack_reader_error(r) == mpack_ok; col++) {
            if (col > 0 && len + 1 < cap) g_rows_json[len++] = ',';

            char colname[64];
            mpack_expect_cstr(r, colname, sizeof(colname));
            if (len + 1 < cap) g_rows_json[len++] = '"';
            JA(colname);
            if (len + 2 < cap) { g_rows_json[len++] = '"'; g_rows_json[len++] = ':'; }

            mpack_tag_t tag = mpack_peek_tag(r);
            char tmp[32];
            switch (mpack_tag_type(&tag)) {
                case mpack_type_int:
                    snprintf(tmp, sizeof(tmp), "%lld", (long long)mpack_expect_i64(r));
                    JA(tmp); break;
                case mpack_type_uint:
                    snprintf(tmp, sizeof(tmp), "%llu", (unsigned long long)mpack_expect_u64(r));
                    JA(tmp); break;
                case mpack_type_float:
                    snprintf(tmp, sizeof(tmp), "%g", (double)mpack_expect_float(r));
                    JA(tmp); break;
                case mpack_type_double:
                    snprintf(tmp, sizeof(tmp), "%g", mpack_expect_double(r));
                    JA(tmp); break;
                case mpack_type_str: {
                    char sv[512];
                    mpack_expect_cstr(r, sv, sizeof(sv));
                    if (len + 1 < cap) g_rows_json[len++] = '"';
                    for (const char* p = sv; *p && len + 3 < cap; p++) {
                        if (*p == '"' || *p == '\\') g_rows_json[len++] = '\\';
                        else if ((unsigned char)*p < 0x20) { g_rows_json[len++] = '?'; continue; }
                        g_rows_json[len++] = *p;
                    }
                    if (len + 1 < cap) g_rows_json[len++] = '"';
                    break;
                }
                default:
                    mpack_discard(r);
                    JA("null");
                    break;
            }
        }
        mpack_done_map(r);
        if (len + 1 < cap) g_rows_json[len++] = '}';
    }

    mpack_done_array(r);
    if (len + 1 < cap) g_rows_json[len++] = ']';
    g_rows_json[len < cap ? len : cap - 1] = '\0';

#undef JA
}

/* ── Output helpers ──────────────────────────────────────────────────────── */

static void emit_sql_query(const char* sql) {
    mpack_writer_t w;
    mpack_writer_init(&w, g_out, sizeof(g_out));
    mpack_start_map(&w, 2);
    mpack_write_cstr(&w, "type"); mpack_write_cstr(&w, "sql_query");
    mpack_write_cstr(&w, "sql");  mpack_write_cstr(&w, sql);
    mpack_finish_map(&w);
    size_t used = mpack_writer_buffer_used(&w);
    mpack_writer_destroy(&w);
    fputs("sql_query\n", stdout);
    fwrite(g_out, 1, used, stdout);
}

static void emit_answer(const char* answer) {
    mpack_writer_t w;
    mpack_writer_init(&w, g_out, sizeof(g_out));
    mpack_start_map(&w, 2);
    mpack_write_cstr(&w, "type");   mpack_write_cstr(&w, "agent_response_masked");
    mpack_write_cstr(&w, "answer"); mpack_write_cstr(&w, answer ? answer : "");
    mpack_finish_map(&w);
    size_t used = mpack_writer_buffer_used(&w);
    mpack_writer_destroy(&w);
    fputs("agent_response_masked\n", stdout);
    fwrite(g_out, 1, used, stdout);
}

/* ── Save/restore helpers ────────────────────────────────────────────────── */

/* Persist {history, [messages, round]} to LMDB. Caller owns all cJSON objects. */
static void save_state(MDB_env* env, const char* key,
                       cJSON* history, cJSON* messages, int round) {
    cJSON* st = cJSON_CreateObject();
    cJSON_AddItemReferenceToObject(st, "history",  history);
    if (messages) {
        cJSON_AddItemReferenceToObject(st, "messages", messages);
        cJSON_AddNumberToObject(st, "round", round);
    }
    char* s = cJSON_PrintUnformatted(st);
    if (s && env) state_save(env, key, s);
    free(s);
    cJSON_Delete(st);
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(void) {
    size_t n = fread(g_in, 1, sizeof(g_in) - 1, stdin);

    const char* corr_id = getenv("ACTOR_CORRELATION_ID");
    const char* ollama  = getenv("OLLAMA_URL");   if (!ollama)  ollama  = "http://localhost:11434";
    const char* model   = getenv("OLLAMA_MODEL"); if (!model)   model   = "gemma4:e2b";
    const char* db_path = getenv("EMPLOYEE_DB");  if (!db_path) db_path = "./employee.db";

    if (!corr_id || !*corr_id) {
        emit_answer("internal error: no correlation id");
        return 1;
    }

    MDB_env* env = lmdb_open();

    /* dedup by tuple id — MDB_NOOVERWRITE is atomic, works across instances */
    const char* tuple_id = getenv("ACTOR_TUPLE_ID");
    if (dedup_check(env, tuple_id)) {
        fprintf(stderr, "[llm-agent] dedup: skipping already-seen tuple %s\n", tuple_id);
        if (env) mdb_env_close(env);
        return 0;
    }

    /* decode mpack input */
    char in_type[32]  = {0};
    char in_error[256] = {0};
    g_query[0] = g_rows_json[0] = '\0';

    mpack_reader_t rdr;
    mpack_reader_init_data(&rdr, g_in, n);

    static const char* in_keys[] = { "type", "query", "rows", "error" };
    bool in_found[4] = { false, false, false, false };

    uint32_t in_sz = mpack_expect_map_max(&rdr, 8);
    for (uint32_t i = 0; i < in_sz && mpack_reader_error(&rdr) == mpack_ok; i++) {
        switch (mpack_expect_key_cstr(&rdr, in_keys, in_found, 4)) {
            case 0: mpack_expect_cstr(&rdr, in_type,   sizeof(in_type));   break;
            case 1: mpack_expect_cstr(&rdr, g_query,   sizeof(g_query));   break;
            case 2: decode_rows_to_json(&rdr); break;
            case 3: mpack_expect_cstr(&rdr, in_error,  sizeof(in_error));  break;
            default: mpack_discard(&rdr); break;
        }
    }
    mpack_done_map(&rdr);

    if (mpack_reader_error(&rdr) != mpack_ok || !in_type[0]) {
        mpack_reader_destroy(&rdr);
        emit_answer("internal error: bad mpack input");
        return 1;
    }
    mpack_reader_destroy(&rdr);

    cJSON*   history  = NULL;  /* cross-turn memory: [{role,content},...] */
    cJSON*   messages = NULL;  /* ollama messages for current ReAct turn  */
    int      round    = 0;

    /* load LMDB state keyed by correlation_id */
    const char* raw = env ? state_load(env, corr_id) : NULL;
    cJSON* stored   = raw ? cJSON_Parse(raw) : NULL;

    if (strcmp(in_type, "sql_result_masked") == 0) {
        /* ── Round 2+: continue ReAct loop ──────────────────────────────── */
        if (!stored) {
            fprintf(stderr, "[llm-agent] no state for corr=%s, discarding\n", corr_id);
            if (env) mdb_env_close(env);
            return 0;
        }

        round    = (int)cJSON_GetNumberValue(cJSON_GetObjectItem(stored, "round"));
        history  = cJSON_DetachItemFromObject(stored, "history");
        messages = cJSON_DetachItemFromObject(stored, "messages");
        cJSON_Delete(stored);

        if (!history)  history  = cJSON_CreateArray();
        if (!messages) { emit_answer("internal error: missing messages in state"); goto done; }

        /* find last tool_call_id */
        cJSON* last_tc_id = NULL;
        int mlen = cJSON_GetArraySize(messages);
        for (int i = mlen - 1; i >= 0; i--) {
            cJSON* m   = cJSON_GetArrayItem(messages, i);
            cJSON* tcs = cJSON_GetObjectItem(m, "tool_calls");
            if (tcs && cJSON_GetArraySize(tcs) > 0) {
                last_tc_id = cJSON_GetObjectItem(
                    cJSON_GetArrayItem(tcs, 0), "id");
                break;
            }
        }

        /* if sqlite-tool returned an error, feed that back so the model can retry */
        const char* rows_str = in_error[0]
            ? in_error
            : (g_rows_json[0] ? g_rows_json : "[]");
        cJSON* tool_msg = cJSON_CreateObject();
        cJSON_AddStringToObject(tool_msg, "role",    "tool");
        cJSON_AddStringToObject(tool_msg, "content", rows_str);
        if (last_tc_id)
            cJSON_AddStringToObject(tool_msg, "tool_call_id",
                                    cJSON_GetStringValue(last_tc_id));
        cJSON_AddItemToArray(messages, tool_msg);

    } else {
        /* ── Round 1: new user message ───────────────────────────────────── */
        make_system_prompt();

        /* load cross-turn history */
        if (stored) {
            history = cJSON_DetachItemFromObject(stored, "history");
            cJSON_Delete(stored);
        }
        if (!history) history = cJSON_CreateArray();

        /* append this user message to history */
        cJSON* user_entry = cJSON_CreateObject();
        cJSON_AddStringToObject(user_entry, "role",    "user");
        cJSON_AddStringToObject(user_entry, "content", g_query);
        cJSON_AddItemToArray(history, user_entry);

        /* build Ollama messages: system + full history */
        messages = cJSON_CreateArray();
        cJSON* sys_msg = cJSON_CreateObject();
        cJSON_AddStringToObject(sys_msg, "role",    "system");
        cJSON_AddStringToObject(sys_msg, "content", g_sys);
        cJSON_AddItemToArray(messages, sys_msg);

        int hlen = cJSON_GetArraySize(history);
        for (int i = 0; i < hlen; i++)
            cJSON_AddItemToArray(messages,
                cJSON_Duplicate(cJSON_GetArrayItem(history, i), 1));
    }

    if (round >= MAX_ROUNDS) {
        emit_answer("(max rounds reached without a final answer)");
        /* still save history as-is */
        save_state(env, corr_id, history, NULL, 0);
        goto done;
    }
    round++;

    /* keep tools available while we have successful results to build on;
     * on a sql error the model needs tools to retry with corrected SQL;
     * only force final answer (strip tools+system) after a successful result */
    /* keep tools available so the model can keep querying across rounds;
     * on the final round strip tools + system to force a text answer */
    cJSON* msgs_to_send = messages;
    cJSON* stripped     = NULL;
    cJSON* tools        = NULL;

    if (round < MAX_ROUNDS) {
        tools = make_tools();
    } else {
        stripped = cJSON_CreateArray();
        int mlen = cJSON_GetArraySize(messages);
        for (int i = 0; i < mlen; i++) {
            cJSON* m = cJSON_GetArrayItem(messages, i);
            const char* role = cJSON_GetStringValue(cJSON_GetObjectItem(m, "role"));
            if (role && strcmp(role, "system") != 0)
                cJSON_AddItemToArray(stripped, cJSON_Duplicate(m, 1));
        }
        msgs_to_send = stripped;
    }

    const char* resp_raw = ollama_chat(ollama, model, msgs_to_send, tools);
    if (stripped) cJSON_Delete(stripped);
    if (tools)    cJSON_Delete(tools);

    if (!resp_raw) {
        emit_answer("error: no response from ollama");
        save_state(env, corr_id, history, NULL, 0);
        goto done;
    }

    cJSON* resp = cJSON_Parse(resp_raw);
    if (!resp) {
        emit_answer("error: bad json from ollama");
        save_state(env, corr_id, history, NULL, 0);
        goto done;
    }

    {
        cJSON* msg        = cJSON_GetObjectItem(resp, "message");
        cJSON* tool_calls = cJSON_GetObjectItem(msg,  "tool_calls");
        const char* content = cJSON_GetStringValue(cJSON_GetObjectItem(msg, "content"));

        if (tool_calls && cJSON_GetArraySize(tool_calls) > 0) {
            /* save state: history + full react messages + round, emit sql */
            cJSON_AddItemToArray(messages, cJSON_Duplicate(msg, 1));

            cJSON* tc   = cJSON_GetArrayItem(tool_calls, 0);
            cJSON* args = cJSON_GetObjectItem(
                              cJSON_GetObjectItem(tc, "function"), "arguments");
            const char* sql = cJSON_GetStringValue(cJSON_GetObjectItem(args, "sql"));

            if (!sql || !*sql) {
                emit_answer("error: model returned empty sql");
                save_state(env, corr_id, history, NULL, 0);
            } else {
                save_state(env, corr_id, history, messages, round);
                emit_sql_query(sql);
            }
        } else {
            /* final answer: append to history and persist memory */
            cJSON* asst = cJSON_CreateObject();
            cJSON_AddStringToObject(asst, "role",    "assistant");
            cJSON_AddStringToObject(asst, "content", content ? content : "");
            cJSON_AddItemToArray(history, asst);

            save_state(env, corr_id, history, NULL, 0);
            emit_answer(content);
        }

        cJSON_Delete(resp);
    }

done:
    if (messages) cJSON_Delete(messages);
    if (history)  cJSON_Delete(history);
    if (env)      mdb_env_close(env);
    return 0;
}
