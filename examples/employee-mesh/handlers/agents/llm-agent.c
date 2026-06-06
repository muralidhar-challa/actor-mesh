#define _POSIX_C_SOURCE 200809L
/*
 * handlers/llm-agent.c — capability-aware ReAct agent
 *
 * Receives: mpack user_message  {type, query}
 *           mpack sql_result    {type, rows:[{col:val,...},...]}
 *           mpack _tool_list     {type, capabilities:[{actor,capabilities:[...]}]}
 *
 * Emits:    <tool_topic>\n  mpack {type:"...", ...}
 *        or agent_response\n mpack {type:"agent_response_masked", answer:"..."}
 *
 * Capabilities: stored in LMDB under __capabilities key.
 * Tools and system prompt built dynamically from stored capabilities.
 * Falls back to hardcoded SQL tool if no capabilities stored (backward compat).
 *
 * Env: LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, EMPLOYEE_DB, ACTOR_LMDB_PATH,
 *      ACTOR_CORRELATION_ID
 * Deps: libcurl, liblmdb, libsqlite3, mpack, cJSON
 */

#include "cJSON.h"
#include "mpack.h"
#include <curl/curl.h>
#include <lmdb.h>
#include <sqlite3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_IN      (256 * 1024)
#define MAX_RESP    (2   * 1024 * 1024)
#define MAX_STATE   (512 * 1024)
#define MAX_SCHEMA  (64  * 1024)
#define MAX_META    (8   * 1024)
#define MAX_SYS     (128 * 1024)
#define MAX_ROWS_J  (128 * 1024)
#define MAX_CAPS    (64  * 1024)
#define MAX_OUT     (1   * 1024 * 1024)
#define MAX_ROUNDS  6

static char   g_in[MAX_IN];
static char   g_resp[MAX_RESP];
static size_t g_resp_len;
static char   g_state_buf[MAX_STATE];
static char   g_schema[MAX_SCHEMA];
static char   g_depts[MAX_META];
static char   g_titles[MAX_META];
static char   g_sys[MAX_SYS];
static char   g_rows_json[MAX_ROWS_J];
static char   g_query[MAX_IN];
static char   g_out[MAX_OUT];
static char   g_caps[MAX_CAPS];     /* stored capabilities JSON */
static char   g_tool_topic[64];     /* topic for current tool call */

/* ── Helpers ──────────────────────────────────────────────────────────────── */

static size_t resp_cb(void* data, size_t sz, size_t nmemb, void* ctx) {
    (void)ctx;
    size_t total = sz * nmemb;
    if (g_resp_len + total + 1 <= MAX_RESP) {
        memcpy(g_resp + g_resp_len, data, total);
        g_resp_len += total;
    }
    return total;
}

static MDB_env* lmdb_open(void) {
    const char* path = getenv("ACTOR_LMDB_PATH");
    if (!path) return NULL;
    MDB_env* env;
    if (mdb_env_create(&env)) return NULL;
    mdb_env_set_mapsize(env, 64UL * 1024 * 1024);
    mdb_env_set_maxdbs(env, 2);
    if (mdb_env_open(env, path, 0, 0664)) { mdb_env_close(env); return NULL; }
    return env;
}

static void state_save(MDB_env* env, const char* key, const char* val) {
    if (!env || !key || !val) return;
    MDB_txn* txn;
    MDB_dbi  dbi;
    mdb_txn_begin(env, NULL, 0, &txn);
    mdb_dbi_open(txn, "state", MDB_CREATE, &dbi);
    MDB_val k = { (int)strlen(key), (void*)key };
    MDB_val v = { (int)strlen(val), (void*)val };
    mdb_put(txn, dbi, &k, &v, 0);
    mdb_txn_commit(txn);
}

static char* state_load(MDB_env* env, const char* key) {
    if (!env || !key) return NULL;
    MDB_txn* txn;
    MDB_dbi  dbi;
    if (mdb_txn_begin(env, NULL, MDB_RDONLY, &txn)) return NULL;
    mdb_dbi_open(txn, "state", 0, &dbi);
    MDB_val k = { (int)strlen(key), (void*)key };
    MDB_val v = {0, NULL};
    if (mdb_get(txn, dbi, &k, &v)) { mdb_txn_abort(txn); return NULL; }
    char* s = malloc(v.mv_size + 1);
    if (s) { memcpy(s, v.mv_data, v.mv_size); s[v.mv_size] = '\0'; }
    mdb_txn_abort(txn);
    return s;
}

static int dedup_check(MDB_env* env, const char* tuple_id) {
    if (!env || !tuple_id) return 0;
    MDB_txn* txn;
    MDB_dbi  dbi;
    if (mdb_txn_begin(env, NULL, 0, &txn)) return 0;
    mdb_dbi_open(txn, "dedup", MDB_CREATE, &dbi);
    MDB_val k = { 32, (void*)tuple_id };
    MDB_val v = { 0, NULL };
    int rc = mdb_put(txn, dbi, &k, &v, MDB_NOOVERWRITE);
    mdb_txn_commit(txn);
    return rc == MDB_KEYEXIST;
}

/* ── Forward decls ───────────────────────────────────────────────────────── */

static void make_system_prompt(const char* db_path);
static cJSON* make_tools(void);
static void emit_sql_query(const char* sql);
static void emit_answer(const char* answer);
static void decode_rows_to_json(mpack_reader_t* r);
static void save_state(MDB_env* env, const char* key, cJSON* history, cJSON* messages, int round);
static const char* openai_chat(const char* base_url, const char* api_key, const char* model, cJSON* messages, cJSON* tools);
static void db_meta(const char* db_path);

/* ── DB Meta ─────────────────────────────────────────────────────────────── */

static void db_meta(const char* db_path) {
    sqlite3* db;
    if (sqlite3_open(db_path, &db)) return;

    g_schema[0] = g_depts[0] = g_titles[0] = '\0';

    sqlite3_stmt* stmt;
    if (!sqlite3_prepare_v2(db, "SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' AND sql NOT NULL ORDER BY name", -1, &stmt, NULL)) {
        size_t len = 0;
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            const char* s = (const char*)sqlite3_column_text(stmt, 0);
            if (s) {
                size_t sl = strlen(s);
                if (len + sl + 2 < MAX_SCHEMA) {
                    memcpy(g_schema + len, s, sl); len += sl;
                    g_schema[len++] = '\n';
                }
            }
        }
        g_schema[len] = '\0';
        sqlite3_finalize(stmt);
    }

    if (!sqlite3_prepare_v2(db, "SELECT DISTINCT dept_name FROM departments ORDER BY dept_name LIMIT 50", -1, &stmt, NULL)) {
        size_t len = 0;
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            const char* s = (const char*)sqlite3_column_text(stmt, 0);
            if (s) {
                size_t sl = strlen(s);
                if (len + sl + 3 < MAX_META) {
                    memcpy(g_depts + len, s, sl); len += sl;
                    g_depts[len++] = ','; g_depts[len++] = ' ';
                }
            }
        }
        if (len > 1) g_depts[len - 2] = '\0';
        sqlite3_finalize(stmt);
    }

    if (!sqlite3_prepare_v2(db, "SELECT DISTINCT title FROM titles ORDER BY title LIMIT 50", -1, &stmt, NULL)) {
        size_t len = 0;
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            const char* s = (const char*)sqlite3_column_text(stmt, 0);
            if (s) {
                size_t sl = strlen(s);
                if (len + sl + 3 < MAX_META) {
                    memcpy(g_titles + len, s, sl); len += sl;
                    g_titles[len++] = ','; g_titles[len++] = ' ';
                }
            }
        }
        if (len > 1) g_titles[len - 2] = '\0';
        sqlite3_finalize(stmt);
    }

    sqlite3_close(db);
}

/* ── System Prompt ───────────────────────────────────────────────────────── */

static void make_system_prompt(const char* db_path) {
    size_t len = 0;

#define APPEND(s) do { \
    size_t _l = strlen(s); \
    if (len + _l + 1 < sizeof(g_sys)) { memcpy(g_sys + len, s, _l); len += _l; } \
} while(0)

    /* Check for capability-driven prompt */
    if (g_caps[0]) {
        APPEND("You are a helpful AI assistant with access to tools.\n\n");
        APPEND("Available tools:\n");
        cJSON* caps = cJSON_Parse(g_caps);
        if (caps && cJSON_IsArray(caps)) {
            int nc = cJSON_GetArraySize(caps);
            for (int i = 0; i < nc; i++) {
                cJSON* actor = cJSON_GetArrayItem(caps, i);
                cJSON* name  = cJSON_GetObjectItem(actor, "actor");
                cJSON* tools = cJSON_GetObjectItem(actor, "capabilities");
                if (cJSON_IsString(name) && cJSON_IsArray(tools)) {
                    char buf[256];
                    snprintf(buf, sizeof(buf), "- %s: ", cJSON_GetStringValue(name));
                    APPEND(buf);
                    int nt = cJSON_GetArraySize(tools);
                    for (int j = 0; j < nt; j++) {
                        cJSON* tool = cJSON_GetArrayItem(tools, j);
                        cJSON* tn = cJSON_GetObjectItem(tool, "name");
                        cJSON* td = cJSON_GetObjectItem(tool, "description");
                        if (cJSON_IsString(tn)) {
                            APPEND(cJSON_GetStringValue(tn));
                            if (cJSON_IsString(td)) { APPEND(" ("); APPEND(cJSON_GetStringValue(td)); APPEND(")"); }
                            if (j < nt - 1) APPEND(", ");
                        }
                    }
                    APPEND("\n");
                }
            }
        }
        if (caps) cJSON_Delete(caps);
        APPEND("\nUse tools when appropriate. Call tools by their exact name.\n");
        g_sys[len] = '\0';
        return;
    }

    /* ── Default: no tools registered ── */
    APPEND("You are a helpful assistant. No tools are currently available.\n");

#undef APPEND
    g_sys[len] = '\0';
}

/* ── Tool definition ─────────────────────────────────────────────────────── */

static cJSON* make_tools(void) {
    /* Check for capability-driven tools */
    if (g_caps[0]) {
        cJSON* caps = cJSON_Parse(g_caps);
        cJSON* tools_arr = cJSON_CreateArray();
        if (caps && cJSON_IsArray(caps)) {
            int nc = cJSON_GetArraySize(caps);
            for (int i = 0; i < nc; i++) {
                cJSON* actor = cJSON_GetArrayItem(caps, i);
                cJSON* actor_tools = cJSON_GetObjectItem(actor, "capabilities");
                if (cJSON_IsArray(actor_tools)) {
                    int nt = cJSON_GetArraySize(actor_tools);
                    for (int j = 0; j < nt; j++) {
                        cJSON* tool  = cJSON_GetArrayItem(actor_tools, j);
                        cJSON* tname = cJSON_GetObjectItem(tool, "name");
                        cJSON* tdesc = cJSON_GetObjectItem(tool, "description");
                        cJSON* tsch  = cJSON_GetObjectItem(tool, "inputSchema");

                        cJSON* t = cJSON_CreateObject();
                        cJSON* fn = cJSON_CreateObject();
                        cJSON_AddStringToObject(fn, "name", cJSON_GetStringValue(tname));
                        if (cJSON_IsString(tdesc))
                            cJSON_AddStringToObject(fn, "description", cJSON_GetStringValue(tdesc));
                        if (tsch) cJSON_AddItemToObject(fn, "parameters", cJSON_Duplicate(tsch, 1));
                        cJSON_AddStringToObject(t, "type", "function");
                        cJSON_AddItemToObject(t, "function", fn);
                        cJSON_AddItemToArray(tools_arr, t);
                    }
                }
            }
        }
        if (caps) cJSON_Delete(caps);
        if (cJSON_GetArraySize(tools_arr) > 0) return tools_arr;
        cJSON_Delete(tools_arr);
    }

    /* ── Default: no tools registered ── */
    return cJSON_CreateArray();
}

/* ── Lookup tool topic from capabilities ─────────────────────────────────── */

static const char* find_tool_topic(const char* tool_name) {
    if (!g_caps[0]) return "sql_query"; /* fallback */
    g_tool_topic[0] = '\0';

    cJSON* caps = cJSON_Parse(g_caps);
    if (!caps) return "sql_query";

    /* search each actor's capabilities for matching tool name */
    int nc = cJSON_GetArraySize(caps);
    for (int i = 0; i < nc; i++) {
        cJSON* actor = cJSON_GetArrayItem(caps, i);
        cJSON* tools = cJSON_GetObjectItem(actor, "capabilities");
        if (!cJSON_IsArray(tools)) continue;
        int nt = cJSON_GetArraySize(tools);
        for (int j = 0; j < nt; j++) {
            cJSON* tool = cJSON_GetArrayItem(tools, j);
            cJSON* tn   = cJSON_GetObjectItem(tool, "name");
            if (cJSON_IsString(tn) && strcmp(cJSON_GetStringValue(tn), tool_name) == 0) {
                /* tool found — get actor name as topic */
                cJSON* an = cJSON_GetObjectItem(actor, "actor");
                if (cJSON_IsString(an)) {
                    snprintf(g_tool_topic, sizeof(g_tool_topic), "%s", cJSON_GetStringValue(an));
                }
                cJSON_Delete(caps);
                return g_tool_topic[0] ? g_tool_topic : "sql_query";
            }
        }
    }
    cJSON_Delete(caps);
    return "sql_query";
}

/* ── mpack: decode sql_result rows into g_rows_json ─────────────────────── */


/* ── mpack: decode sql_result rows into g_rows_json ─────────────────────── */

static void decode_rows_to_json(mpack_reader_t* r) {
    size_t len = 0;
#define JA(s) do { \
    size_t _l = strlen(s); \
    if (len + _l + 1 < sizeof(g_rows_json)) { memcpy(g_rows_json + len, s, _l); len += _l; } \
} while(0)

    uint32_t nrows = mpack_expect_array_max(r, 100);
    JA("[");
    for (uint32_t row = 0; row < nrows && mpack_reader_error(r) == mpack_ok; row++) {
        if (row > 0) JA(","); JA("{");
        uint32_t ncols = mpack_expect_map_max(r, 64);
        for (uint32_t col = 0; col < ncols && mpack_reader_error(r) == mpack_ok; col++) {
            if (col > 0) JA(",");
            char colname[64]; mpack_expect_cstr(r, colname, sizeof(colname));
            JA("\""); JA(colname); JA("\":");
            mpack_tag_t tag = mpack_read_tag(r);
            switch (mpack_tag_type(&tag)) {
            case mpack_type_str: {
                char sv[256]; uint32_t sl = tag.v.l; if (sl >= sizeof(sv)) sl = sizeof(sv)-1;
                mpack_read_bytes(r, sv, sl); sv[sl] = 0; mpack_done_str(r);
                JA("\""); JA(sv); JA("\""); break; }
            case mpack_type_int:
            case mpack_type_uint: {
                char tmp[32]; snprintf(tmp, sizeof(tmp), "%lld", (long long)tag.v.i);
                JA(tmp); break; }
            case mpack_type_float: case mpack_type_double: {
                char tmp[32]; snprintf(tmp, sizeof(tmp), "%g", tag.v.f); JA(tmp); break; }
            default: JA("null"); mpack_discard(r); break;
            }
        }
        mpack_done_map(r); JA("}");
    }
    mpack_done_array(r); JA("]");
    g_rows_json[len] = '\0';
#undef JA
}

/* ── Output helpers ──────────────────────────────────────────────────────── */

static void emit_sql_query(const char* sql) {
    const char* topic = find_tool_topic("query_db");
    /* Add _masked suffix if PII mode is enabled */
    static char topic_buf[80];
    const char* pii = getenv("PII_ENABLED");
    if (pii && strcmp(pii, "1") == 0) {
        snprintf(topic_buf, sizeof(topic_buf), "%s_masked", topic);
        topic = topic_buf;
    }
    mpack_writer_t w;
    mpack_writer_init(&w, g_out, sizeof(g_out));
    mpack_start_map(&w, 2);
    mpack_write_cstr(&w, "type"); mpack_write_cstr(&w, "tool_call");
    mpack_write_cstr(&w, "sql");  mpack_write_cstr(&w, sql);
    mpack_finish_map(&w);
    size_t used = mpack_writer_buffer_used(&w);
    mpack_writer_destroy(&w);
    fputs(topic, stdout); fputc('\n', stdout);
    fwrite(g_out, 1, used, stdout);
}

static void emit_answer(const char* answer) {
    const char* pii = getenv("PII_ENABLED");
    const char* topic = (pii && strcmp(pii, "1") == 0) ? "agent_response_masked" : "agent_response";
    mpack_writer_t w;
    mpack_writer_init(&w, g_out, sizeof(g_out));
    mpack_start_map(&w, 2);
    mpack_write_cstr(&w, "type");   mpack_write_cstr(&w, topic);
    mpack_write_cstr(&w, "answer"); mpack_write_cstr(&w, answer ? answer : "");
    mpack_finish_map(&w);
    size_t used = mpack_writer_buffer_used(&w);
    mpack_writer_destroy(&w);
    fputs(topic, stdout); fputc('\n', stdout);
    fwrite(g_out, 1, used, stdout);
}

/* ── Save/restore ────────────────────────────────────────────────────────── */

static void save_state(MDB_env* env, const char* key, cJSON* history, cJSON* messages, int round) {
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

/* ── OpenAI Chat ─────────────────────────────────────────────────────────── */

static const char* openai_chat(const char* base_url, const char* api_key,
                                const char* model, cJSON* messages, cJSON* tools) {
    char endpoint[512];
    snprintf(endpoint, sizeof(endpoint), "%s/v1/chat/completions", base_url);

    cJSON* body = cJSON_CreateObject();
    cJSON_AddStringToObject(body, "model", model);
    cJSON_AddItemToObject(body, "messages", cJSON_Duplicate(messages, 1));
    if (tools && cJSON_GetArraySize(tools) > 0)
        cJSON_AddItemToObject(body, "tools", cJSON_Duplicate(tools, 1));
    if (tools) cJSON_AddStringToObject(body, "tool_choice", "auto");

    char* post = cJSON_PrintUnformatted(body);
    cJSON_Delete(body);

    CURL* curl = curl_easy_init();
    if (!curl) { free(post); return NULL; }

    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    char auth[256];
    snprintf(auth, sizeof(auth), "Authorization: Bearer %s", api_key);
    headers = curl_slist_append(headers, auth);

    g_resp_len = 0; g_resp[0] = 0;
    curl_easy_setopt(curl, CURLOPT_URL, endpoint);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, post);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, resp_cb);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 120L);

    CURLcode rc = curl_easy_perform(curl);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    free(post);

    if (rc != CURLE_OK) return NULL;
    g_resp[g_resp_len] = '\0';
    return g_resp;
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(void) {
    size_t n = fread(g_in, 1, sizeof(g_in) - 1, stdin);

    const char* corr_id  = getenv("ACTOR_CORRELATION_ID");
    const char* base_url = getenv("LLM_BASE_URL"); if (!base_url) base_url = "https://api.deepseek.com";
    const char* api_key  = getenv("LLM_API_KEY");
    const char* model    = getenv("LLM_MODEL");    if (!model)    model    = "deepseek-chat";
    const char* db_path  = getenv("EMPLOYEE_DB");  if (!db_path)  db_path  = "./employee.db";

    if (!corr_id || !*corr_id) { emit_answer("internal error: no correlation id"); return 1; }

    MDB_env* env = lmdb_open();

    /* Load stored capabilities from LMDB */
    { char* caps_raw = state_load(env, "__capabilities");
      if (caps_raw) { snprintf(g_caps, sizeof(g_caps), "%s", caps_raw); free(caps_raw); }
      else g_caps[0] = '\0'; }

    const char* tuple_id = getenv("ACTOR_TUPLE_ID");
    if (dedup_check(env, tuple_id)) {
        fprintf(stderr, "[llm-agent] dedup: skipping already-seen tuple %s\n", tuple_id);
        if (env) mdb_env_close(env); return 0;
    }

    /* decode mpack input */
    char in_type[32] = {0};
    g_query[0] = g_rows_json[0] = '\0';
    mpack_reader_t rdr;
    mpack_reader_init_data(&rdr, g_in, n);
    static const char* in_keys[] = { "type", "query", "rows" };
    bool in_found[3] = { false, false, false };
    uint32_t in_sz = mpack_expect_map_max(&rdr, 8);
    for (uint32_t i = 0; i < in_sz && mpack_reader_error(&rdr) == mpack_ok; i++) {
        switch (mpack_expect_key_cstr(&rdr, in_keys, in_found, 3)) {
            case 0: mpack_expect_cstr(&rdr, in_type, sizeof(in_type)); break;
            case 1: mpack_expect_cstr(&rdr, g_query, sizeof(g_query)); break;
            case 2: decode_rows_to_json(&rdr); break;
            default: mpack_discard(&rdr); break;
        }
    }
    mpack_done_map(&rdr);
    if (mpack_reader_error(&rdr) != mpack_ok || !in_type[0]) {
        mpack_reader_destroy(&rdr); emit_answer("internal error: bad mpack input"); if (env) mdb_env_close(env); return 1;
    }
    mpack_reader_destroy(&rdr);

    /* Handle _tool_list: store capabilities in LMDB */
    if (strcmp(in_type, "_tool_list") == 0) {
        if (g_rows_json[0] && env) { state_save(env, "__capabilities", g_rows_json); }
        fprintf(stderr, "[llm-agent] capabilities updated\n");
        if (env) mdb_env_close(env); return 0;
    }

    cJSON* history  = NULL;
    cJSON* messages = NULL;
    int    round    = 0;

    const char* raw = env ? state_load(env, corr_id) : NULL;
    cJSON* stored   = raw ? cJSON_Parse(raw) : NULL;

    if (strcmp(in_type, "sql_result_masked") == 0 || strcmp(in_type, "sql_result") == 0) {
        if (!stored) { if (env) mdb_env_close(env); return 0; }
        round    = (int)cJSON_GetNumberValue(cJSON_GetObjectItem(stored, "round"));
        history  = cJSON_DetachItemFromObject(stored, "history");
        messages = cJSON_DetachItemFromObject(stored, "messages");
        cJSON_Delete(stored);
        if (!history) history = cJSON_CreateArray();
        if (!messages) { emit_answer("error: missing messages state"); goto done; }

        cJSON* last_tc_id = NULL;
        int mlen = cJSON_GetArraySize(messages);
        for (int i = mlen - 1; i >= 0; i--) {
            cJSON* m = cJSON_GetArrayItem(messages, i);
            cJSON* tcs = cJSON_GetObjectItem(m, "tool_calls");
            if (tcs && cJSON_GetArraySize(tcs) > 0) {
                last_tc_id = cJSON_GetObjectItem(cJSON_GetArrayItem(tcs, 0), "id"); break;
            }
        }
        cJSON* tool_msg = cJSON_CreateObject();
        cJSON_AddStringToObject(tool_msg, "role", "tool");
        cJSON_AddStringToObject(tool_msg, "content", g_rows_json[0] ? g_rows_json : "[]");
        if (last_tc_id) cJSON_AddStringToObject(tool_msg, "tool_call_id", cJSON_GetStringValue(last_tc_id));
        cJSON_AddItemToArray(messages, tool_msg);
    } else {
        make_system_prompt(db_path);
        if (stored) { history = cJSON_DetachItemFromObject(stored, "history"); cJSON_Delete(stored); }
        if (!history) history = cJSON_CreateArray();
        cJSON* user_entry = cJSON_CreateObject();
        cJSON_AddStringToObject(user_entry, "role", "user");
        cJSON_AddStringToObject(user_entry, "content", g_query);
        cJSON_AddItemToArray(history, user_entry);
        messages = cJSON_CreateArray();
        cJSON* sys_msg = cJSON_CreateObject();
        cJSON_AddStringToObject(sys_msg, "role", "system");
        cJSON_AddStringToObject(sys_msg, "content", g_sys);
        cJSON_AddItemToArray(messages, sys_msg);
        int hlen = cJSON_GetArraySize(history);
        for (int i = 0; i < hlen; i++)
            cJSON_AddItemToArray(messages, cJSON_Duplicate(cJSON_GetArrayItem(history, i), 1));
    }

    if (round >= MAX_ROUNDS) { emit_answer("(max rounds)"); save_state(env, corr_id, history, NULL, 0); goto done; }
    round++;

    cJSON* msgs_to_send = messages;
    cJSON* stripped = NULL;
    cJSON* tools = NULL;
    if (round == 1) { tools = make_tools(); }
    else { stripped = cJSON_CreateArray(); int mlen = cJSON_GetArraySize(messages);
        for (int i = 0; i < mlen; i++) { cJSON* m = cJSON_GetArrayItem(messages, i);
            const char* role = cJSON_GetStringValue(cJSON_GetObjectItem(m, "role"));
            if (role && strcmp(role, "system") != 0) cJSON_AddItemToArray(stripped, cJSON_Duplicate(m, 1)); }
        msgs_to_send = stripped; }

    const char* resp_raw = openai_chat(base_url, api_key, model, msgs_to_send, tools);
    if (stripped) cJSON_Delete(stripped);
    if (tools) cJSON_Delete(tools);
    if (!resp_raw) { emit_answer("error: no response from LLM"); save_state(env, corr_id, history, NULL, 0); goto done; }

    cJSON* resp = cJSON_Parse(resp_raw);
    if (!resp) { emit_answer("error: bad JSON from LLM"); save_state(env, corr_id, history, NULL, 0); goto done; }

    cJSON* choices    = cJSON_GetObjectItem(resp, "choices");
    cJSON* choice0    = cJSON_GetArrayItem(choices, 0);
    cJSON* msg        = cJSON_GetObjectItem(choice0, "message");
    cJSON* tool_calls = cJSON_GetObjectItem(msg, "tool_calls");
    const char* content = cJSON_GetStringValue(cJSON_GetObjectItem(msg, "content"));

    if (tool_calls && cJSON_GetArraySize(tool_calls) > 0) {
        cJSON_AddItemToArray(messages, cJSON_Duplicate(msg, 1));
        cJSON* tc   = cJSON_GetArrayItem(tool_calls, 0);
        cJSON* fn   = cJSON_GetObjectItem(tc, "function");
        const char* fn_name = cJSON_GetStringValue(cJSON_GetObjectItem(fn, "name"));
        const char* args_str = cJSON_GetStringValue(cJSON_GetObjectItem(fn, "arguments"));
        cJSON* args = args_str ? cJSON_Parse(args_str) : NULL;
        const char* sql = args ? cJSON_GetStringValue(cJSON_GetObjectItem(args, "sql")) : NULL;

        if (fn_name && sql && *sql) {
            /* Store tool name for topic routing */
            const char* topic = find_tool_topic(fn_name);
            save_state(env, corr_id, history, messages, round);
            emit_sql_query(sql);
        } else {
            emit_answer("error: LLM called tool without sql argument");
            save_state(env, corr_id, history, NULL, 0);
        }
        if (args) cJSON_Delete(args);
    } else {
        cJSON* asst = cJSON_CreateObject();
        cJSON_AddStringToObject(asst, "role", "assistant");
        cJSON_AddStringToObject(asst, "content", content ? content : "");
        cJSON_AddItemToArray(history, asst);
        save_state(env, corr_id, history, NULL, 0);
        emit_answer(content);
    }
    cJSON_Delete(resp);

done:
    if (messages) cJSON_Delete(messages);
    if (history) cJSON_Delete(history);
    if (env) mdb_env_close(env);
    return 0;
}
