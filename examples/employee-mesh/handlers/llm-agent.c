#define _POSIX_C_SOURCE 200809L
/*
 * handlers/llm-agent.c — agentic ReAct loop handler
 *
 * Receives: mpack user_message  {type, query, session_id}
 *           mpack sql_result    {type, rows:[{col:val,...},...]}
 *
 * Emits:    sql_query\n    mpack {type:"sql_query",  sql:"..."}
 *        or agent_response\n mpack {type:"agent_response", answer, query, session_id}
 *
 * State lives in LMDB keyed by ACTOR_CORRELATION_ID.
 * Zero dynamic allocation — all buffers are static file-scope arrays.
 * mpack for mesh I/O; cJSON for Ollama HTTP (complex nested tool JSON).
 *
 * Env: OLLAMA_URL, OLLAMA_MODEL, EMPLOYEE_DB, ACTOR_LMDB_PATH, ACTOR_CORRELATION_ID
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
static char   g_query[4096];
static char   g_session[256];
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

/* returns pointer into g_state_buf, or NULL */
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

static void state_del(MDB_env* env, const char* key) {
    MDB_txn* txn;
    if (mdb_txn_begin(env, NULL, 0, &txn)) return;
    MDB_dbi dbi;
    if (mdb_dbi_open(txn, "state", 0, &dbi)) { mdb_txn_abort(txn); return; }
    MDB_val k = { strlen(key), (void*)key };
    mdb_del(txn, dbi, &k, NULL);
    mdb_txn_commit(txn);
}

/* ── Ollama: returns pointer to g_resp, or NULL ──────────────────────────── */

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

/* ── Schema: single DB open, results into static buffers ────────────────── */

static void query_join(sqlite3* db, const char* sql, char sep,
                       char* out, size_t cap) {
    size_t len = 0;
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            const char* v = (const char*)sqlite3_column_text(stmt, 0);
            if (!v) continue;
            size_t vlen = strlen(v);
            if (len + vlen + 3 >= cap) break;
            memcpy(out + len, v, vlen);
            len += vlen;
            out[len++] = sep;
            if (sep == ',') out[len++] = ' ';
        }
        sqlite3_finalize(stmt);
    }
    if (len > 0 && out[len-1] == ' ' && len > 1) len -= 2;
    else if (len > 0 && out[len-1] == sep) len--;
    out[len] = '\0';
}

static void db_meta(const char* db_path) {
    sqlite3* db;
    g_schema[0] = g_depts[0] = g_titles[0] = '\0';
    if (sqlite3_open_v2(db_path, &db, SQLITE_OPEN_READONLY, NULL) != SQLITE_OK) return;
    query_join(db,
        "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type,name",
        '\n', g_schema, sizeof(g_schema));
    query_join(db,
        "SELECT dept_name FROM departments ORDER BY dept_name",
        ',', g_depts, sizeof(g_depts));
    query_join(db,
        "SELECT DISTINCT title FROM titles ORDER BY title",
        ',', g_titles, sizeof(g_titles));
    sqlite3_close(db);
}

/* ── System prompt: writes into g_sys ───────────────────────────────────── */

static void make_system_prompt(const char* db_path) {
    db_meta(db_path);
    size_t len = 0;

#define APPEND(s) do { \
    size_t _l = strlen(s); \
    if (len + _l + 1 < sizeof(g_sys)) { memcpy(g_sys + len, s, _l); len += _l; } \
} while(0)

    APPEND("You are an HR assistant. You MUST call query_db for every question. Never answer from memory.\n\n");
    APPEND("Schema:\n"); APPEND(g_schema); APPEND("\n\n");
    APPEND("Exact department names: "); APPEND(g_depts); APPEND("\n");
    APPEND("Exact job titles: ");       APPEND(g_titles); APPEND("\n\n");
    APPEND("Rules:\n");
    APPEND("- dept_no is a short code (d001 etc). Always JOIN departments to filter by dept_name.\n");
    APPEND("- Always use LIKE for name matching (never exact =)\n");
    APPEND("- Use to_date='9999-01-01' to get current records only\n");
    APPEND("- Always LIMIT 20\n");
    APPEND("- There is NO 'Engineering' department — use 'Development'\n\n");
    APPEND("Examples:\n");
    APPEND("-- manager: SELECT e.first_name,e.last_name,d.dept_name FROM employees e JOIN dept_manager dm ON e.emp_no=dm.emp_no JOIN departments d ON dm.dept_no=d.dept_no WHERE d.dept_name LIKE '%Development%' AND dm.to_date='9999-01-01' LIMIT 20\n");
    APPEND("-- title:   SELECT e.first_name,e.last_name,t.title FROM employees e JOIN titles t ON e.emp_no=t.emp_no WHERE t.title LIKE '%Engineer%' AND t.to_date='9999-01-01' LIMIT 20\n");
    APPEND("-- salary:  SELECT e.first_name,e.last_name,s.salary FROM employees e JOIN salaries s ON e.emp_no=s.emp_no WHERE s.to_date='9999-01-01' ORDER BY s.salary DESC LIMIT 20\n");

#undef APPEND
    g_sys[len] = '\0';
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

/* ── mpack input: decode rows array into g_rows_json ────────────────────── */

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
                    JA(tmp);
                    break;
                case mpack_type_uint:
                    snprintf(tmp, sizeof(tmp), "%llu", (unsigned long long)mpack_expect_u64(r));
                    JA(tmp);
                    break;
                case mpack_type_float:
                    snprintf(tmp, sizeof(tmp), "%g", (double)mpack_expect_float(r));
                    JA(tmp);
                    break;
                case mpack_type_double:
                    snprintf(tmp, sizeof(tmp), "%g", mpack_expect_double(r));
                    JA(tmp);
                    break;
                case mpack_type_str: {
                    char sv[512];
                    mpack_expect_cstr(r, sv, sizeof(sv));
                    if (len + 1 < cap) g_rows_json[len++] = '"';
                    for (const char* p = sv; *p && len + 3 < cap; p++) {
                        if (*p == '"' || *p == '\\') g_rows_json[len++] = '\\';
                        else if ((unsigned char)*p < 0x20) { len++; g_rows_json[len-1] = '?'; continue; }
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

/* ── mpack emit helpers ──────────────────────────────────────────────────── */

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

static void emit_agent_response(const char* answer, const char* query,
                                const char* session_id) {
    mpack_writer_t w;
    mpack_writer_init(&w, g_out, sizeof(g_out));
    mpack_start_map(&w, 4);
    mpack_write_cstr(&w, "type");       mpack_write_cstr(&w, "agent_response");
    mpack_write_cstr(&w, "answer");     mpack_write_cstr(&w, answer     ? answer     : "");
    mpack_write_cstr(&w, "query");      mpack_write_cstr(&w, query      ? query      : "");
    mpack_write_cstr(&w, "session_id"); mpack_write_cstr(&w, session_id ? session_id : "");
    mpack_finish_map(&w);
    size_t used = mpack_writer_buffer_used(&w);
    mpack_writer_destroy(&w);
    fputs("agent_response\n", stdout);
    fwrite(g_out, 1, used, stdout);
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(void) {
    size_t n = fread(g_in, 1, sizeof(g_in) - 1, stdin);

    const char* corr_id = getenv("ACTOR_CORRELATION_ID");
    const char* ollama  = getenv("OLLAMA_URL");   if (!ollama)  ollama  = "http://localhost:11434";
    const char* model   = getenv("OLLAMA_MODEL"); if (!model)   model   = "gemma4:e2b";
    const char* db_path = getenv("EMPLOYEE_DB");  if (!db_path) db_path = "./employee.db";

    if (!corr_id || !*corr_id) {
        emit_agent_response("internal error: no correlation id", "", "");
        return 1;
    }

    /* decode mpack input */
    char in_type[32] = {0};
    g_query[0] = g_session[0] = g_rows_json[0] = '\0';

    mpack_reader_t rdr;
    mpack_reader_init_data(&rdr, g_in, n);

    static const char* in_keys[] = { "type", "query", "session_id", "rows" };
    bool in_found[4] = { false, false, false, false };

    uint32_t in_sz = mpack_expect_map_max(&rdr, 8);
    for (uint32_t i = 0; i < in_sz && mpack_reader_error(&rdr) == mpack_ok; i++) {
        switch (mpack_expect_key_cstr(&rdr, in_keys, in_found, 4)) {
            case 0: mpack_expect_cstr(&rdr, in_type,  sizeof(in_type));  break;
            case 1: mpack_expect_cstr(&rdr, g_query,  sizeof(g_query));  break;
            case 2: mpack_expect_cstr(&rdr, g_session, sizeof(g_session)); break;
            case 3: decode_rows_to_json(&rdr); break;
            default: mpack_discard(&rdr); break;
        }
    }
    mpack_done_map(&rdr);

    if (mpack_reader_error(&rdr) != mpack_ok || !in_type[0]) {
        mpack_reader_destroy(&rdr);
        emit_agent_response("internal error: bad mpack input", "", "");
        return 1;
    }
    mpack_reader_destroy(&rdr);

    MDB_env* env    = lmdb_open();
    cJSON*   messages = NULL;
    int      round    = 0;

    if (strcmp(in_type, "sql_result") == 0) {
        /* Round 2+: load conversation state from LMDB */
        const char* raw = env ? state_load(env, corr_id) : NULL;
        if (!raw) {
            fprintf(stderr, "[llm-agent] no state for corr=%s, discarding\n", corr_id);
            if (env) mdb_env_close(env);
            return 0;
        }

        cJSON* state = cJSON_Parse(raw);

        round = (int)cJSON_GetNumberValue(cJSON_GetObjectItem(state, "round"));
        const char* q   = cJSON_GetStringValue(cJSON_GetObjectItem(state, "query"));
        const char* sid = cJSON_GetStringValue(cJSON_GetObjectItem(state, "session_id"));
        snprintf(g_query,   sizeof(g_query),   "%s", q   ? q   : "");
        snprintf(g_session, sizeof(g_session), "%s", sid ? sid : "");
        messages = cJSON_DetachItemFromObject(state, "messages");
        cJSON_Delete(state);

        /* find last tool_call_id to correlate this result */
        cJSON* last_tc_id = NULL;
        int mlen = cJSON_GetArraySize(messages);
        for (int i = mlen - 1; i >= 0; i--) {
            cJSON* m   = cJSON_GetArrayItem(messages, i);
            cJSON* tcs = cJSON_GetObjectItem(m, "tool_calls");
            if (tcs && cJSON_GetArraySize(tcs) > 0) {
                last_tc_id = cJSON_GetObjectItem(cJSON_GetArrayItem(tcs, 0), "id");
                break;
            }
        }

        /* append tool result message using rows JSON decoded from mpack input */
        const char* rows_str = g_rows_json[0] ? g_rows_json : "[]";

        cJSON* tool_msg = cJSON_CreateObject();
        cJSON_AddStringToObject(tool_msg, "role",    "tool");
        cJSON_AddStringToObject(tool_msg, "content", rows_str);
        if (last_tc_id)
            cJSON_AddStringToObject(tool_msg, "tool_call_id",
                                    cJSON_GetStringValue(last_tc_id));
        cJSON_AddItemToArray(messages, tool_msg);

    } else {
        /* Round 1: fresh conversation from user_message (g_query/g_session set by mpack decode) */
        make_system_prompt(db_path);

        messages = cJSON_CreateArray();

        cJSON* sys_msg = cJSON_CreateObject();
        cJSON_AddStringToObject(sys_msg, "role",    "system");
        cJSON_AddStringToObject(sys_msg, "content", g_sys);
        cJSON_AddItemToArray(messages, sys_msg);

        cJSON* usr_msg = cJSON_CreateObject();
        cJSON_AddStringToObject(usr_msg, "role",    "user");
        cJSON_AddStringToObject(usr_msg, "content", g_query);
        cJSON_AddItemToArray(messages, usr_msg);
    }

    if (round >= MAX_ROUNDS) {
        emit_agent_response("(max rounds reached without a final answer)", g_query, g_session);
        if (messages) cJSON_Delete(messages);
        if (env) mdb_env_close(env);
        return 0;
    }
    round++;

    /* round 1: offer tools; round 2+: strip system msg + no tools → force final answer */
    cJSON* msgs_to_send = messages;
    cJSON* stripped     = NULL;
    cJSON* tools        = NULL;

    if (round == 1) {
        tools = make_tools();
    } else {
        stripped = cJSON_CreateArray();
        int mlen = cJSON_GetArraySize(messages);
        for (int i = 0; i < mlen; i++) {
            cJSON* m    = cJSON_GetArrayItem(messages, i);
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
        emit_agent_response("error: no response from ollama", g_query, g_session);
        cJSON_Delete(messages);
        if (env) mdb_env_close(env);
        return 1;
    }

    cJSON* resp = cJSON_Parse(resp_raw);
    if (!resp) {
        emit_agent_response("error: bad json from ollama", g_query, g_session);
        cJSON_Delete(messages);
        if (env) mdb_env_close(env);
        return 1;
    }

    cJSON* msg        = cJSON_GetObjectItem(resp, "message");
    cJSON* tool_calls = cJSON_GetObjectItem(msg,  "tool_calls");
    const char* content = cJSON_GetStringValue(cJSON_GetObjectItem(msg, "content"));

    if (tool_calls && cJSON_GetArraySize(tool_calls) > 0) {
        /* model wants to call a tool — save state and emit sql_query */
        cJSON_AddItemToArray(messages, cJSON_Duplicate(msg, 1));

        cJSON* tc  = cJSON_GetArrayItem(tool_calls, 0);
        cJSON* args = cJSON_GetObjectItem(cJSON_GetObjectItem(tc, "function"), "arguments");
        const char* sql = cJSON_GetStringValue(cJSON_GetObjectItem(args, "sql"));

        if (!sql || !*sql) {
            emit_agent_response("error: model returned empty sql", g_query, g_session);
        } else {
            cJSON* st = cJSON_CreateObject();
            cJSON_AddNumberToObject(st, "round",      round);
            cJSON_AddStringToObject(st, "query",      g_query);
            cJSON_AddStringToObject(st, "session_id", g_session);
            cJSON_AddItemReferenceToObject(st, "messages", messages);
            char* st_str = cJSON_PrintUnformatted(st);
            if (env) state_save(env, corr_id, st_str);
            free(st_str);
            cJSON_Delete(st);

            emit_sql_query(sql);
        }
    } else {
        /* final answer */
        if (env) state_del(env, corr_id);
        emit_agent_response(content, g_query, g_session);
    }

    cJSON_Delete(resp);
    cJSON_Delete(messages);
    if (env) mdb_env_close(env);
    return 0;
}
