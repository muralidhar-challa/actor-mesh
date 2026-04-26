#define _POSIX_C_SOURCE 200809L
/*
 * handlers/llm-agent.c — agentic ReAct loop handler
 *
 * Receives: user_message  { "type": "user_message", "query": "..." }
 *           sql_result    { "type": "sql_result",   "rows": [...] }
 *
 * Emits:    sql_query     { "type": "sql_query", "sql": "..." }
 *        or agent_response{ "type": "agent_response", "answer": "...", "query": "..." }
 *
 * Conversation state lives in LMDB keyed by ACTOR_CORRELATION_ID.
 * The actor runtime carries correlation_id through the full chain automatically.
 *
 * Env: OLLAMA_URL, OLLAMA_MODEL, EMPLOYEE_DB, ACTOR_LMDB_PATH, ACTOR_CORRELATION_ID
 * Deps: libcurl, liblmdb, cJSON
 */

#include "../vendor/cjson/cJSON.h"
#include <curl/curl.h>
#include <lmdb.h>
#include <sqlite3.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_IN       (256 * 1024)
#define MAX_RESP     (2  * 1024 * 1024)
#define MAX_STATE    (512 * 1024)
#define MAX_ROUNDS   6

/* ── Dynamic buffer for curl response ───────────────────────────────────── */

typedef struct { char* buf; size_t len; size_t cap; } Buf;

static size_t curl_cb(void* data, size_t sz, size_t nmemb, void* userp) {
    Buf* b = userp;
    size_t add = sz * nmemb;
    if (b->len + add + 1 > b->cap) {
        b->cap = b->len + add + 1 + 65536;
        b->buf = realloc(b->buf, b->cap);
        if (!b->buf) return 0;
    }
    memcpy(b->buf + b->len, data, add);
    b->len += add;
    b->buf[b->len] = '\0';
    return add;
}

/* ── LMDB state helpers ──────────────────────────────────────────────────── */

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

static char* state_load(MDB_env* env, const char* key) {
    /* named DBs must be opened from a write txn in a fresh MDB_env before
     * they can be accessed — use write txn, abort it to avoid any mutation */
    MDB_txn* txn;
    if (mdb_txn_begin(env, NULL, 0, &txn)) return NULL;
    MDB_dbi dbi;
    if (mdb_dbi_open(txn, "state", 0, &dbi)) { mdb_txn_abort(txn); return NULL; }
    MDB_val k = { strlen(key), (void*)key };
    MDB_val v;
    char* out = NULL;
    if (mdb_get(txn, dbi, &k, &v) == 0) {
        out = malloc(v.mv_size + 1);
        memcpy(out, v.mv_data, v.mv_size);
        out[v.mv_size] = '\0';
    }
    mdb_txn_abort(txn);
    return out;
}

static void state_save(MDB_env* env, const char* key, const char* val) {
    MDB_txn* txn;
    if (mdb_txn_begin(env, NULL, 0, &txn)) return;
    MDB_dbi dbi;
    if (mdb_dbi_open(txn, "state", MDB_CREATE, &dbi)) { mdb_txn_abort(txn); return; }
    MDB_val k = { strlen(key),  (void*)key };
    MDB_val v = { strlen(val),  (void*)val };
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

/* ── Ollama call ─────────────────────────────────────────────────────────── */

static char* ollama_chat(const char* url, const char* model,
                         cJSON* messages, cJSON* tools) {
    cJSON* req = cJSON_CreateObject();
    cJSON_AddStringToObject(req, "model", model);
    cJSON_AddItemReferenceToObject(req, "messages", messages);
    if (tools) cJSON_AddItemReferenceToObject(req, "tools", tools);
    cJSON_AddFalseToObject(req, "stream");

    char* body = cJSON_PrintUnformatted(req);
    cJSON_Delete(req);

    Buf resp = { malloc(65536), 0, 65536 };

    CURL* ch = curl_easy_init();
    char endpoint[256];
    snprintf(endpoint, sizeof(endpoint), "%s/api/chat", url);
    struct curl_slist* hdrs = curl_slist_append(NULL, "content-type: application/json");

    curl_easy_setopt(ch, CURLOPT_URL,            endpoint);
    curl_easy_setopt(ch, CURLOPT_HTTPHEADER,     hdrs);
    curl_easy_setopt(ch, CURLOPT_POSTFIELDS,     body);
    curl_easy_setopt(ch, CURLOPT_WRITEFUNCTION,  curl_cb);
    curl_easy_setopt(ch, CURLOPT_WRITEDATA,      &resp);
    curl_easy_setopt(ch, CURLOPT_TIMEOUT,        120L);

    curl_easy_perform(ch);
    curl_slist_free_all(hdrs);
    curl_easy_cleanup(ch);
    free(body);

    return resp.buf; /* caller frees */
}

/* ── Schema introspection (read-only, metadata only) ────────────────────── */

static char* db_schema(const char* db_path) {
    sqlite3* db;
    if (sqlite3_open_v2(db_path, &db, SQLITE_OPEN_READONLY, NULL) != SQLITE_OK)
        return strdup("");

    Buf out = { malloc(16384), 0, 16384 };

    sqlite3_stmt* stmt;
    const char* q = "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type,name";
    if (sqlite3_prepare_v2(db, q, -1, &stmt, NULL) == SQLITE_OK) {
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            const char* s = (const char*)sqlite3_column_text(stmt, 0);
            size_t slen = strlen(s);
            if (out.len + slen + 2 > out.cap) {
                out.cap = out.len + slen + 2 + 4096;
                out.buf = realloc(out.buf, out.cap);
            }
            memcpy(out.buf + out.len, s, slen);
            out.len += slen;
            out.buf[out.len++] = '\n';
        }
        sqlite3_finalize(stmt);
    }
    out.buf[out.len] = '\0';
    sqlite3_close(db);
    return out.buf;
}

static char* db_values(const char* db_path, const char* sql) {
    sqlite3* db;
    if (sqlite3_open_v2(db_path, &db, SQLITE_OPEN_READONLY, NULL) != SQLITE_OK)
        return strdup("");
    Buf out = { malloc(4096), 0, 4096 };
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            const char* v = (const char*)sqlite3_column_text(stmt, 0);
            if (!v) continue;
            size_t vlen = strlen(v);
            if (out.len + vlen + 3 > out.cap) {
                out.cap += vlen + 4096;
                out.buf = realloc(out.buf, out.cap);
            }
            memcpy(out.buf + out.len, v, vlen);
            out.len += vlen;
            out.buf[out.len++] = ',';
            out.buf[out.len++] = ' ';
        }
        sqlite3_finalize(stmt);
    }
    if (out.len >= 2) out.len -= 2; /* strip trailing ", " */
    out.buf[out.len] = '\0';
    sqlite3_close(db);
    return out.buf;
}

/* ── Tool definition ─────────────────────────────────────────────────────── */

static cJSON* make_tools(void) {
    /* [{"type":"function","function":{"name":"query_db",...}}] */
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

    cJSON_AddStringToObject(fn, "name",        "query_db");
    cJSON_AddStringToObject(fn, "description",
        "Execute a SQLite query against the employee database. Always include LIMIT 20.");
    cJSON_AddItemToObject(fn, "parameters", params);

    cJSON_AddStringToObject(t, "type", "function");
    cJSON_AddItemToObject(t, "function", fn);
    cJSON_AddItemToArray(tools, t);
    return tools;
}

/* ── System prompt ───────────────────────────────────────────────────────── */

static char* make_system_prompt(const char* db_path) {
    char* schema     = db_schema(db_path);
    char* dept_names = db_values(db_path,
        "SELECT dept_name FROM departments ORDER BY dept_name");
    char* titles     = db_values(db_path,
        "SELECT DISTINCT title FROM titles ORDER BY title");

    Buf out = { malloc(16384), 0, 16384 };

#define APPEND(s) do { \
    size_t _l = strlen(s); \
    if (out.len + _l + 1 > out.cap) { out.cap += _l + 4096; out.buf = realloc(out.buf, out.cap); } \
    memcpy(out.buf + out.len, s, _l); out.len += _l; } while(0)

    APPEND("You are an HR assistant. You MUST call query_db for every question. Never answer from memory.\n\n");
    APPEND("Schema:\n"); APPEND(schema); APPEND("\n\n");
    APPEND("Exact department names: "); APPEND(dept_names); APPEND("\n");
    APPEND("Exact job titles: ");       APPEND(titles);     APPEND("\n\n");
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

    out.buf[out.len] = '\0';
    free(schema); free(dept_names); free(titles);
    return out.buf;
}

/* ── Emit helpers ────────────────────────────────────────────────────────── */

static void emit_sql_query(const char* sql) {
    cJSON* o = cJSON_CreateObject();
    cJSON_AddStringToObject(o, "type", "sql_query");
    cJSON_AddStringToObject(o, "sql",  sql);
    char* s = cJSON_PrintUnformatted(o);
    fputs("sql_query\n", stdout);   /* topic prefix for runtime routing */
    fputs(s, stdout);
    free(s); cJSON_Delete(o);
}

static void emit_agent_response(const char* answer, const char* query,
                                const char* session_id) {
    cJSON* o = cJSON_CreateObject();
    cJSON_AddStringToObject(o, "type",       "agent_response");
    cJSON_AddStringToObject(o, "answer",     answer     ? answer     : "");
    cJSON_AddStringToObject(o, "query",      query      ? query      : "");
    cJSON_AddStringToObject(o, "session_id", session_id ? session_id : "");
    char* s = cJSON_PrintUnformatted(o);
    fputs("agent_response\n", stdout);
    fputs(s, stdout);
    free(s); cJSON_Delete(o);
}

/* ── State serialisation ─────────────────────────────────────────────────── */

/* state stored in LMDB: {"round":N,"query":"...","session_id":"...","messages":[...]} */

static cJSON* state_to_json(int round, const char* query,
                             const char* session_id, cJSON* messages) {
    cJSON* s = cJSON_CreateObject();
    cJSON_AddNumberToObject(s, "round",      round);
    cJSON_AddStringToObject(s, "query",      query      ? query      : "");
    cJSON_AddStringToObject(s, "session_id", session_id ? session_id : "");
    cJSON_AddItemReferenceToObject(s, "messages", messages);
    return s;
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(void) {
    static char g_in[MAX_IN];
    size_t n = fread(g_in, 1, sizeof(g_in) - 1, stdin);
    g_in[n] = '\0';

    const char* corr_id  = getenv("ACTOR_CORRELATION_ID");
    const char* ollama   = getenv("OLLAMA_URL");   if (!ollama)  ollama  = "http://localhost:11434";
    const char* model    = getenv("OLLAMA_MODEL"); if (!model)   model   = "gemma4:e2b";
    const char* db_path  = getenv("EMPLOYEE_DB");  if (!db_path) db_path = "./employee.db";

    if (!corr_id || !*corr_id) {
        emit_agent_response("internal error: no correlation id", "", "");
        return 1;
    }

    cJSON* payload = cJSON_Parse(g_in);
    if (!payload) {
        emit_agent_response("internal error: bad payload json", "", "");
        return 1;
    }

    const char* type = cJSON_GetStringValue(cJSON_GetObjectItem(payload, "type"));

    MDB_env* env = lmdb_open();

    cJSON*  messages   = NULL;
    int     round      = 0;
    char*   query      = NULL;
    char*   session_id = NULL;

    if (type && strcmp(type, "sql_result") == 0) {
        /* ── Round 2+: load conversation state from LMDB ─────────────────── */
        char* raw = env ? state_load(env, corr_id) : NULL;
        if (!raw) {
            fprintf(stderr, "[llm-agent] no state for corr=%s, discarding\n", corr_id);
            cJSON_Delete(payload);
            if (env) mdb_env_close(env);
            return 0;
        }

        cJSON* state = cJSON_Parse(raw);
        free(raw);

        round      = (int)cJSON_GetNumberValue(cJSON_GetObjectItem(state, "round"));
        query      = strdup(cJSON_GetStringValue(cJSON_GetObjectItem(state, "query")));
        session_id = strdup(cJSON_GetStringValue(cJSON_GetObjectItem(state, "session_id")));
        messages   = cJSON_DetachItemFromObject(state, "messages");
        cJSON_Delete(state);

        /* find the last tool_call_id in messages to correlate this result */
        cJSON* last_tc_id = NULL;
        int mlen = cJSON_GetArraySize(messages);
        for (int i = mlen - 1; i >= 0; i--) {
            cJSON* m = cJSON_GetArrayItem(messages, i);
            cJSON* tcs = cJSON_GetObjectItem(m, "tool_calls");
            if (tcs && cJSON_GetArraySize(tcs) > 0) {
                last_tc_id = cJSON_GetObjectItem(
                    cJSON_GetArrayItem(tcs, 0), "id");
                break;
            }
        }

        /* append tool result message */
        cJSON* rows = cJSON_GetObjectItem(payload, "rows");
        cJSON* tool_msg = cJSON_CreateObject();
        cJSON_AddStringToObject(tool_msg, "role", "tool");
        char* rows_str = cJSON_PrintUnformatted(rows ? rows : cJSON_CreateArray());
        cJSON_AddStringToObject(tool_msg, "content", rows_str);
        free(rows_str);
        if (last_tc_id)
            cJSON_AddStringToObject(tool_msg, "tool_call_id",
                                    cJSON_GetStringValue(last_tc_id));
        cJSON_AddItemToArray(messages, tool_msg);

    } else {
        /* ── Round 1: fresh conversation from user_message ───────────────── */
        const char* q  = cJSON_GetStringValue(cJSON_GetObjectItem(payload, "query"));
        const char* sid= cJSON_GetStringValue(cJSON_GetObjectItem(payload, "session_id"));
        query      = strdup(q   ? q   : "");
        session_id = strdup(sid ? sid : "");

        char* sys_prompt = make_system_prompt(db_path);
        messages = cJSON_CreateArray();

        cJSON* sys_msg = cJSON_CreateObject();
        cJSON_AddStringToObject(sys_msg, "role",    "system");
        cJSON_AddStringToObject(sys_msg, "content", sys_prompt);
        cJSON_AddItemToArray(messages, sys_msg);
        free(sys_prompt);

        cJSON* usr_msg = cJSON_CreateObject();
        cJSON_AddStringToObject(usr_msg, "role",    "user");
        cJSON_AddStringToObject(usr_msg, "content", query);
        cJSON_AddItemToArray(messages, usr_msg);
    }

    cJSON_Delete(payload);

    if (round >= MAX_ROUNDS) {
        emit_agent_response("(max rounds reached without a final answer)", query, session_id);
        free(query); free(session_id);
        if (messages) cJSON_Delete(messages);
        if (env) mdb_env_close(env);
        return 0;
    }
    round++;

    /* ── Call Ollama ─────────────────────────────────────────────────────── */

    /* round 1: offer tools so model can call query_db
     * round 2+: strip system message + no tools → force final answer */
    cJSON* msgs_to_send = messages;
    cJSON* stripped     = NULL;
    cJSON* tools        = NULL;

    if (round == 1) {
        tools = make_tools();
    } else {
        /* drop system message to free context window */
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

    char* resp_raw = ollama_chat(ollama, model, msgs_to_send, tools);
    if (stripped) cJSON_Delete(stripped);
    if (tools)    cJSON_Delete(tools);

    if (!resp_raw) {
        emit_agent_response("error: no response from ollama", query, session_id);
        free(query); free(session_id);
        cJSON_Delete(messages);
        if (env) mdb_env_close(env);
        return 1;
    }

    cJSON* resp = cJSON_Parse(resp_raw);
    free(resp_raw);

    if (!resp) {
        emit_agent_response("error: bad json from ollama", query, session_id);
        free(query); free(session_id);
        cJSON_Delete(messages);
        if (env) mdb_env_close(env);
        return 1;
    }

    cJSON* msg       = cJSON_GetObjectItem(resp, "message");
    cJSON* tool_calls= cJSON_GetObjectItem(msg,  "tool_calls");
    const char* content = cJSON_GetStringValue(cJSON_GetObjectItem(msg, "content"));

    if (tool_calls && cJSON_GetArraySize(tool_calls) > 0) {
        /* ── Model wants to call a tool ─────────────────────────────────── */

        /* append assistant message (with tool_calls) to conversation */
        cJSON* asst = cJSON_Duplicate(msg, 1);
        cJSON_AddItemToArray(messages, asst);

        /* extract sql from first tool call */
        cJSON* tc   = cJSON_GetArrayItem(tool_calls, 0);
        cJSON* args = cJSON_GetObjectItem(
                        cJSON_GetObjectItem(tc, "function"), "arguments");
        const char* sql = cJSON_GetStringValue(cJSON_GetObjectItem(args, "sql"));

        if (!sql || !*sql) {
            emit_agent_response("error: model returned empty sql", query, session_id);
        } else {
            /* save conversation state to LMDB, emit sql_query to bus */
            cJSON* st = state_to_json(round, query, session_id, messages);
            char*  st_str = cJSON_PrintUnformatted(st);
            if (env) state_save(env, corr_id, st_str);
            free(st_str);
            cJSON_Delete(st);

            emit_sql_query(sql);
        }

    } else {
        /* ── Final answer ────────────────────────────────────────────────── */
        if (env) state_del(env, corr_id);
        emit_agent_response(content, query, session_id);
    }

    cJSON_Delete(resp);
    cJSON_Delete(messages);
    free(query);
    free(session_id);
    if (env) mdb_env_close(env);
    return 0;
}
