/*
 * handlers/sqlite-tool.c — SQLite query executor actor handler
 *
 * Receives: sql_query  { "sql": "SELECT ..." }
 * Emits:    sql_result { "rows": [...] }
 *
 * Correlation chain is carried by the actor runtime — no session tracking needed.
 *
 * Env: EMPLOYEE_DB
 * Deps: libsqlite3, cJSON
 */

#include "../vendor/cjson/cJSON.h"
#include <sqlite3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_IN    (256 * 1024)
#define MAX_ROWS  20

static char g_in[MAX_IN];

static void emit_error(const char* msg) {
    cJSON* e = cJSON_CreateObject();
    cJSON_AddStringToObject(e, "type",  "sql_result");
    cJSON_AddStringToObject(e, "error", msg);
    cJSON_AddItemToObject(e, "rows", cJSON_CreateArray());
    char* s = cJSON_PrintUnformatted(e);
    fputs(s, stdout);
    free(s);
    cJSON_Delete(e);
}

int main(void) {
    size_t n = fread(g_in, 1, sizeof(g_in) - 1, stdin);
    g_in[n] = '\0';

    cJSON* root = cJSON_Parse(g_in);
    if (!root) { emit_error("json parse failed"); return 1; }

    const char* sql = cJSON_GetStringValue(cJSON_GetObjectItem(root, "sql"));
    if (!sql || !*sql) {
        cJSON_Delete(root);
        emit_error("missing sql field");
        return 1;
    }

    const char* db_path = getenv("EMPLOYEE_DB");
    if (!db_path) db_path = "./employee.db";

    sqlite3* db;
    if (sqlite3_open_v2(db_path, &db, SQLITE_OPEN_READONLY, NULL) != SQLITE_OK) {
        cJSON_Delete(root);
        emit_error("db open failed");
        return 1;
    }

    /* wrap in a LIMIT cap so the tool can never return unbounded rows */
    char capped[4096];
    snprintf(capped, sizeof(capped),
             "SELECT * FROM (%s) LIMIT %d", sql, MAX_ROWS);

    cJSON* rows = cJSON_CreateArray();
    sqlite3_stmt* stmt;

    if (sqlite3_prepare_v2(db, capped, -1, &stmt, NULL) == SQLITE_OK) {
        int ncols = sqlite3_column_count(stmt);
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            cJSON* row = cJSON_CreateObject();
            for (int i = 0; i < ncols; i++) {
                const char* col = sqlite3_column_name(stmt, i);
                switch (sqlite3_column_type(stmt, i)) {
                    case SQLITE_INTEGER:
                        cJSON_AddNumberToObject(row, col,
                            (double)sqlite3_column_int64(stmt, i));
                        break;
                    case SQLITE_FLOAT:
                        cJSON_AddNumberToObject(row, col,
                            sqlite3_column_double(stmt, i));
                        break;
                    case SQLITE_TEXT:
                        cJSON_AddStringToObject(row, col,
                            (const char*)sqlite3_column_text(stmt, i));
                        break;
                    default:
                        cJSON_AddNullToObject(row, col);
                }
            }
            cJSON_AddItemToArray(rows, row);
        }
        sqlite3_finalize(stmt);
    } else {
        fprintf(stderr, "[sqlite-tool] prepare failed: %s\n",
                sqlite3_errmsg(db));
    }

    sqlite3_close(db);
    cJSON_Delete(root);

    cJSON* result = cJSON_CreateObject();
    cJSON_AddStringToObject(result, "type", "sql_result");
    cJSON_AddItemToObject(result, "rows", rows);

    char* out = cJSON_PrintUnformatted(result);
    fputs(out, stdout);
    free(out);
    cJSON_Delete(result);
    return 0;
}
