#define _POSIX_C_SOURCE 200809L
/*
 * handlers/sqlite-tool.c — SQLite query executor actor handler
 *
 * Receives: mpack {type:"sql_query", sql:"SELECT ..."}
 * Emits:    mpack {type:"sql_result", rows:[{col:val,...},...]}
 *
 * Env: EMPLOYEE_DB
 * Deps: libsqlite3, mpack
 */

#include "mpack.h"
#include <sqlite3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_IN    (256 * 1024)
#define MAX_OUT   (1   * 1024 * 1024)
#define MAX_ROWS  20

static char g_in[MAX_IN];
static char g_capped[MAX_IN + 40];
static char g_out[MAX_OUT];

static void emit_error(const char* msg) {
    mpack_writer_t w;
    mpack_writer_init(&w, g_out, sizeof(g_out));
    mpack_start_map(&w, 3);
    mpack_write_cstr(&w, "type");  mpack_write_cstr(&w, "sql_result");
    mpack_write_cstr(&w, "error"); mpack_write_cstr(&w, msg);
    mpack_write_cstr(&w, "rows");  mpack_start_array(&w, 0); mpack_finish_array(&w);
    mpack_finish_map(&w);
    fwrite(g_out, 1, mpack_writer_buffer_used(&w), stdout);
    mpack_writer_destroy(&w);
}

int main(void) {
    size_t n = fread(g_in, 1, sizeof(g_in) - 1, stdin);

    mpack_reader_t r;
    mpack_reader_init_data(&r, g_in, n);

    static const char* keys[] = { "type", "sql" };
    bool found[2] = { false, false };
    char sql_val[MAX_IN];
    sql_val[0] = '\0';

    uint32_t map_sz = mpack_expect_map_max(&r, 8);
    for (uint32_t i = 0; i < map_sz && mpack_reader_error(&r) == mpack_ok; i++) {
        switch (mpack_expect_key_cstr(&r, keys, found, 2)) {
            case 0: mpack_discard(&r); break;
            case 1: mpack_expect_cstr(&r, sql_val, sizeof(sql_val)); break;
            default: mpack_discard(&r); break;
        }
    }
    mpack_done_map(&r);
    mpack_reader_destroy(&r);

    if (!sql_val[0]) { emit_error("missing sql field"); return 0; }

    /* strip trailing whitespace and semicolons the LLM often appends */
    {
        size_t l = strlen(sql_val);
        while (l > 0 && (sql_val[l-1] == ';' || sql_val[l-1] == ' '
                         || sql_val[l-1] == '\n' || sql_val[l-1] == '\r'))
            sql_val[--l] = '\0';
    }

    const char* db_path = getenv("EMPLOYEE_DB");
    if (!db_path) db_path = "./employee.db";

    sqlite3* db;
    if (sqlite3_open_v2(db_path, &db, SQLITE_OPEN_READONLY, NULL) != SQLITE_OK) {
        emit_error(sqlite3_errmsg(db));
        sqlite3_close(db);
        return 0;
    }

    /* wrap in LIMIT cap so the tool can never return unbounded rows */
    snprintf(g_capped, sizeof(g_capped), "SELECT * FROM (%s) LIMIT %d", sql_val, MAX_ROWS);

    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, g_capped, -1, &stmt, NULL) != SQLITE_OK) {
        emit_error(sqlite3_errmsg(db));
        sqlite3_close(db);
        return 0;
    }

    int ncols = sqlite3_column_count(stmt);

    /* pass 1: count rows */
    int nrows = 0;
    int step_rc;
    while ((step_rc = sqlite3_step(stmt)) == SQLITE_ROW) nrows++;
    if (step_rc != SQLITE_DONE) {
        emit_error(sqlite3_errmsg(db));
        sqlite3_finalize(stmt);
        sqlite3_close(db);
        return 0;
    }
    sqlite3_reset(stmt);

    /* pass 2: write mpack output */
    mpack_writer_t w;
    mpack_writer_init(&w, g_out, sizeof(g_out));
    mpack_start_map(&w, 2);
    mpack_write_cstr(&w, "type"); mpack_write_cstr(&w, "sql_result");
    mpack_write_cstr(&w, "rows"); mpack_start_array(&w, (uint32_t)nrows);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        mpack_start_map(&w, (uint32_t)ncols);
        for (int i = 0; i < ncols; i++) {
            mpack_write_cstr(&w, sqlite3_column_name(stmt, i));
            switch (sqlite3_column_type(stmt, i)) {
                case SQLITE_INTEGER:
                    mpack_write_i64(&w, sqlite3_column_int64(stmt, i)); break;
                case SQLITE_FLOAT:
                    mpack_write_double(&w, sqlite3_column_double(stmt, i)); break;
                case SQLITE_TEXT:
                    mpack_write_cstr(&w, (const char*)sqlite3_column_text(stmt, i)); break;
                default:
                    mpack_write_nil(&w); break;
            }
        }
        mpack_finish_map(&w);
    }

    mpack_finish_array(&w);
    mpack_finish_map(&w);

    sqlite3_finalize(stmt);
    sqlite3_close(db);

    fwrite(g_out, 1, mpack_writer_buffer_used(&w), stdout);
    mpack_writer_destroy(&w);
    return 0;
}
