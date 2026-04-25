#!/bin/sh
# sql.sh — SQLite query handler
#
# Reads payload from stdin: {"query": "SELECT ..."}
# Writes result to stdout:  {"rows": "..."}
#
# Dependencies: sqlite3, jq
# Env: DATABASE_PATH (path to .db file)

set -e

query=$(cat | jq -r '.query')
result=$(sqlite3 -json "$DATABASE_PATH" "$query")
printf '{"rows":%s}' "${result:-[]}"
