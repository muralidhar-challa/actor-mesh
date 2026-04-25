#!/bin/sh
# handlers/sqlite-tool.sh
#
# Receives: sql_query payload  { "natural": "...", "query": "...", "session_id": "..." }
# Emits:    sql_result payload { "rows": [...], "natural": "...", "session_id": "..." }
#
# Dependencies: sqlite3, jq

set -e

payload=$(cat)

# extract the SQL query
# in production this would use the LLM to generate SQL from natural language
# here we use a simple keyword match for the employee DB
natural=$(printf '%s' "$payload" | jq -r '.natural')
session=$(printf '%s' "$payload" | jq -r '.session_id')

# simple keyword routing — production would use LLM for text-to-SQL
query=$(printf '%s' "$natural" | tr '[:upper:]' '[:lower:]')

case "$query" in
    *all*|*list*|*everyone*)
        sql="SELECT id, name, department, role, salary FROM employees ORDER BY name"
        ;;
    *department*)
        dept=$(printf '%s' "$natural" | grep -oiE '[a-zA-Z]+(?= department)' | head -1)
        sql="SELECT id, name, department, role, salary FROM employees WHERE department LIKE '%${dept}%'"
        ;;
    *salary*|*paid*|*earn*)
        sql="SELECT id, name, department, role, salary FROM employees ORDER BY salary DESC"
        ;;
    *hire*|*recent*|*new*)
        sql="SELECT id, name, department, role, hire_date FROM employees ORDER BY hire_date DESC LIMIT 10"
        ;;
    *)
        sql="SELECT id, name, department, role FROM employees LIMIT 20"
        ;;
esac

# execute against employee SQLite DB
rows=$(sqlite3 -json "${EMPLOYEE_DB:-/db/employees.db}" "$sql" 2>/dev/null || echo "[]")

# emit sql_result
printf '%s' "$payload" | jq -c \
    --argjson rows "$rows" \
    --arg sql "$sql" \
    '{
        type:       "sql_result",
        natural:    .natural,
        session_id: .session_id,
        sql:        $sql,
        rows:       $rows
    }'
