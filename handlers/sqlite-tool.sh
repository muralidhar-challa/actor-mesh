#!/bin/sh
# handlers/sqlite-tool.sh
#
# Receives: sql_query payload  { "natural": "...", "query": "...", "session_id": "..." }
# Emits:    sql_result payload { "rows": [...], "natural": "...", "session_id": "..." }
#
# Schema: employees, departments, dept_emp, titles, salaries
# Dependencies: sqlite3, jq

set -e

payload=$(cat)
natural=$(printf '%s' "$payload" | jq -r '.natural')
session=$(printf '%s' "$payload" | jq -r '.session_id')

query=$(printf '%s' "$natural" | tr '[:upper:]' '[:lower:]')

case "$query" in
    *salary*|*paid*|*earn*)
        sql="SELECT e.emp_no, e.first_name, e.last_name, s.salary
             FROM employees e JOIN salaries s ON e.emp_no = s.emp_no
             WHERE s.to_date = '9999-01-01'
             ORDER BY s.salary DESC LIMIT 20"
        ;;
    *manager*)
        sql="SELECT e.emp_no, e.first_name, e.last_name, d.dept_name
             FROM employees e
             JOIN dept_manager dm ON e.emp_no = dm.emp_no
             JOIN departments d ON dm.dept_no = d.dept_no
             WHERE dm.to_date = '9999-01-01'"
        ;;
    *department*|*dept*)
        dept=$(printf '%s' "$natural" | grep -oiE '\b[A-Za-z]+\b' | tail -1)
        sql="SELECT e.emp_no, e.first_name, e.last_name, d.dept_name, t.title
             FROM employees e
             JOIN dept_emp de ON e.emp_no = de.emp_no
             JOIN departments d ON de.dept_no = d.dept_no
             JOIN titles t ON e.emp_no = t.emp_no
             WHERE de.to_date = '9999-01-01' AND t.to_date = '9999-01-01'
             AND d.dept_name LIKE '%${dept}%'
             LIMIT 20"
        ;;
    *engineer*|*title*|*role*)
        sql="SELECT e.emp_no, e.first_name, e.last_name, t.title, d.dept_name
             FROM employees e
             JOIN titles t ON e.emp_no = t.emp_no
             JOIN dept_emp de ON e.emp_no = de.emp_no
             JOIN departments d ON de.dept_no = d.dept_no
             WHERE t.to_date = '9999-01-01' AND de.to_date = '9999-01-01'
             AND t.title LIKE '%Engineer%'
             LIMIT 20"
        ;;
    *hire*|*recent*|*new*)
        sql="SELECT emp_no, first_name, last_name, hire_date
             FROM employees
             ORDER BY hire_date DESC LIMIT 10"
        ;;
    *)
        sql="SELECT e.emp_no, e.first_name, e.last_name, t.title, d.dept_name
             FROM employees e
             JOIN titles t ON e.emp_no = t.emp_no
             JOIN dept_emp de ON e.emp_no = de.emp_no
             JOIN departments d ON de.dept_no = d.dept_no
             WHERE t.to_date = '9999-01-01' AND de.to_date = '9999-01-01'
             LIMIT 20"
        ;;
esac

rows=$(sqlite3 -json "${EMPLOYEE_DB:-./employee.db}" "$sql" 2>/dev/null || echo "[]")

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
