#!/bin/sh
# handlers/llm-agent.sh — agentic tool-calling loop
#
# Receives: user_message payload  { "query": "...", "session_id": "..." }
# Emits:    agent_response        { "answer": "...", "session_id": "..." }
#
# Dependencies: curl, jq, sqlite3
# Env: OLLAMA_URL, OLLAMA_MODEL, EMPLOYEE_DB

set -e

OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-gemma4:e2b}"
EMPLOYEE_DB="${EMPLOYEE_DB:-./employee.db}"
MAX_ROWS=20

payload=$(cat)
natural=$(printf '%s' "$payload" | jq -r '.query')
session=$(printf '%s' "$payload" | jq -r '.session_id')

schema=$(sqlite3 "$EMPLOYEE_DB" .schema 2>/dev/null)
dept_names=$(sqlite3 "$EMPLOYEE_DB" "SELECT dept_name FROM departments ORDER BY dept_name;" 2>/dev/null | tr '\n' ', ' | sed 's/, $//')
titles_sample=$(sqlite3 "$EMPLOYEE_DB" "SELECT DISTINCT title FROM titles ORDER BY title;" 2>/dev/null | tr '\n' ', ' | sed 's/, $//')

TOOL_DEF='[{"type":"function","function":{"name":"query_db","description":"Execute a SQLite query against the employee database. Always include LIMIT 20.","parameters":{"type":"object","properties":{"sql":{"type":"string","description":"SQLite query with LIMIT 20"}},"required":["sql"]}}}]'

MSGS=$(mktemp)
RESP=$(mktemp)
ROWS=$(mktemp)
CALLS=$(mktemp)
trap 'rm -f "$MSGS" "$RESP" "$ROWS" "$CALLS"' EXIT

system_prompt="You are an HR assistant. You MUST call query_db for every question. Never answer from memory.

Schema:
${schema}

Exact department names in DB: ${dept_names}
Exact job titles in DB: ${titles_sample}

Rules:
- dept_no is a short code (d001 etc). Always JOIN departments to filter by dept_name.
- Always use LIKE for name matching (never exact =)
- Use to_date='9999-01-01' to get current records only
- Always LIMIT 20
- There is NO 'Engineering' department — use 'Development'

Correct query examples:
-- dept manager: SELECT e.first_name, e.last_name, d.dept_name FROM employees e JOIN dept_manager dm ON e.emp_no=dm.emp_no JOIN departments d ON dm.dept_no=d.dept_no WHERE d.dept_name LIKE '%Development%' AND dm.to_date='9999-01-01' LIMIT 20
-- by title:     SELECT e.first_name, e.last_name, t.title FROM employees e JOIN titles t ON e.emp_no=t.emp_no WHERE t.title LIKE '%Engineer%' AND t.to_date='9999-01-01' LIMIT 20
-- top salary:   SELECT e.first_name, e.last_name, s.salary FROM employees e JOIN salaries s ON e.emp_no=s.emp_no WHERE s.to_date='9999-01-01' ORDER BY s.salary DESC LIMIT 20"

jq -cn --arg q "$natural" --arg sys "$system_prompt" \
    '[{"role":"system","content":$sys},{"role":"user","content":$q}]' > "$MSGS"

answer=""
i=0
while [ "$i" -lt 5 ]; do
    i=$((i + 1))

    has_tool_results=$(jq 'map(select(.role=="tool")) | length > 0' "$MSGS")

    if [ "$has_tool_results" = "true" ]; then
        # round 2+: drop system message to free context window, drop tools to force final answer
        req_extra='{"stream":false}'
        msgs_to_send=$(jq 'map(select(.role != "system"))' "$MSGS")
    else
        req_extra=$(jq -cn --argjson tools "$TOOL_DEF" '{tools:$tools,"stream":false}')
        msgs_to_send=$(cat "$MSGS")
    fi

    jq -cn \
        --arg model "$OLLAMA_MODEL" \
        --argjson msgs "$msgs_to_send" \
        --argjson extra "$req_extra" \
        '{model:$model,messages:$msgs} + $extra' \
        | curl -s -X POST "${OLLAMA_URL}/api/chat" \
            -H "content-type: application/json" \
            -d @- > "$RESP"

    tool_calls=$(jq -c 'if (.message.tool_calls // []) | length > 0 then .message.tool_calls else empty end' "$RESP")
    answer=$(jq -r '.message.content // ""' "$RESP")

    if [ -z "$tool_calls" ]; then
        break
    fi

    # append assistant message (with tool_calls) to conversation
    jq -c --slurpfile resp "$RESP" '. + [$resp[0].message]' "$MSGS" > "${MSGS}.tmp"
    mv "${MSGS}.tmp" "$MSGS"

    # write call list to file — avoids pipe subshell so MSGS updates are visible to parent
    printf '%s' "$tool_calls" | jq -c '.[]' > "$CALLS"

    while IFS= read -r call; do
        call_id=$(printf '%s' "$call" | jq -r '.id')
        sql=$(printf '%s' "$call" | jq -r '.function.arguments.sql')

        # coerce exact = to LIKE for name columns regardless of what model generates
        sql=$(printf '%s' "$sql" \
            | sed "s/dept_name *= *'\([^']*\)'/dept_name LIKE '%\1%'/g" \
            | sed "s/title *= *'\([^']*\)'/title LIKE '%\1%'/g")

        capped="SELECT * FROM ($sql) LIMIT $MAX_ROWS"
        sqlite3 -json "$EMPLOYEE_DB" "$capped" 2>/dev/null > "$ROWS" || printf '[]' > "$ROWS"

        # append tool result to conversation
        jq -c \
            --arg id "$call_id" \
            --slurpfile rows "$ROWS" \
            '. + [{"role":"tool","content":($rows[0]|tostring),"tool_call_id":$id}]' \
            "$MSGS" > "${MSGS}.tmp"
        mv "${MSGS}.tmp" "$MSGS"
    done < "$CALLS"
done

jq -cn \
    --arg answer "$answer" \
    --arg session "$session" \
    --arg query "$natural" \
    '{type:"agent_response",answer:$answer,session_id:$session,query:$query}'
