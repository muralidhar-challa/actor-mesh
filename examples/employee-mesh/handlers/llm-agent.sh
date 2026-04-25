#!/bin/sh
# handlers/llm-agent.sh
#
# Receives: sql_result payload  { "natural": "...", "rows": [...], "session_id": "..." }
# Emits:    agent_response      { "answer": "...", "session_id": "..." }
#
# Dependencies: curl, jq
# Env:
#   OLLAMA_URL    Ollama endpoint  default http://localhost:11434
#   OLLAMA_MODEL  model to use     default llama3.2

set -e

OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2}"

payload=$(cat)

natural=$(printf '%s' "$payload"    | jq -r '.natural')
session=$(printf '%s' "$payload"    | jq -r '.session_id')
rows=$(printf '%s' "$payload"       | jq -c '.rows')
sql=$(printf '%s' "$payload"        | jq -r '.sql')

# build prompt
prompt="You are an HR assistant with access to employee data.

The user asked: ${natural}

The following SQL was run: ${sql}

Results:
${rows}

Please answer the user's question clearly and concisely based on the data above.
If the results are empty, say so politely."

# call Ollama
request=$(jq -cn \
    --arg model "$OLLAMA_MODEL" \
    --arg prompt "$prompt" \
    '{
        model:  $model,
        prompt: $prompt,
        stream: false
    }')

response=$(printf '%s' "$request" | curl -s \
    -X POST "${OLLAMA_URL}/api/generate" \
    -H "content-type: application/json" \
    -d @-)

answer=$(printf '%s' "$response" | jq -r '.response // "I could not generate a response."')

# emit agent_response
jq -cn \
    --arg answer "$answer" \
    --arg session "$session" \
    --arg natural "$natural" \
    '{
        type:       "agent_response",
        answer:     $answer,
        session_id: $session,
        query:      $natural
    }'
