#!/bin/sh
# handler.sh — LLM agent handler via Ollama
#
# Reads payload from stdin: {"query": "..."}
# Writes result to stdout:  {"answer": "...", "model": "..."}
#
# Dependencies: curl, jq
# Env: OLLAMA_HOST (default http://localhost:11434)

set -e

OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"

payload=$(cat)

request=$(printf '%s' "$payload" | jq -c '{
  model: "gemma4:e2b",
  prompt: .query,
  stream: false
}')

response=$(printf '%s' "$request" | curl -s \
  -X POST "${OLLAMA_HOST}/api/generate" \
  -H "content-type: application/json" \
  -d @-)

printf '%s' "$response" | jq -c '{
  answer: .response,
  model:  .model
}'
