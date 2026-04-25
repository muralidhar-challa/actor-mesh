#!/bin/sh
# handler.sh — example LLM agent handler
#
# Reads payload from stdin (raw bytes, your schema)
# Writes result to stdout (raw bytes, your schema)
# Knows nothing about the mesh, ZMQ, or headers
#
# Dependencies: curl, jq — available on alpine

set -e

# read payload from stdin
payload=$(cat)

# build LLM request with jq
request=$(printf '%s' "$payload" | jq -c '{
  model: "claude-haiku-4-5-20251001",
  max_tokens: 1000,
  messages: [{
    role: "user",
    content: .query
  }]
}')

# call LLM API
response=$(printf '%s' "$request" | curl -s \
  -X POST https://api.anthropic.com/v1/messages \
  -H "x-api-key: ${ANTHROPIC_API_KEY}" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d @-)

# extract result and write to stdout
printf '%s' "$response" | jq -c '{
  answer: .content[0].text,
  model:  .model,
  tokens: .usage.output_tokens
}'
