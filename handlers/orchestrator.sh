#!/bin/sh
# orchestrator.sh — orchestrator handler
#
# Reads payload from stdin: {"id": "actor-id", "inbox": N, "outbox": N}
# Writes result to stdout:  action taken or {}
#
# Logs backlog warnings to stderr. No external dependencies beyond jq.

set -e

payload=$(cat)
inbox=$(printf '%s' "$payload" | jq -r '.inbox')
id=$(printf '%s' "$payload"    | jq -r '.id')

if [ "$inbox" -gt 100 ]; then
    printf '[orchestrator] backlog on %s: inbox=%s\n' "$id" "$inbox" >&2
    printf '{"action":"backlog_alert","target":"%s","inbox":%s}' "$id" "$inbox"
else
    printf '{}'
fi
