#!/bin/sh
# announce-tools.sh — called by MCP actors on startup to register tools
# Reads MCP tools/list response from stdin, publishes to mesh.
# Env: ACTOR_BUS_PUB (mesh pub endpoint), ACTOR_ID (this actor's name)

TOOLS=$(cat | sed 's/.*"tools":\[/[/;s/\].*/]/' 2>/dev/null)
[ -z "$TOOLS" ] && exit 0

NAME="${ACTOR_ID:-unknown}"
BUS_PUB="${ACTOR_BUS_PUB:-tcp://127.0.0.1:5557}"

# Build announce payload: {type: "_tool_announce", name: "...", tools: [...]}
PAYLOAD="{\"type\":\"_tool_announce\",\"name\":\"$NAME\",\"tools\":$TOOLS}"

# Publish using nng pub via a one-shot helper
# For now, write to a file that the registry can pick up
echo "$PAYLOAD"
