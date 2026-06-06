#!/bin/sh
# tool-bridge.sh — generic mesh → MCP adapter
# Env: MCP_SERVER, MCP_TOOL, MCP_ARG, BRIDGE_TOPIC, BRIDGE_LIB
# Receives mpack from stdin, sends init+call to MCP server, returns mpack.

LIB="${BRIDGE_LIB:-$(dirname "$0")/../lib}"
MCP="${MCP_SERVER:-mcp-server-sqlite}"
TOOL="${MCP_TOOL:-read_query}"
ARG="${MCP_ARG:-sql}"
TOPIC="${BRIDGE_TOPIC:-sql_result}"

# Save stdin to file (binary-safe, reusable)
INPUT=$(mktemp)
cat > "$INPUT"

# Extract argument value from mpack
VAL=$("$LIB/mpack-get" "$ARG" < "$INPUT" 2>/dev/null)
[ -z "$VAL" ] && { rm -f "$INPUT"; exit 0; }

# Send initialize + tool call to a single MCP process
REQ=$(mktemp)
cat > "$REQ" <<REQUESTS
{"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"mesh","version":"1.0"}}}
{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"$TOOL","arguments":{"$ARG":"$VAL"}}}
REQUESTS

RESPONSE=$(cat "$REQ" | $MCP 2>/dev/null | tail -1)
rm -f "$INPUT" "$REQ"

# Extract text content from MCP response
TEXT=$(echo "$RESPONSE" | jq -r '.result.content[0].text // empty' 2>/dev/null)
[ -z "$TEXT" ] && TEXT="error: no response from MCP server"

# Emit mesh result
printf "%s\n" "$TOPIC"
"$LIB/mpack-pack" "text" "$TEXT"
