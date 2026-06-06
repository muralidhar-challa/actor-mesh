#!/bin/sh
# mcp-bridge.sh — generic mesh → MCP adapter
# Receives mpack payload from stdin, forwards to MCP server.
# Env: MCP_SERVER (path), MCP_TOOL (name), MCP_ARGS (key to extract)
# Output: topic_prefix\n + mpack result

# --- step 1: extract argument from mpack ---
VAL=$(handlers/lib/mpack-get "${MCP_ARGS:-sql}" 2>/dev/null)
[ -z "$VAL" ] && exit 0

# --- step 2: build MCP request ---
REQUEST=$(printf '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"%s","arguments":{"%s":"%s"}}}' \
  "${MCP_TOOL:-query}" "${MCP_ARGS:-sql}" "$VAL")

# --- step 3: call MCP server ---
RESPONSE=$(printf '%s\n' "$REQUEST" | ${MCP_SERVER:-mcp-server-sqlite} 2>/dev/null)

# --- step 4: extract text from MCP response ---
TEXT=$(echo "$RESPONSE" | sed 's/.*"text":"//;s/"}].*//;s/\\"/"/g;s/\\n/\
/g')

# --- step 5: emit as mesh result ---
printf "sql_result\n"
handlers/lib/mpack-pack "text" "$TEXT"
