#!/bin/bash
# test-mcp-mesh.sh — end-to-end MCP mesh test
# Starts proxy + registry + SQLite MCP + agent, sends query, waits for response.
# Usage: LLM_MODEL=granite4.1:8b bash test-mcp-mesh.sh

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT/examples/employee-mesh"

BUS_SUB=tcp://127.0.0.1:5556
BUS_PUB=tcp://127.0.0.1:5557
COMMON="ACTOR_BUS_SUB=$BUS_SUB ACTOR_BUS_PUB=$BUS_PUB ACTOR_HEARTBEAT_MS=2000"
MODEL="${LLM_MODEL:-granite4.1:8b}"
BASE_URL="${LLM_BASE_URL:-http://localhost:11434}"
LMDB="/tmp/mcp-test"
QUERY="${1:-how many employees are there?}"

echo "=== cleaning up ==="
pkill -9 mesh-proxy 2>/dev/null || true
pkill -9 actor 2>/dev/null || true
sleep 0.3
rm -rf "$LMDB"
mkdir -p "$LMDB"/{reg,db,ag}

echo "=== building ==="
make handlers 2>&1 | tail -1

echo "=== proxy ==="
PROXY_SUB_BIND=tcp://127.0.0.1:5557 PROXY_PUB_BIND=tcp://127.0.0.1:5556 \
  "$ROOT/mesh-proxy" >>"$LMDB/proxy.log" 2>&1 &
sleep 0.3

echo "=== tool registry ==="
$COMMON ACTOR_ID=tool-registry \
  ACTOR_TOPIC=_tool_announce,_tool_discover ACTOR_RESULT_TOPIC=_tool_list \
  ACTOR_HANDLER="$ROOT/examples/employee-mesh/handlers/registry/tool-registry.sh" \
  ACTOR_LMDB_PATH="$LMDB/reg" \
  "$ROOT/actor" >>"$LMDB/reg.log" 2>&1 &
sleep 0.2

echo "=== SQLite MCP ==="
$COMMON ACTOR_ID=sqlite-mcp \
  ACTOR_TOPIC=sql_query ACTOR_RESULT_TOPIC=sql_result \
  ACTOR_HANDLER="$ROOT/examples/employee-mesh/handlers/mcp/tool-bridge.sh" \
  ACTOR_LMDB_PATH="$LMDB/db" \
  MCP_SERVER="python3 $ROOT/examples/employee-mesh/handlers/mcp/mcp-sqlite.py" \
  MCP_TOOL=read_query MCP_ARG=sql \
  BRIDGE_LIB="$ROOT/examples/employee-mesh/handlers/lib" \
  EMPLOYEE_DB="$ROOT/examples/employee-mesh/db/employee.db" \
  "$ROOT/actor" >>"$LMDB/db.log" 2>&1 &
sleep 0.2

echo "=== agent ==="
$COMMON ACTOR_ID=llm-agent \
  ACTOR_TOPIC=user_message,sql_result,_tool_list ACTOR_RESULT_TOPIC=agent_response \
  ACTOR_HANDLER="$ROOT/examples/employee-mesh/handlers/agents/llm-agent" \
  ACTOR_LMDB_PATH="$LMDB/ag" \
  LLM_BASE_URL="$BASE_URL" LLM_MODEL="$MODEL" \
  "$ROOT/actor" >>"$LMDB/ag.log" 2>&1 &
sleep 0.5

echo "=== announce tools ==="
python3 << PYEOF
import msgpack, struct, time, os, ctypes, ctypes.util

payload = msgpack.dumps({
    'type': '_tool_announce', 'actor': 'sqlite-mcp',
    'capabilities': [{'name':'read_query','description':'Execute SQL query',
        'inputSchema':{'type':'object','properties':{'sql':{'type':'string'}},'required':['sql']}}]
})

hdr = bytearray(256)
hdr[0:15] = b'_tool_announce'
ts = int(time.time() * 1000)
for i in range(6): hdr[32+i] = (ts >> (40-i*8)) & 0xFF
for i in range(6,16): hdr[32+i] = os.urandom(1)[0]
hdr[38] = (hdr[38] & 0x0f) | 0x70; hdr[40] = (hdr[40] & 0x3f) | 0x80
hdr[80:87] = b'desktop'
ns = int(time.time() * 1e9); struct.pack_into('>q', hdr, 112, ns)
struct.pack_into('>I', hdr, 138, len(payload))

nng = ctypes.CDLL(ctypes.util.find_library('nng'))
s = ctypes.c_int()
nng.nng_pub0_open(ctypes.byref(s))
nng.nng_dial(s, b'$BUS_PUB', None, 0)
nng.nng_send(s, bytes(hdr)+payload, len(hdr)+len(payload), 0)
nng.nng_close(s)
PYEOF
sleep 0.5

echo "=== verify tools stored ==="
cat "$LMDB/reg/tools/sqlite-mcp.json" 2>/dev/null && echo "✓" || echo "✗"

echo ""
echo "=== sending query: $QUERY ==="
"$ROOT/examples/ping-pong/pub" user_message "$QUERY"

echo ""
echo "=== waiting for response (30s) ==="
sleep 30

echo ""
echo "=== agent log (last 10 lines) ==="
tail -10 "$LMDB/ag.log" 2>/dev/null || echo "(empty)"

echo ""
echo "=== cleanup ==="
pkill -9 mesh-proxy actor 2>/dev/null
echo "done — logs at $LMDB/*.log"
