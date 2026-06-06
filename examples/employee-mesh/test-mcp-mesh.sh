#!/bin/bash
# test-mcp-mesh.sh — end-to-end MCP mesh test
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT/examples/employee-mesh"

BUS=tcp://127.0.0.1:555
MODEL="${LLM_MODEL:-granite4.1:8b}"
BASE_URL="${LLM_BASE_URL:-http://localhost:11434}"
LMDB="/tmp/mcp-test"
QUERY="${1:-how many employees are there?}"

echo "=== cleaning ==="
fuser -k 5557/tcp 5556/tcp 2>/dev/null
pkill -9 mesh-proxy actor 2>/dev/null
sleep 0.3
rm -rf "$LMDB"
mkdir -p "$LMDB"/{reg,db,ag}
export BRIDGE_LIB="$PWD/handlers/lib"

echo "=== proxy ==="
PROXY_SUB_BIND=tcp://127.0.0.1:5557 PROXY_PUB_BIND=tcp://127.0.0.1:5556 "$ROOT/mesh-proxy" &
sleep 0.3

echo "=== registry ==="
ACTOR_BUS_SUB="${BUS}6" ACTOR_BUS_PUB="${BUS}7" ACTOR_HEARTBEAT_MS=2000 \
ACTOR_ID=tool-registry ACTOR_TOPIC=_tool_announce,_tool_discover ACTOR_RESULT_TOPIC=_tool_list \
ACTOR_HANDLER="$PWD/handlers/registry/tool-registry.sh" ACTOR_LMDB_PATH="$LMDB/reg" \
"$ROOT/actor" &
sleep 0.3

echo "=== SQLite MCP ==="
ACTOR_BUS_SUB="${BUS}6" ACTOR_BUS_PUB="${BUS}7" ACTOR_HEARTBEAT_MS=2000 \
ACTOR_ID=sqlite-mcp ACTOR_TOPIC=sql_query ACTOR_RESULT_TOPIC=sql_result \
ACTOR_HANDLER="$PWD/handlers/mcp/tool-bridge.sh" ACTOR_LMDB_PATH="$LMDB/db" \
MCP_SERVER="python3 $PWD/handlers/mcp/mcp-sqlite.py" MCP_TOOL=read_query MCP_ARG=sql \
EMPLOYEE_DB="$PWD/db/employee.db" \
"$ROOT/actor" &
sleep 0.3

echo "=== agent ==="
ACTOR_BUS_SUB="${BUS}6" ACTOR_BUS_PUB="${BUS}7" ACTOR_HEARTBEAT_MS=2000 \
ACTOR_ID=llm-agent ACTOR_TOPIC=user_message,sql_result,_tool_list ACTOR_RESULT_TOPIC=agent_response \
ACTOR_HANDLER="$PWD/handlers/agents/llm-agent" ACTOR_LMDB_PATH="$LMDB/ag" \
LLM_BASE_URL="$BASE_URL" LLM_MODEL="$MODEL" \
"$ROOT/actor" &
sleep 0.5

echo "=== announce tools ==="
python3 << 'PYEOF'
import msgpack, struct, time, ctypes, ctypes.util
p = msgpack.dumps({
    'type': '_tool_announce', 'actor': 'sqlite-mcp',
    'capabilities': [{'name': 'read_query', 'description': 'Execute SQL query',
        'inputSchema': {'type': 'object', 'properties': {'sql': {'type': 'string'}}, 'required': ['sql']}}]
})
hdr = bytearray(256)
hdr[0:15] = b'_tool_announce'
hdr[80:87] = b'desktop'
ns = int(time.time() * 1e9)
struct.pack_into('>q', hdr, 112, ns)
struct.pack_into('>I', hdr, 138, len(p))
nng = ctypes.CDLL(ctypes.util.find_library('nng'))
s = ctypes.c_int()
nng.nng_pub0_open(ctypes.byref(s))
nng.nng_dial(s, b'tcp://127.0.0.1:5557', None, 0)
nng.nng_send(s, bytes(hdr)+p, len(hdr)+len(p), 0)
nng.nng_close(s)
print('announced')
PYEOF
sleep 0.5

echo "=== stored? ==="
cat "$LMDB/reg/tools/sqlite-mcp.json" 2>/dev/null && echo "YES" || echo "NO"

echo ""
echo "=== query: $QUERY ==="
"$ROOT/examples/ping-pong/pub" user_message "$QUERY"
echo "waiting 30s..."
sleep 30

echo ""
echo "=== agent log ==="
tail -10 "$LMDB/ag.log" 2>/dev/null || echo "(empty)"

echo ""
echo "=== done ==="
