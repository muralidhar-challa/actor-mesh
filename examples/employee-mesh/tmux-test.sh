#!/bin/bash
# tmux-test.sh — run MCP mesh test in tmux session
# Creates panes for each component so you can see everything live.

SESSION="mcp-test"
ROOT="$HOME/Projects/mesh-actors/examples/employee-mesh"

tmux kill-session -t "$SESSION" 2>/dev/null
tmux new-session -d -s "$SESSION" -c "$ROOT"

# Pane 0: proxy
tmux send-keys "cd $ROOT && pkill -9 mesh-proxy actor 2>/dev/null; sleep 0.2; rm -rf /tmp/mcp-test; mkdir -p /tmp/mcp-test/{reg,db,ag}; PROXY_SUB_BIND=tcp://127.0.0.1:5557 PROXY_PUB_BIND=tcp://127.0.0.1:5556 ../../mesh-proxy" Enter

# Pane 1: registry  
tmux split-window -v
tmux send-keys "sleep 0.5 && cd $ROOT && ACTOR_BUS_SUB=tcp://127.0.0.1:5556 ACTOR_BUS_PUB=tcp://127.0.0.1:5557 ACTOR_HEARTBEAT_MS=2000 ACTOR_ID=tool-registry ACTOR_TOPIC=_tool_announce,_tool_discover ACTOR_RESULT_TOPIC=_tool_list ACTOR_HANDLER='$PWD/handlers/registry/tool-registry.sh' ACTOR_LMDB_PATH=/tmp/mcp-test/reg ../../actor" Enter

# Pane 2: SQLite MCP
tmux split-window -v
tmux send-keys "sleep 0.7 && cd $ROOT && ACTOR_BUS_SUB=tcp://127.0.0.1:5556 ACTOR_BUS_PUB=tcp://127.0.0.1:5557 ACTOR_HEARTBEAT_MS=2000 ACTOR_ID=sqlite-mcp ACTOR_TOPIC=sql_query ACTOR_RESULT_TOPIC=sql_result ACTOR_HANDLER='$PWD/handlers/mcp/tool-bridge.sh' ACTOR_LMDB_PATH=/tmp/mcp-test/db MCP_SERVER='python3 $PWD/handlers/mcp/mcp-sqlite.py' MCP_TOOL=read_query MCP_ARG=sql BRIDGE_LIB='$PWD/handlers/lib' EMPLOYEE_DB='$PWD/db/employee.db' ../../actor" Enter

# Pane 3: agent
tmux split-window -v
tmux send-keys "sleep 0.9 && cd $ROOT && ACTOR_BUS_SUB=tcp://127.0.0.1:5556 ACTOR_BUS_PUB=tcp://127.0.0.1:5557 ACTOR_HEARTBEAT_MS=2000 ACTOR_ID=llm-agent ACTOR_TOPIC=user_message,sql_result,_tool_list ACTOR_RESULT_TOPIC=agent_response ACTOR_HANDLER='$PWD/handlers/agents/llm-agent' ACTOR_LMDB_PATH=/tmp/mcp-test/ag LLM_BASE_URL=http://localhost:11434 LLM_MODEL=granite4.1:8b ../../actor" Enter

# Pane 4: test commands
tmux split-window -v
tmux send-keys "sleep 1.2 && cd $ROOT && echo '=== announce tools ===' && python3 -c \"
import msgpack,struct,time,os,ctypes,ctypes.util
p=msgpack.dumps({'type':'_tool_announce','actor':'sqlite-mcp','capabilities':[{'name':'read_query','description':'Execute SQL','inputSchema':{'type':'object','properties':{'sql':{'type':'string'}},'required':['sql']}}]})
h=bytearray(256);h[0:15]=b'_tool_announce'
ts=int(time.time()*1000)
for i in range(6):h[32+i]=(ts>>(40-i*8))&0xFF
for i in range(6,16):h[32+i]=os.urandom(1)[0]
h[38]=(h[38]&0x0f)|0x70;h[40]=(h[40]&0x3f)|0x80
h[80:87]=b'desktop'
ns=int(time.time()*1e9);struct.pack_into('>q',h,112,ns)
struct.pack_into('>I',h,138,len(p))
n=ctypes.CDLL(ctypes.util.find_library('nng'))
s=ctypes.c_int();n.nng_pub0_open(ctypes.byref(s))
n.nng_dial(s,b'tcp://127.0.0.1:5557',None,0)
n.nng_send(s,bytes(h)+p,len(h)+len(p),0);n.nng_close(s)
print('done')
\" && sleep 0.5 && cat /tmp/mcp-test/reg/tools/sqlite-mcp.json && echo && echo '=== sending query ===' && ../../examples/ping-pong/pub user_message 'how many employees?' && echo 'waiting...'" Enter

tmux select-layout even-vertical
tmux attach -t "$SESSION"
