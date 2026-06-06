#!/usr/bin/env python3
"""tool-registry.py — stores tool announcements, serves discoveries in mpack.
Input: mpack via stdin. Output: mpack + topic prefix via stdout."""
import sys, os, json, msgpack

LMDB = os.environ.get("ACTOR_LMDB_PATH", "/tmp/tool-registry")
TOOLS_DIR = os.path.join(LMDB, "tools")
os.makedirs(TOOLS_DIR, exist_ok=True)

data = msgpack.unpackb(sys.stdin.buffer.read())
msg_type = data.get("type", "")

if msg_type == "_tool_announce":
    actor = data.get("actor", "")
    caps = data.get("capabilities", [])
    if actor and caps:
        with open(os.path.join(TOOLS_DIR, f"{actor}.json"), "w") as f:
            json.dump({"actor": actor, "capabilities": caps}, f)

elif msg_type == "_tool_discover":
    all_caps = []
    for fn in sorted(os.listdir(TOOLS_DIR)):
        if fn.endswith(".json"):
            with open(os.path.join(TOOLS_DIR, fn)) as f:
                all_caps.append(json.load(f))
    result = msgpack.dumps({"type": "_tool_list", "capabilities": all_caps})
    sys.stdout.write("_tool_list\n")
    sys.stdout.buffer.write(result)
