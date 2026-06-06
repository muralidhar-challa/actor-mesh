#!/usr/bin/env python3
"""mcp-sqlite.py — minimal MCP SQLite server over stdio.
Handles initialize, tools/list, tools/call for read_query."""
import json, sys, sqlite3, os

DB = os.environ.get("EMPLOYEE_DB", os.environ.get("SQLITE_DB", "db/employee.db"))

def respond(id, result=None, error=None):
    r = {"jsonrpc": "2.0", "id": id}
    if result is not None: r["result"] = result
    if error is not None: r["error"] = error
    sys.stdout.write(json.dumps(r) + "\n")
    sys.stdout.flush()

tools = [{
    "name": "read_query",
    "description": "Execute a read-only SQL query against the database",
    "inputSchema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "SQL SELECT query"}},
        "required": ["query"]
    }
}]

for line in sys.stdin:
    try:
        req = json.loads(line.strip())
    except:
        continue

    mid = req.get("id", 0)
    method = req.get("method", "")
    params = req.get("params", {})

    if method == "initialize":
        respond(mid, result={
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": "sqlite", "version": "1.0"},
            "capabilities": {"tools": {}}
        })
    elif method == "tools/list":
        respond(mid, result={"tools": tools})
    elif method == "tools/call":
        name = params.get("name", "")
        args = params.get("arguments", {})
        if name == "read_query":
            query = args.get("query", "")
            try:
                conn = sqlite3.connect(DB)
                cur = conn.execute(query)
                cols = [d[0] for d in cur.description] if cur.description else []
                rows = [dict(zip(cols, row)) for row in cur.fetchall()]
                conn.close()
                text = json.dumps(rows, indent=2) if rows else "[]"
                respond(mid, result={"content": [{"type": "text", "text": text}]})
            except Exception as e:
                respond(mid, result={"content": [{"type": "text", "text": f"Error: {e}"}]})
        else:
            respond(mid, error={"code": -32601, "message": f"Unknown tool: {name}"})
    else:
        respond(mid, error={"code": -32601, "message": f"Unknown method: {method}"})
