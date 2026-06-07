Subject: Actor Mesh — lightweight distributed agent runtime

Hi,

Attached is a fossil repo containing Actor Mesh — a minimal distributed agent runtime. No frameworks, no sidecars, no brokers. ~500 lines of C.

**Quick start (3 commands):**

```
fossil clone actor-mesh.fossil actor-mesh && cd actor-mesh && fossil open ../actor-mesh.fossil
sudo dnf install nng-devel lmdb-devel gcc make python3-msgpack
make && cd examples/employee-mesh && make
```

**Run the test suite (no API key needed):**

```
gcc -Wall -O2 -std=c11 tests/test-mesh.c -lnng -o bin/test-mesh
bin/test-mesh    # 10/10 passing
```

**What this is:**

A pub/sub mesh where each agent is a process that reads stdin and writes stdout. Handlers can be shell scripts, Python, compiled binaries — anything. MCP servers work as handlers directly. Agents auto-discover tools via a topic-based registry.

Two docs at the root explain everything: `START.md` (getting started) and `AGENTS.md` (multi-agent patterns + Claude Platform comparison).

Let me know if you hit any issues.

—
