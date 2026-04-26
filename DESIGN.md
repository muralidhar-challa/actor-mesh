# Actor Mesh — Design Document

## Overview

A minimal distributed actor mesh built on Unix primitives.
No frameworks. No sidecars. No brokers. Just processes.

The runtime is ~430 lines of C. The proxy is 74 lines of C.
A handler is any process that speaks stdio.

---

## Core Principle

Everything in the mesh is an Actor. The only difference between actor types
is what they subscribe to and what handler they run.

```
SQL Tool    →  actor + handlers/sqlite-tool   subscribes to: sql_query
LLM Agent   →  actor + handlers/llm-agent     subscribes to: user_message, sql_result
Observer    →  actor + <any handler>           subscribes to: heartbeat
```

One binary. Config and handler define behaviour.

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│                  ZMQ Proxy                       │
│   XSUB — collects published tuples               │
│   XPUB — fans out to subscribers                 │
│   HWM  — backpressure built in                   │
└───────────────────┬──────────────────────────────┘
                    │  tcp://
          ┌─────────┼─────────┐
          ↓         ↓         ↓
       Actor A   Actor B   Actor C
       (any)     (any)     (any)
```

Each actor:
- Connects to proxy via ZMQ SUB (receive) and PUB (send)
- Forks a handler process per tuple
- Writes payload to handler stdin
- Reads result from handler stdout
- Publishes result back to bus
- Maintains local LMDB for durability

---

## Tuple Header

Fixed 256-byte binary header. No serialisation library. Cast and read — zero copy.

```c
typedef struct __attribute__((packed)) {
    char     topic[32];           // ZMQ subscription filter
    uint8_t  id[16];              // uuidv7 binary
    uint8_t  correlation_id[16];  // trace chain end to end
    uint8_t  causation_id[16];    // direct parent tuple id
    char     origin[32];          // actor id who emitted
    int64_t  emitted_at;          // unix nanoseconds
    int64_t  ttl;                 // nanoseconds, 0 = no expiry
    int32_t  attempt;             // retry count, 0 = first
    uint32_t payload_len;         // bytes following this header
    uint8_t  _reserved[120];      // pad to 256 bytes
} actor_header_t;
```

Wire format:
```
[ actor_header_t 256 bytes ][ payload bytes ]
```

Payload is opaque bytes. The runtime never inspects it.

---

## Handler Contract

Handler is any process. The only contract:

```
read payload bytes from stdin
do work
write result bytes to stdout
exit
```

Runtime forks one handler per tuple. Fresh process every time.
No persistent state in handler. No memory leaks. No OOM.

Result topic routing: if stdout starts with a line of the form `topic_name\n`,
the runtime strips it and uses that as the publish topic, overriding
`ACTOR_RESULT_TOPIC`. This lets a single handler emit different tuple types.

```
sql_query\n{"type":"sql_query","sql":"SELECT ..."}
```

Header env vars available to every handler:

| Variable | Value |
|---|---|
| `ACTOR_TUPLE_ID` | hex id of incoming tuple |
| `ACTOR_CORRELATION_ID` | hex correlation id (same across chain) |
| `ACTOR_CAUSATION_ID` | hex id of direct parent tuple |
| `ACTOR_TUPLE_ORIGIN` | origin actor id |
| `ACTOR_ATTEMPT` | retry count |

---

## Multi-topic Subscription

`ACTOR_TOPIC` accepts a comma-separated list:

```sh
ACTOR_TOPIC=user_message,sql_result ./actor
```

One actor instance receives tuples on all listed topics.

---

## Handler Examples

**Shell (zero deps):**
```sh
#!/bin/sh
payload=$(cat)
curl -s https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -d "$(echo "$payload" | jq -c '{model:"claude-haiku-4-5-20251001",max_tokens:1000,messages:[{role:"user",content:.query}]}')" \
  | jq -c '{answer:.content[0].text}'
```

**C binary (zero alloc):**
```c
static char g_in[256*1024];
int main(void) {
    size_t n = fread(g_in, 1, sizeof(g_in)-1, stdin);
    g_in[n] = '\0';
    // ... process ...
    fputs(result, stdout);
}
```

---

## Runtime Lifecycle

```
1. Read config from environment
2. Connect ZMQ SUB + PUB to proxy
3. Open LMDB
4. Enter poll loop:
   a. Emit heartbeat every ACTOR_HEARTBEAT_MS
   b. Poll ZMQ (100ms timeout)
   c. Receive header frame
   d. Check TTL — drop if expired
   e. Receive payload frame
   f. Write frame to LMDB inbox (durability)
   g. Set header env vars, fork handler
   h. Write payload to handler stdin
   i. Read result into static buffer (capped at ACTOR_MAX_PAYLOAD)
   j. Build result header (carry correlation chain)
   k. Write result frame to LMDB outbox
   l. Publish result to ZMQ bus
   m. Clear LMDB inbox + outbox
5. On SIGTERM — drain and exit
```

---

## Durability

LMDB per actor pod. Three named databases:

```
inbox/   {uuidv7} → raw frame    written before fork, cleared after publish
outbox/  {uuidv7} → raw frame    written before publish, cleared after publish
state/   {key}    → bytes        handler-managed state (e.g. conversation history)
```

On restart — pending inbox tuples are reprocessed.
Pending outbox tuples are republished.
No central coordinator needed.

Handlers can use the same LMDB (via `ACTOR_LMDB_PATH`) to persist state
keyed by `ACTOR_CORRELATION_ID`, enabling multi-turn conversations across
separate invocations.

---

## Memory Model

Zero heap allocation per tuple in the runtime.

```c
static uint8_t g_result_buf[ACTOR_MAX_PAYLOAD];
static uint8_t g_frame_buf[ACTOR_MAX_PAYLOAD + sizeof(actor_header_t)];
```

Memory footprint is fixed and known at compile time.
Payloads exceeding cap are dropped with a log message — never malloc.

Override at build time:
```bash
make ACTOR_MAX_PAYLOAD=4194304
```

Handlers follow the same discipline: static buffers only, no malloc/realloc.
Data flows as pointers from stdin buffer to stdout — no heap involvement.

---

## Heartbeat

Every actor emits a heartbeat tuple periodically:

```json
{"id": "sqlite-tool-1", "inbox": 0, "outbox": 0}
```

Topic: `heartbeat`
TTL: 3 × heartbeat interval

Any actor subscribed to `heartbeat` can observe the mesh state.

---

## Retry Policy

On handler failure (non-zero exit):
```
attempt 1  →  retry after 100ms
attempt 2  →  retry after 200ms
attempt 3  →  retry after 400ms
max retry  →  drop tuple, log error
```

Controlled by `ACTOR_RETRY_MAX` (default 3).
Payload cap exceeded → drop immediately, no retry.

---

## Configuration

| Variable | Required | Default | Description |
|---|---|---|---|
| `ACTOR_ID` | ✅ | — | Unique instance id |
| `ACTOR_TOPIC` | ✅ | — | ZMQ subscription topic (comma-sep for multi) |
| `ACTOR_RESULT_TOPIC` | ✅ | — | Default topic stamped on result tuples |
| `ACTOR_BUS_SUB` | ✅ | — | ZMQ XPUB endpoint e.g. `tcp://bus:5556` |
| `ACTOR_BUS_PUB` | ✅ | — | ZMQ XSUB endpoint e.g. `tcp://bus:5557` |
| `ACTOR_HANDLER` | ✅ | — | Handler binary path |
| `ACTOR_LMDB_PATH` | ✅ | — | LMDB directory path |
| `ACTOR_TTL_NS` | ☐ | 0 | Default tuple TTL nanoseconds |
| `ACTOR_HEARTBEAT_MS` | ☐ | 5000 | Heartbeat interval ms |
| `ACTOR_RETRY_MAX` | ☐ | 3 | Max handler retries |

Proxy:

| Variable | Default | Description |
|---|---|---|
| `PROXY_XSUB_BIND` | `tcp://*:5557` | Actors publish here |
| `PROXY_XPUB_BIND` | `tcp://*:5556` | Actors subscribe here |

---

## File Layout

```
actor-mesh/
├── Makefile                    build actor + zmq-proxy
├── DESIGN.md
├── runtime/
│   ├── actor.h                 public API — actor_run()
│   ├── actor.c                 runtime (~430 lines, zero malloc)
│   ├── actor_tuple.h           256-byte header + helpers
│   ├── actor_uuid.h            uuidv7 single-header, no deps
│   └── main.c                  12-line entrypoint
├── proxy/
│   └── proxy.c                 ZMQ XPUB/XSUB fanout (74 lines)
├── examples/
│   └── employee-mesh/          HR Q&A demo (ReAct loop over SQLite)
│       ├── Makefile            build + run + query
│       ├── client.c            CLI client — sends query, prints answer
│       ├── handlers/
│       │   ├── llm-agent.c     ReAct agent — Ollama + LMDB state
│       │   └── sqlite-tool.c   SQL executor — reads EMPLOYEE_DB
│       └── db/
│           └── employee.db     sample employee database
└── vendor/
    ├── cjson/                  cJSON v1.7.18
    └── mu_json/                mu_json — zero-alloc JSON parser
```

---

## Build

```bash
# runtime
make

# employee-mesh demo
cd examples/employee-mesh
make
make run
make query Q="who manages the Development department?"
make stop
```

---

## Deployment

**Bare process:**
```sh
./zmq-proxy &
ACTOR_ID=sqlite-tool-1 \
ACTOR_TOPIC=sql_query \
ACTOR_RESULT_TOPIC=sql_result \
ACTOR_BUS_SUB=tcp://localhost:5556 \
ACTOR_BUS_PUB=tcp://localhost:5557 \
ACTOR_HANDLER=./handlers/sqlite-tool \
ACTOR_LMDB_PATH=/var/actor/sqlite-tool \
./actor &
```

**K8s (optional):**
```yaml
env:
  - name: ACTOR_ID
    valueFrom: { fieldRef: { fieldPath: metadata.name } }
  - name: ACTOR_TOPIC
    value: sql_query
  - name: ACTOR_RESULT_TOPIC
    value: sql_result
  - name: ACTOR_BUS_SUB
    value: tcp://zmq-proxy:5556
  - name: ACTOR_BUS_PUB
    value: tcp://zmq-proxy:5557
  - name: ACTOR_HANDLER
    value: /handlers/sqlite-tool
  - name: ACTOR_LMDB_PATH
    value: /var/actor/lmdb
```

K8s is optional. The runtime has no awareness of it.

---

## Lineage

This design independently converges on:

- **Erlang/OTP (1986)** — actor model, let it crash, message passing
- **Unix pipes (1969)** — stdio as universal process interface
- **Linda tuplespace (1986)** — decoupled coordination via shared space
- **Plan 9** — everything is a file, communicate via read/write

The difference: C native, zero framework, same binary edge to cloud.
