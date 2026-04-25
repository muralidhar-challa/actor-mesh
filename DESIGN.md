# Actor Mesh — Design Document

## Overview

A minimal distributed actor mesh built on Unix primitives.
No frameworks. No sidecars. No brokers. Just processes.

The entire runtime is ~430 lines of C.
The proxy is 74 lines of C.
A handler is a shell script.

---

## Core Principle

Everything in the mesh is an Actor. The only difference between
actor types is what they subscribe to and what handler they run.

```
SQL Tool      →  actor + sql.sh        subscribes to: sql_query
LLM Agent     →  actor + agent.sh      subscribes to: sql_result
UI Gateway    →  actor + gateway.sh    subscribes to: user_message
Orchestrator  →  actor + orchestrator.sh  subscribes to: heartbeat
Aggregator    →  actor + aggregator.sh    subscribes to: *
```

One binary. Config and handler define behaviour.

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│                  ZMQ Proxy                       │
│   XSUB — collects published tuples              │
│   XPUB — fans out to subscribers               │
│   HWM  — backpressure built in                 │
│   PROXY_COMPRESS=zlib|zstd|none                │
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

Fixed 256-byte binary header. No serialization library.
Cast and read — zero copy.

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

Payload is opaque bytes. Runtime never inspects it.

---

## Handler Contract

Handler is any process. The only contract:

```
read payload bytes from stdin
do work
write result bytes to stdout
exit
```

Runtime forks handler per tuple. Fresh process every time.
No persistent state in handler. No memory leaks. No OOM.

Handler can be:
```
sh + curl + jq   →  zero dependency, preferred
compiled binary  →  Go, Rust, C
any script       →  Python, Node, Ruby
any executable   →  anything that speaks stdio
```

---

## Handler Examples

**LLM agent (shell):**
```sh
#!/bin/sh
payload=$(cat)
request=$(printf '%s' "$payload" | jq -c '{
  model: "claude-haiku-4-5-20251001",
  max_tokens: 1000,
  messages: [{ role: "user", content: .query }]
}')
printf '%s' "$request" | curl -s \
  -X POST https://api.anthropic.com/v1/messages \
  -H "x-api-key: ${ANTHROPIC_API_KEY}" \
  -H "content-type: application/json" \
  -d @- | jq -c '{answer: .content[0].text}'
```

**SQL tool (shell):**
```sh
#!/bin/sh
query=$(cat | jq -r '.query')
psql "$DATABASE_URL" -t -c "$query" | jq -Rs '{rows: .}'
```

**Orchestrator (shell):**
```sh
#!/bin/sh
payload=$(cat)
inbox=$(printf '%s' "$payload" | jq -r '.inbox')
id=$(printf '%s' "$payload"    | jq -r '.id')
if [ "$inbox" -gt 100 ]; then
  kubectl scale deployment/"$id" --replicas=+1
fi
echo "{}"
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
   e. Receive payload frame (zero copy from ZMQ buffer)
   f. Write frame to LMDB inbox (durability)
   g. Fork handler
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

LMDB per actor pod. Two databases:

```
inbox/   {uuidv7} → raw frame    written before fork, cleared after publish
outbox/  {uuidv7} → raw frame    written before publish, cleared after publish
state/   {key}    → bytes        actor runtime state (retry counts etc)
```

On restart — pending inbox tuples are reprocessed.
Pending outbox tuples are republished.
No central coordinator needed.

---

## Memory Model

Zero heap allocation per tuple.

```c
#ifndef ACTOR_MAX_PAYLOAD
#  define ACTOR_MAX_PAYLOAD (1024 * 1024)   // 1MB default
#endif

static uint8_t g_result_buf[ACTOR_MAX_PAYLOAD];
static uint8_t g_frame_buf[ACTOR_MAX_PAYLOAD + sizeof(actor_header_t)];
```

Memory footprint is fixed and known at compile time.
Payloads exceeding cap are dropped with a log message — never malloc.

Override at build time:
```bash
make actor CFLAGS="-O2 -std=c11 -Iruntime -DACTOR_MAX_PAYLOAD=4194304"
```

---

## Heartbeat

Every actor emits a heartbeat tuple periodically:

```json
{"id": "sql-tool-1", "inbox": 12, "outbox": 3}
```

Topic: `heartbeat`
TTL: 3 × heartbeat interval (auto-expires if actor dies)

Any actor subscribed to `heartbeat` can observe the mesh.
Orchestrator actors use this to scale, drain, or restart.

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

All config from environment variables. Runs anywhere POSIX exists.

| Variable | Required | Default | Description |
|---|---|---|---|
| `ACTOR_ID` | ✅ | — | Unique instance id |
| `ACTOR_TOPIC` | ✅ | — | ZMQ subscription topic |
| `ACTOR_RESULT_TOPIC` | ✅ | — | Topic stamped on result tuples |
| `ACTOR_BUS_SUB` | ✅ | — | ZMQ XPUB endpoint e.g. `tcp://bus:5556` |
| `ACTOR_BUS_PUB` | ✅ | — | ZMQ XSUB endpoint e.g. `tcp://bus:5557` |
| `ACTOR_HANDLER` | ✅ | — | Handler command e.g. `/handlers/agent.sh` |
| `ACTOR_LMDB_PATH` | ✅ | — | LMDB directory path |
| `ACTOR_TTL_NS` | ☐ | 0 | Default tuple TTL nanoseconds |
| `ACTOR_HEARTBEAT_MS` | ☐ | 5000 | Heartbeat interval ms |
| `ACTOR_RETRY_MAX` | ☐ | 3 | Max handler retries |

Proxy config:

| Variable | Default | Description |
|---|---|---|
| `PROXY_XSUB_BIND` | `tcp://*:5557` | Actors publish here |
| `PROXY_XPUB_BIND` | `tcp://*:5556` | Actors subscribe here |
| `PROXY_COMPRESS` | `none` | `zlib` or `zstd` if libzmq supports it |

---

## Deployment

**Bare process (simplest):**
```sh
#!/bin/sh
./zmq-proxy &
ACTOR_ID=sql-tool-1 \
ACTOR_TOPIC=sql_query \
ACTOR_RESULT_TOPIC=sql_result \
ACTOR_BUS_SUB=tcp://localhost:5556 \
ACTOR_BUS_PUB=tcp://localhost:5557 \
ACTOR_HANDLER="./handlers/sql.sh" \
ACTOR_LMDB_PATH=/var/actor/sql-tool \
./actor &
wait
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
    value: /handlers/sql.sh
  - name: ACTOR_LMDB_PATH
    value: /var/actor/lmdb
```

K8s is optional. The runtime has no awareness of it.

---

## Build

```bash
# Dependencies
apt install libzmq3-dev liblmdb-dev

# Build
make all

# Custom payload cap
make actor CFLAGS="-O2 -std=c11 -Iruntime -DACTOR_MAX_PAYLOAD=4194304"
```

Outputs:
- `actor` — ~23KB stripped
- `zmq-proxy` — ~15KB stripped

---

## File Layout

```
actor-mesh/
├── Makefile
├── runtime/
│   ├── actor.h          public API — one function: actor_run()
│   ├── actor.c          runtime implementation (~430 lines)
│   ├── actor_tuple.h    256-byte header + helpers
│   ├── actor_uuid.h     uuidv7 single header, no dependency
│   └── main.c           12-line entrypoint
├── proxy/
│   └── proxy.c          ZMQ XPUB/XSUB dumb fanout (74 lines)
└── handlers/
    └── agent.sh         example LLM handler (curl + jq)
```

---

## Lineage

This design independently converges on:

- **Erlang/OTP (1986)** — actor model, let it crash, message passing
- **Unix pipes (1969)** — stdio as universal process interface
- **Linda tuplespace (1986)** — decoupled coordination via shared space
- **Plan 9** — everything is a file, communicate via read/write

The difference: C native, zero framework, same binary edge to cloud.
