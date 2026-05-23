#!/bin/bash
# run.sh — ping-pong smoke test
#
# Ping actor subscribes to "ping", handler outputs "pong" — publishes on "pong"
# Pong actor subscribes to "pong", handler outputs "ping" — publishes on "ping"
# Sending one "ping" triggers infinite loop. Kills after 3s.

set -e

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ACTOR="$ROOT/actor"
PROXY="$ROOT/mesh-proxy"
LMDB="$ROOT/examples/ping-pong/lmdb"

# always rebuild pub
echo "=== building pub ==="
gcc -Wall -O2 -std=c11 -I"$ROOT/runtime" \
    "$ROOT/examples/ping-pong/pub.c" -lnng \
    -o "$ROOT/examples/ping-pong/pub"

chmod +x "$ROOT/examples/ping-pong/ping-handler.sh" \
         "$ROOT/examples/ping-pong/pong-handler.sh"

# cleanup
pkill -9 mesh-proxy 2>/dev/null || true
pkill -9 actor 2>/dev/null || true
sleep 0.5
rm -rf "$LMDB"
mkdir -p "$LMDB"/ping "$LMDB"/pong

echo "=== starting proxy ==="
PROXY_SUB_BIND=tcp://127.0.0.1:5557 PROXY_PUB_BIND=tcp://127.0.0.1:5556 \
    "$PROXY" &
PROXY_PID=$!
sleep 0.5
echo "  proxy pid=$PROXY_PID"

echo "=== starting pong actor ==="
ACTOR_ID=pong-1 \
ACTOR_TOPIC=pong \
ACTOR_RESULT_TOPIC=ping \
ACTOR_BUS_SUB=tcp://127.0.0.1:5556 \
ACTOR_BUS_PUB=tcp://127.0.0.1:5557 \
ACTOR_HANDLER="$ROOT/examples/ping-pong/pong-handler.sh" \
ACTOR_LMDB_PATH="$LMDB/pong" \
ACTOR_HEARTBEAT_MS=0 \
    "$ACTOR" &
PONG_PID=$!
sleep 0.3
echo "  pong actor pid=$PONG_PID"

echo "=== starting ping actor ==="
ACTOR_ID=ping-1 \
ACTOR_TOPIC=ping \
ACTOR_RESULT_TOPIC=pong \
ACTOR_BUS_SUB=tcp://127.0.0.1:5556 \
ACTOR_BUS_PUB=tcp://127.0.0.1:5557 \
ACTOR_HANDLER="$ROOT/examples/ping-pong/ping-handler.sh" \
ACTOR_LMDB_PATH="$LMDB/ping" \
ACTOR_HEARTBEAT_MS=0 \
    "$ACTOR" &
PING_PID=$!
sleep 0.3
echo "  ping actor pid=$PING_PID"

echo ""
echo "=== firing ping → expect infinite ping-pong ==="
"$ROOT/examples/ping-pong/pub" ping hello
sleep 2
echo "  (mesh running for 2s — killing)"

echo ""
echo "=== checking LMDB ==="
for d in ping pong; do
    if [ -f "$LMDB/$d/data.mdb" ]; then
        sz=$(du -h "$LMDB/$d/data.mdb" | cut -f1)
        echo "  $d actor: $sz"
    else
        echo "  $d actor: empty"
    fi
done

kill $PING_PID $PONG_PID $PROXY_PID 2>/dev/null || true
sleep 0.3
pkill -9 mesh-proxy 2>/dev/null || true
pkill -9 actor 2>/dev/null || true
echo "done ✓"
