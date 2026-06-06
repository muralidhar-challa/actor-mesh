#!/bin/sh
# tool-registry.sh — generic capability discovery (shell-only)
# Stores capabilities in $ACTOR_LMDB_PATH/caps/<actor>.json
# Input: mpack via stdin

LMDB="${ACTOR_LMDB_PATH:-/tmp/tool-registry}"
caps_DIR="$LMDB/caps"
mkdir -p "$caps_DIR"

BIN="$(dirname "$0")/../lib/mpack-get"
TMP=$(mktemp)
cat > "$TMP"

TYPE=$("$BIN" type < "$TMP" 2>/dev/null)

case "$TYPE" in
    _tool_announce)
        ACTOR=$("$BIN" actor < "$TMP" 2>/dev/null)
        caps=$("$BIN" -j capabilities < "$TMP" 2>/dev/null)
        [ -n "$ACTOR" ] && [ -n "$caps" ] && echo "{\"actor\":\"$ACTOR\",\"capabilities\":$caps}" > "$caps_DIR/${ACTOR}.json"
        ;;
    _tool_discover)
        printf "_tool_list\n"
        printf '{"type":"_tool_list","capabilities":['
        first=1
        for f in "$caps_DIR"/*.json; do
            [ -f "$f" ] || continue
            [ $first -eq 1 ] || printf ','
            cat "$f"
            first=0
        done
        printf ']}\n'
        ;;
esac

rm -f "$TMP"
