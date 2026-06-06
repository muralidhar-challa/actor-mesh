#!/bin/sh
# cap-registry.sh — generic capability discovery (shell-only)
# Stores capabilities in $ACTOR_LMDB_PATH/caps/<actor>.json
# Input: mpack via stdin

LMDB="${ACTOR_LMDB_PATH:-/tmp/cap-registry}"
CAPS_DIR="$LMDB/caps"
mkdir -p "$CAPS_DIR"

BIN="$(dirname "$0")/../lib/mpack-get"
TMP=$(mktemp)
cat > "$TMP"

TYPE=$("$BIN" type < "$TMP" 2>/dev/null)

case "$TYPE" in
    _cap_announce)
        ACTOR=$("$BIN" actor < "$TMP" 2>/dev/null)
        CAPS=$("$BIN" -j capabilities < "$TMP" 2>/dev/null)
        [ -n "$ACTOR" ] && [ -n "$CAPS" ] && echo "{\"actor\":\"$ACTOR\",\"capabilities\":$CAPS}" > "$CAPS_DIR/${ACTOR}.json"
        ;;
    _cap_discover)
        printf "_cap_list\n"
        printf '{"type":"_cap_list","capabilities":['
        first=1
        for f in "$CAPS_DIR"/*.json; do
            [ -f "$f" ] || continue
            [ $first -eq 1 ] || printf ','
            cat "$f"
            first=0
        done
        printf ']}\n'
        ;;
esac

rm -f "$TMP"
