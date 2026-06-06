#!/bin/sh
# tool-registry.sh — generic tool discovery
# Stores tools in $ACTOR_LMDB_PATH/tools/<actor>.json
# Gets absolute path to mpack-get based on script location

# BRIDGE_LIB must be set to the lib/ directory path
MPACK_GET="${BRIDGE_LIB:-.}/mpack-get"
LMDB="${ACTOR_LMDB_PATH:-/tmp/tool-registry}"
TOOLS_DIR="$LMDB/tools"
mkdir -p "$TOOLS_DIR"

TMP=$(mktemp)
cat > "$TMP"

TYPE=$("$MPACK_GET" type < "$TMP" 2>/dev/null)

case "$TYPE" in
    _tool_announce)
        ACTOR=$("$MPACK_GET" actor < "$TMP" 2>/dev/null)
        CAPS=$("$MPACK_GET" -j capabilities < "$TMP" 2>/dev/null)
        [ -n "$ACTOR" ] && [ -n "$CAPS" ] && echo "{\"actor\":\"$ACTOR\",\"capabilities\":$CAPS}" > "$TOOLS_DIR/${ACTOR}.json"
        ;;
    _tool_discover)
        printf "_tool_list\n"
        printf '{"type":"_tool_list","capabilities":['
        first=1
        for f in "$TOOLS_DIR"/*.json; do
            [ -f "$f" ] || continue
            [ $first -eq 1 ] || printf ','
            cat "$f"
            first=0
        done
        printf ']}'
        ;;
esac

rm -f "$TMP"
