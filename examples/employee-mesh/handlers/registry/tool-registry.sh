#!/bin/sh
# tool-registry.sh — auto-discovers MCP tools on the mesh
# Saves stdin to temp file, extracts mpack fields via mpack-get.
# Stores tools in $ACTOR_LMDB_PATH/tools/<name>.json

LMDB="${ACTOR_LMDB_PATH:-/tmp/tool-registry}"
mkdir -p "$LMDB/tools"

# Save stdin to temp file (binary-safe)
TMP=$(mktemp)
cat > "$TMP"

TYPE=$(../lib/mpack-get type < "$TMP" 2>/dev/null)

case "$TYPE" in
    _tool_announce)
        NAME=$(../lib/mpack-get name < "$TMP" 2>/dev/null)
        if [ -n "$NAME" ]; then
            TOOLS=$(../lib/mpack-get tools < "$TMP" 2>/dev/null)
            echo "{\"name\":\"$NAME\",\"tools\":$TOOLS}" > "$LMDB/tools/${NAME}.json"
        fi
        ;;
    _tools_list)
        printf "_tools_list\n"
        printf '{"type":"_tools_list","tools":['
        FIRST=1
        for f in "$LMDB/tools"/*.json; do
            [ -f "$f" ] || continue
            [ $FIRST -eq 1 ] || printf ','
            cat "$f"
            FIRST=0
        done
        printf ']}\n'
        ;;
esac

rm -f "$TMP"
