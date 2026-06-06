#!/bin/bash
# clean-kill.sh — kill everything, free ports, clean LMDB
fuser -k 5557/tcp 5556/tcp 2>/dev/null
pkill -9 -f mesh-proxy 2>/dev/null
pkill -9 -f '\./actor' 2>/dev/null
pkill -9 -f 'actor-mesh-desktop' 2>/dev/null
sleep 0.3
rm -rf /tmp/mcp-test /tmp/actor-mesh /tmp/tauri-test /tmp/cap-registry /tmp/tool-registry
echo "all clean"
