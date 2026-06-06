#!/bin/sh
# shell-exec — execute shell commands as an actor handler
#
# Handler contract:
#   read payload bytes from stdin
#   do work
#   write result bytes to stdout (with optional topic prefix line)
#   exit
#
# Input (JSON on stdin):
#   { "command": "ls -la /tmp", "timeout": 30, "cwd": "/home" }
#
# Output (stdout):
#   shell_result\n
#   {"exit_code":0,"stdout":"...","stderr":"..."}
#
# If the first output line matches an identifier pattern ([a-zA-Z0-9_]+),
# the runtime uses it as the publish topic instead of ACTOR_RESULT_TOPIC.

set -e

payload=$(cat)

# Parse fields from JSON
cmd=$(echo "$payload" | jq -r '.command // empty')
timeout_sec=$(echo "$payload" | jq -r '.timeout // 30')
cwd=$(echo "$payload" | jq -r '.cwd // empty')

# Validate
if [ -z "$cmd" ]; then
    printf 'shell_result\n{"exit_code":-1,"stdout":"","stderr":"missing \"command\" field","timed_out":false}\n'
    exit 0
fi

# Clamp timeout
case "$timeout_sec" in
    ''|*[!0-9]*) timeout_sec=30 ;;
esac
if [ "$timeout_sec" -gt 600 ]; then timeout_sec=600; fi
if [ "$timeout_sec" -lt 1 ];   then timeout_sec=30;  fi

# Temp files for stdout/stderr capture
tmpout=$(mktemp)
tmperr=$(mktemp)

# Execute with optional working directory
exit_code=0
if [ -n "$cwd" ] && [ -d "$cwd" ]; then
    cd "$cwd" && timeout "$timeout_sec" sh -c "$cmd" >"$tmpout" 2>"$tmperr" || exit_code=$?
else
    timeout "$timeout_sec" sh -c "$cmd" >"$tmpout" 2>"$tmperr" || exit_code=$?
fi

# timeout returns 124 if it killed the process
timed_out=false
if [ "$exit_code" -eq 124 ]; then
    timed_out=true
    exit_code=-1
fi

# JSON-encode stdout/stderr with jq (handles all escaping)
stdout_json=$(jq -Rs . "$tmpout")
stderr_json=$(jq -Rs . "$tmperr")

rm -f "$tmpout" "$tmperr"

# Emit result with topic routing prefix
printf 'shell_result\n{"exit_code":%d,"stdout":%s,"stderr":%s,"timed_out":%s}\n' \
    "$exit_code" "$stdout_json" "$stderr_json" "$timed_out"
