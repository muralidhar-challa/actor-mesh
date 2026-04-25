#!/bin/sh
# handlers/gateway.sh
#
# Receives: user_message payload  { "query": "...", "session_id": "..." }
# Emits:    sql_query payload     { "query": "...", "natural": "..." }
#
# Also receives agent_response and delivers to user via stdout
# Gateway is the only actor that touches the external world
#
# Dependencies: jq

set -e

payload=$(cat)

type=$(printf '%s' "$payload" | jq -r '.type // "user_message"')

case "$type" in

    user_message)
        # translate user natural language query into sql_query tuple
        natural=$(printf '%s' "$payload" | jq -r '.query')
        session=$(printf '%s' "$payload" | jq -r '.session_id')

        # emit sql_query — runtime carries correlation_id forward
        printf '%s' "$payload" | jq -c '{
            type:       "sql_query",
            natural:    .query,
            session_id: .session_id,
            query:      "SELECT * FROM employees"
        }'
        ;;

    agent_response)
        # response arrived — deliver back to caller
        # in a real gateway this would push to WebSocket
        # here we just emit it so socat can forward it
        printf '%s' "$payload" | jq -c '{
            type:    "delivered",
            answer:  .answer,
            session: .session_id
        }'
        ;;

    *)
        printf '{"type":"error","message":"unknown type"}\n'
        ;;
esac
