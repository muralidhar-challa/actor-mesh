# Makefile — actor mesh
#
# ── Build ──────────────────────────────────────────────────────────────────
#   make all              build actor + proxy
#   make actor            build actor binary only
#   make proxy            build proxy binary only
#   make clean            remove binaries
#
# ── Run (employee mesh demo) ───────────────────────────────────────────────
#   make run              start proxy + all actors
#   make stop             kill all mesh processes
#   make query Q="list all engineers"   send a test query
#   make logs             show recent thread log
#
# ── Deps ──────────────────────────────────────────────────────────────────
#   apt install libzmq3-dev liblmdb-dev sqlite3
#   Ollama running locally

# ── Compiler ───────────────────────────────────────────────────────────────

CC     = gcc
CFLAGS = -Wall -Wextra -O2 -std=c11 -Iruntime
LIBS   = -lzmq -llmdb

ifdef ACTOR_MAX_PAYLOAD
CFLAGS += -DACTOR_MAX_PAYLOAD=$(ACTOR_MAX_PAYLOAD)
endif

# ── Build targets ──────────────────────────────────────────────────────────

.PHONY: all actor proxy client clean

all: actor proxy client

actor: runtime/main.c runtime/actor.c runtime/actor.h \
       runtime/actor_tuple.h runtime/actor_uuid.h
	$(CC) $(CFLAGS) runtime/main.c runtime/actor.c $(LIBS) -o actor

proxy: proxy/proxy.c
	$(CC) $(CFLAGS) proxy/proxy.c $(LIBS) -o zmq-proxy

client: client.c runtime/actor_tuple.h runtime/actor_uuid.h
	$(CC) $(CFLAGS) client.c $(LIBS) -o client

clean:
	rm -f actor zmq-proxy client

# ── Paths ──────────────────────────────────────────────────────────────────

LMDB_BASE   ?= /tmp/actor-mesh
EMPLOYEE_DB ?= $(PWD)/employee.db
SQLITE_LOG  ?= /tmp/actor-mesh/gateway.db

# ── Bus ────────────────────────────────────────────────────────────────────

PROXY_XSUB_BIND ?= tcp://*:5557
PROXY_XPUB_BIND ?= tcp://*:5556
BUS_SUB         ?= tcp://localhost:5556
BUS_PUB         ?= tcp://localhost:5557

# ── Ollama ─────────────────────────────────────────────────────────────────

OLLAMA_URL   ?= http://localhost:11434
OLLAMA_MODEL ?= gemma4:e2b

# ── Common actor env ───────────────────────────────────────────────────────

COMMON_ENV = \
	ACTOR_BUS_SUB=$(BUS_SUB) \
	ACTOR_BUS_PUB=$(BUS_PUB) \
	ACTOR_HEARTBEAT_MS=5000 \
	ACTOR_RETRY_MAX=3

# ── Run targets ────────────────────────────────────────────────────────────

.PHONY: run stop run-proxy run-gateway run-sqlite-tool run-llm-agent query logs

run: run-proxy run-sqlite-tool run-llm-agent run-gateway
	@echo "[mesh] employee mesh running"
	@echo "[mesh] try: make query Q=\"list all engineers\""

stop:
	@pkill -9 -f zmq-proxy || true
	@pkill -9 -f './actor'  || true
	@echo "[mesh] stopped"

# ── Proxy ──────────────────────────────────────────────────────────────────

run-proxy:
	@mkdir -p $(LMDB_BASE)
	PROXY_XSUB_BIND=$(PROXY_XSUB_BIND) \
	PROXY_XPUB_BIND=$(PROXY_XPUB_BIND) \
	./zmq-proxy &
	@sleep 0.2
	@echo "[proxy] started"

# ── Gateway ────────────────────────────────────────────────────────────────

run-gateway:
	@mkdir -p $(LMDB_BASE)/gateway
	$(COMMON_ENV) \
	ACTOR_ID=gateway-1 \
	ACTOR_TOPIC=user_message \
	ACTOR_RESULT_TOPIC=sql_query \
	ACTOR_HANDLER="sh handlers/gateway.sh" \
	ACTOR_LMDB_PATH=$(LMDB_BASE)/gateway \
	SQLITE_LOG=$(SQLITE_LOG) \
	./actor &
	@echo "[actor] gateway started"

# ── SQLite Tool ────────────────────────────────────────────────────────────

run-sqlite-tool:
	@mkdir -p $(LMDB_BASE)/sqlite-tool
	$(COMMON_ENV) \
	ACTOR_ID=sqlite-tool-1 \
	ACTOR_TOPIC=sql_query \
	ACTOR_RESULT_TOPIC=sql_result \
	ACTOR_HANDLER="sh handlers/sqlite-tool.sh" \
	ACTOR_LMDB_PATH=$(LMDB_BASE)/sqlite-tool \
	EMPLOYEE_DB=$(EMPLOYEE_DB) \
	./actor &
	@echo "[actor] sqlite-tool started"

# ── LLM Agent ─────────────────────────────────────────────────────────────

run-llm-agent:
	@mkdir -p $(LMDB_BASE)/llm-agent
	$(COMMON_ENV) \
	ACTOR_ID=llm-agent-1 \
	ACTOR_TOPIC=user_message \
	ACTOR_RESULT_TOPIC=agent_response \
	ACTOR_HANDLER="sh handlers/llm-agent.sh" \
	ACTOR_LMDB_PATH=$(LMDB_BASE)/llm-agent \
	OLLAMA_URL=$(OLLAMA_URL) \
	OLLAMA_MODEL=$(OLLAMA_MODEL) \
	EMPLOYEE_DB=$(EMPLOYEE_DB) \
	./actor &
	@echo "[actor] llm-agent started"

# ── Test query ─────────────────────────────────────────────────────────────

Q ?= list all employees

query:
	@echo "[query] sending: $(Q)"
	@echo '{"type":"user_message","query":"$(Q)","session_id":"test-001"}' | \
		curl -s -X POST http://localhost:8080 \
		-H "content-type: application/json" \
		-d @- | jq .

# ── Logs ───────────────────────────────────────────────────────────────────

logs:
	@sqlite3 $(SQLITE_LOG) \
		"SELECT datetime(emitted_at/1000000000,'unixepoch'), topic, origin \
		 FROM thread_log ORDER BY emitted_at DESC LIMIT 20" 2>/dev/null \
		|| echo "[logs] no thread log yet"
