# Makefile — employee mesh demo
#
# ── Build ──────────────────────────────────────────────────────────────────
#   make all              build client + handlers
#   make clean            remove local binaries
#
# ── Run ────────────────────────────────────────────────────────────────────
#   make run              start proxy + all actors
#   make stop             kill all mesh processes + wipe LMDB
#   make query Q="list all engineers"
#
# ── Prerequisites ──────────────────────────────────────────────────────────
#   Build runtime first:  cd ../.. && make
#   Deps: libzmq3-dev liblmdb-dev libcurl-devel sqlite3-devel
#   Ollama running locally

ROOT    = ../..
RUNTIME = $(ROOT)/runtime
VENDOR  = $(ROOT)/vendor

CC           = gcc
CFLAGS       = -Wall -Wextra -O2 -std=c11 -I$(RUNTIME) -I$(VENDOR)/cjson
LIBS         = -lzmq -llmdb
HANDLER_LIBS = -lcurl -llmdb -lsqlite3
CJSON        = $(VENDOR)/cjson/cJSON.c

ACTOR  = $(ROOT)/actor
PROXY  = $(ROOT)/zmq-proxy

# ── Build ──────────────────────────────────────────────────────────────────

.PHONY: all client handlers clean

all: client handlers

client: client.c
	$(CC) $(CFLAGS) client.c $(LIBS) -o client

handlers: handlers/llm-agent handlers/sqlite-tool

handlers/llm-agent: handlers/llm-agent.c $(CJSON)
	$(CC) $(CFLAGS) handlers/llm-agent.c $(CJSON) $(HANDLER_LIBS) -o handlers/llm-agent

handlers/sqlite-tool: handlers/sqlite-tool.c $(CJSON)
	$(CC) $(CFLAGS) handlers/sqlite-tool.c $(CJSON) -lsqlite3 -o handlers/sqlite-tool

clean:
	rm -f client handlers/llm-agent handlers/sqlite-tool

# ── Paths ──────────────────────────────────────────────────────────────────

LMDB_BASE   ?= /tmp/actor-mesh
EMPLOYEE_DB ?= $(abspath db/employee.db)

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

# ── Run ────────────────────────────────────────────────────────────────────

.PHONY: run stop run-proxy run-sqlite-tool run-llm-agent query

run: run-proxy run-sqlite-tool run-llm-agent
	@echo "[mesh] employee mesh running"
	@echo "[mesh] try: make query Q=\"list all engineers\""

stop:
	@pkill -9 -f zmq-proxy   || true
	@pkill -9 -f './actor'   || true
	@rm -rf $(LMDB_BASE)
	@echo "[mesh] stopped"

run-proxy:
	@mkdir -p $(LMDB_BASE)
	PROXY_XSUB_BIND=$(PROXY_XSUB_BIND) \
	PROXY_XPUB_BIND=$(PROXY_XPUB_BIND) \
	$(PROXY) &
	@sleep 0.2
	@echo "[proxy] started"

run-sqlite-tool:
	@mkdir -p $(LMDB_BASE)/sqlite-tool
	$(COMMON_ENV) \
	ACTOR_ID=sqlite-tool-1 \
	ACTOR_TOPIC=sql_query \
	ACTOR_RESULT_TOPIC=sql_result \
	ACTOR_HANDLER=$(abspath handlers/sqlite-tool) \
	ACTOR_LMDB_PATH=$(LMDB_BASE)/sqlite-tool \
	EMPLOYEE_DB=$(EMPLOYEE_DB) \
	$(ACTOR) &
	@echo "[actor] sqlite-tool started"

run-llm-agent:
	@mkdir -p $(LMDB_BASE)/llm-agent
	$(COMMON_ENV) \
	ACTOR_ID=llm-agent-1 \
	ACTOR_TOPIC=user_message,sql_result \
	ACTOR_RESULT_TOPIC=agent_response \
	ACTOR_HANDLER=$(abspath handlers/llm-agent) \
	ACTOR_LMDB_PATH=$(LMDB_BASE)/llm-agent \
	OLLAMA_URL=$(OLLAMA_URL) \
	OLLAMA_MODEL=$(OLLAMA_MODEL) \
	EMPLOYEE_DB=$(EMPLOYEE_DB) \
	$(ACTOR) &
	@echo "[actor] llm-agent started"

# ── Query ──────────────────────────────────────────────────────────────────

Q ?= list all employees

query:
	@./client "$(Q)"
