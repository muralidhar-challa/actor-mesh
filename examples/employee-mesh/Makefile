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
CFLAGS       = -Wall -Wextra -O2 -std=c11 -I$(RUNTIME) -I$(VENDOR)/mpack/src/mpack
LIBS         = -lzmq -llmdb
HANDLER_LIBS = -lcurl -llmdb

MPACK_SRCS = \
	$(VENDOR)/mpack/src/mpack/mpack-common.c \
	$(VENDOR)/mpack/src/mpack/mpack-platform.c \
	$(VENDOR)/mpack/src/mpack/mpack-reader.c \
	$(VENDOR)/mpack/src/mpack/mpack-writer.c \
	$(VENDOR)/mpack/src/mpack/mpack-expect.c \
	$(VENDOR)/mpack/src/mpack/mpack-node.c

CJSON = $(VENDOR)/cjson/cJSON.c

ACTOR  = $(ROOT)/actor
PROXY  = $(ROOT)/zmq-proxy

# ── Build ──────────────────────────────────────────────────────────────────

.PHONY: all client tui handlers clean

all: client tui handlers

client: client.c $(MPACK_SRCS)
	$(CC) $(CFLAGS) client.c $(MPACK_SRCS) $(LIBS) -o client

tui: tui.c $(MPACK_SRCS)
	$(CC) $(CFLAGS) tui.c $(MPACK_SRCS) $(LIBS) -lpthread -lncursesw -o tui

handlers: handlers/llm-agent handlers/sqlite-tool

handlers/llm-agent: handlers/llm-agent.c $(MPACK_SRCS) $(CJSON)
	$(CC) $(CFLAGS) -I$(VENDOR)/cjson handlers/llm-agent.c $(MPACK_SRCS) $(CJSON) $(HANDLER_LIBS) -o handlers/llm-agent

handlers/sqlite-tool: handlers/sqlite-tool.c $(MPACK_SRCS)
	$(CC) $(CFLAGS) handlers/sqlite-tool.c $(MPACK_SRCS) -lsqlite3 -o handlers/sqlite-tool

clean:
	rm -f client tui handlers/llm-agent handlers/sqlite-tool

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
	ACTOR_RETRY_MAX=3 \
	ACTOR_TTL_NS=300000000000

# ── Run ────────────────────────────────────────────────────────────────────

.PHONY: run stop run-proxy run-sqlite-tool run-llm-agent query chat

run: run-proxy run-sqlite-tool run-llm-agent
	@echo "[mesh] employee mesh running"
	@echo "[mesh] try: make query Q=\"list all engineers\""

stop:
	@pkill -9 zmq-proxy || true
	@pkill -9 actor     || true
	@rm -rf $(LMDB_BASE)
	@echo "[mesh] stopped  (logs were in $(LOGDIR))"

LOGDIR ?= /tmp/actor-mesh/logs

run-proxy:
	@mkdir -p $(LMDB_BASE) $(LOGDIR)
	@PROXY_XSUB_BIND=$(PROXY_XSUB_BIND) \
	PROXY_XPUB_BIND=$(PROXY_XPUB_BIND) \
	$(PROXY) >>$(LOGDIR)/proxy.log 2>&1 &
	@sleep 0.2
	@echo "[proxy] started  (log: $(LOGDIR)/proxy.log)"

run-sqlite-tool:
	@mkdir -p $(LMDB_BASE)/sqlite-tool $(LOGDIR)
	@$(COMMON_ENV) \
	ACTOR_ID=sqlite-tool-1 \
	ACTOR_TOPIC=sql_query \
	ACTOR_RESULT_TOPIC=sql_result \
	ACTOR_HANDLER=$(abspath handlers/sqlite-tool) \
	ACTOR_LMDB_PATH=$(LMDB_BASE)/sqlite-tool \
	EMPLOYEE_DB=$(EMPLOYEE_DB) \
	$(ACTOR) >>$(LOGDIR)/sqlite-tool.log 2>&1 &
	@echo "[actor] sqlite-tool started  (log: $(LOGDIR)/sqlite-tool.log)"

run-llm-agent:
	@mkdir -p $(LMDB_BASE)/llm-agent $(LOGDIR)
	@$(COMMON_ENV) \
	ACTOR_ID=llm-agent-1 \
	ACTOR_TOPIC=user_message,sql_result \
	ACTOR_RESULT_TOPIC=agent_response \
	ACTOR_HANDLER=$(abspath handlers/llm-agent) \
	ACTOR_LMDB_PATH=$(LMDB_BASE)/llm-agent \
	OLLAMA_URL=$(OLLAMA_URL) \
	OLLAMA_MODEL=$(OLLAMA_MODEL) \
	EMPLOYEE_DB=$(EMPLOYEE_DB) \
	$(ACTOR) >>$(LOGDIR)/llm-agent.log 2>&1 &
	@echo "[actor] llm-agent started  (log: $(LOGDIR)/llm-agent.log)"

# ── Query / Chat ───────────────────────────────────────────────────────────

Q ?= list all employees

query:
	@./client "$(Q)"

chat:
	@./tui
