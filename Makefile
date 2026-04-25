# Makefile — actor mesh
#
# ── Build ──────────────────────────────────────────────────────────────────
#   make all              build actor + proxy
#   make actor            build actor binary only
#   make proxy            build proxy binary only
#   make clean            remove binaries
#
# ── Run ───────────────────────────────────────────────────────────────────
#   make run              start proxy + all actors
#   make stop             kill all mesh processes
#   make run-proxy        start proxy only
#   make run-sql-tool     start sql tool actor
#   make run-agent        start llm agent actor
#   make run-orchestrator start orchestrator actor
#
# ── Deps ──────────────────────────────────────────────────────────────────
#   apt install libzmq3-dev liblmdb-dev   (build)
#   apt install libzmq5 liblmdb0          (runtime)

# ── Compiler ───────────────────────────────────────────────────────────────

CC     = gcc
CFLAGS = -Wall -Wextra -O2 -std=c11 -Iruntime
LIBS   = -lzmq -llmdb

# Override payload cap at build time:
#   make actor ACTOR_MAX_PAYLOAD=4194304
ifdef ACTOR_MAX_PAYLOAD
CFLAGS += -DACTOR_MAX_PAYLOAD=$(ACTOR_MAX_PAYLOAD)
endif

# ── Bus endpoints ──────────────────────────────────────────────────────────

PROXY_XSUB_BIND ?= tcp://*:5557
PROXY_XPUB_BIND ?= tcp://*:5556
PROXY_COMPRESS  ?= none

BUS_SUB ?= tcp://localhost:5556
BUS_PUB ?= tcp://localhost:5557

# ── LMDB paths ─────────────────────────────────────────────────────────────

LMDB_BASE ?= /tmp/actor-mesh

# ── Heartbeat + retry ──────────────────────────────────────────────────────

HEARTBEAT_MS ?= 5000
RETRY_MAX    ?= 3

# ── Build targets ──────────────────────────────────────────────────────────

.PHONY: all actor proxy clean

all: actor proxy

actor: runtime/main.c runtime/actor.c runtime/actor.h \
       runtime/actor_tuple.h runtime/actor_uuid.h
	$(CC) $(CFLAGS) runtime/main.c runtime/actor.c $(LIBS) -o actor

proxy: proxy/proxy.c
	$(CC) $(CFLAGS) proxy/proxy.c $(LIBS) -o zmq-proxy

clean:
	rm -f actor zmq-proxy

# ── Run targets ────────────────────────────────────────────────────────────

.PHONY: run stop run-proxy run-sql-tool run-agent run-orchestrator

run: run-proxy run-sql-tool run-agent run-orchestrator
	@echo "[mesh] all actors started"

stop:
	@pkill zmq-proxy || true
	@pkill -f './actor' || true
	@echo "[mesh] stopped"

# ── Proxy ──────────────────────────────────────────────────────────────────

run-proxy:
	@mkdir -p $(LMDB_BASE)
	PROXY_XSUB_BIND=$(PROXY_XSUB_BIND) \
	PROXY_XPUB_BIND=$(PROXY_XPUB_BIND) \
	PROXY_COMPRESS=$(PROXY_COMPRESS) \
	./zmq-proxy &
	@echo "[proxy] started xsub=$(PROXY_XSUB_BIND) xpub=$(PROXY_XPUB_BIND)"

# ── SQL Tool ───────────────────────────────────────────────────────────────

run-sql-tool:
	@mkdir -p $(LMDB_BASE)/sql-tool
	ACTOR_ID=sql-tool-1 \
	ACTOR_TOPIC=sql_query \
	ACTOR_RESULT_TOPIC=sql_result \
	ACTOR_BUS_SUB=$(BUS_SUB) \
	ACTOR_BUS_PUB=$(BUS_PUB) \
	ACTOR_HANDLER="./handlers/sql.sh" \
	ACTOR_LMDB_PATH=$(LMDB_BASE)/sql-tool \
	ACTOR_HEARTBEAT_MS=$(HEARTBEAT_MS) \
	ACTOR_RETRY_MAX=$(RETRY_MAX) \
	./actor &
	@echo "[actor] sql-tool started"

# ── LLM Agent ─────────────────────────────────────────────────────────────

run-agent:
	@mkdir -p $(LMDB_BASE)/agent
	ACTOR_ID=agent-1 \
	ACTOR_TOPIC=sql_result \
	ACTOR_RESULT_TOPIC=agent_response \
	ACTOR_BUS_SUB=$(BUS_SUB) \
	ACTOR_BUS_PUB=$(BUS_PUB) \
	ACTOR_HANDLER="./handlers/agent.sh" \
	ACTOR_LMDB_PATH=$(LMDB_BASE)/agent \
	ACTOR_HEARTBEAT_MS=$(HEARTBEAT_MS) \
	ACTOR_RETRY_MAX=$(RETRY_MAX) \
	./actor &
	@echo "[actor] agent started"

# ── Orchestrator ───────────────────────────────────────────────────────────

run-orchestrator:
	@mkdir -p $(LMDB_BASE)/orchestrator
	ACTOR_ID=orchestrator-1 \
	ACTOR_TOPIC=heartbeat \
	ACTOR_RESULT_TOPIC=control \
	ACTOR_BUS_SUB=$(BUS_SUB) \
	ACTOR_BUS_PUB=$(BUS_PUB) \
	ACTOR_HANDLER="./handlers/orchestrator.sh" \
	ACTOR_LMDB_PATH=$(LMDB_BASE)/orchestrator \
	ACTOR_HEARTBEAT_MS=$(HEARTBEAT_MS) \
	ACTOR_RETRY_MAX=$(RETRY_MAX) \
	./actor &
	@echo "[actor] orchestrator started"
