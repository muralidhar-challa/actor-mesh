# Makefile — employee mesh
#
# ── What this does ─────────────────────────────────────────────────────────
#   A simple employee Q&A mesh:
#     gateway     →  receives user questions
#     sqlite-tool →  queries employee SQLite DB
#     llm-agent   →  calls Ollama to answer using query results
#
# ── Quick start ────────────────────────────────────────────────────────────
#   make build        build actor + proxy binaries
#   make run          start full mesh
#   make query Q="list all engineers"   send a test query
#   make stop         stop all mesh processes
#
# ── Dependencies ───────────────────────────────────────────────────────────
#   apt install libzmq3-dev liblmdb-dev sqlite3
#   Ollama running locally: https://ollama.ai
#   curl, jq — available on alpine/ubuntu

# ── Paths ──────────────────────────────────────────────────────────────────

ACTOR_BIN   ?= ../actor-mesh/actor
PROXY_BIN   ?= ../actor-mesh/zmq-proxy
LMDB_BASE   ?= /tmp/employee-mesh/lmdb
EMPLOYEE_DB ?= $(dir $(abspath $(lastword $(MAKEFILE_LIST))))db/employees.db
SQLITE_LOG  ?= /tmp/employee-mesh/gateway.db

# ── Bus ────────────────────────────────────────────────────────────────────

PROXY_XSUB_BIND ?= tcp://*:5557
PROXY_XPUB_BIND ?= tcp://*:5556
BUS_SUB         ?= tcp://localhost:5556
BUS_PUB         ?= tcp://localhost:5557

# ── Ollama ─────────────────────────────────────────────────────────────────

OLLAMA_URL   ?= http://localhost:11434
OLLAMA_MODEL ?= llama3.2

# ── Common actor env ───────────────────────────────────────────────────────

COMMON_ENV = \
	ACTOR_BUS_SUB=$(BUS_SUB) \
	ACTOR_BUS_PUB=$(BUS_PUB) \
	ACTOR_HEARTBEAT_MS=5000 \
	ACTOR_RETRY_MAX=3

# ── Targets ────────────────────────────────────────────────────────────────

.PHONY: all build run stop \
        run-proxy run-gateway run-sqlite-tool run-llm-agent \
        query logs clean

all: build

# ── Build ──────────────────────────────────────────────────────────────────

build:
	@$(MAKE) -C ../actor-mesh all
	@echo "[build] actor and proxy ready"

# ── Run ────────────────────────────────────────────────────────────────────

run: run-proxy run-sqlite-tool run-llm-agent run-gateway
	@echo "[mesh] employee mesh running"
	@echo "[mesh] try: make query Q=\"list all engineers\""

stop:
	@pkill -f zmq-proxy || true
	@pkill -f './actor'  || true
	@echo "[mesh] stopped"

# ── Proxy ──────────────────────────────────────────────────────────────────

run-proxy:
	@mkdir -p $(LMDB_BASE)
	PROXY_XSUB_BIND=$(PROXY_XSUB_BIND) \
	PROXY_XPUB_BIND=$(PROXY_XPUB_BIND) \
	$(PROXY_BIN) &
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
	$(ACTOR_BIN) &
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
	$(ACTOR_BIN) &
	@echo "[actor] sqlite-tool started"

# ── LLM Agent ─────────────────────────────────────────────────────────────

run-llm-agent:
	@mkdir -p $(LMDB_BASE)/llm-agent
	$(COMMON_ENV) \
	ACTOR_ID=llm-agent-1 \
	ACTOR_TOPIC=sql_result \
	ACTOR_RESULT_TOPIC=agent_response \
	ACTOR_HANDLER="sh handlers/llm-agent.sh" \
	ACTOR_LMDB_PATH=$(LMDB_BASE)/llm-agent \
	OLLAMA_URL=$(OLLAMA_URL) \
	OLLAMA_MODEL=$(OLLAMA_MODEL) \
	$(ACTOR_BIN) &
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

# ── Clean ──────────────────────────────────────────────────────────────────

clean:
	@rm -rf /tmp/employee-mesh
	@echo "[clean] done"
