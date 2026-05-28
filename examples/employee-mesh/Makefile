# Makefile — employee mesh demo
#
# ── Quick start ────────────────────────────────────────────────────────────
#   make                build handlers
#   make ui             setup desktop UI deps
#   make run-ui         start proxy + actors + desktop UI
#   make stop           kill everything
#
# ── LLM key ────────────────────────────────────────────────────────────────
#   export LLM_API_KEY=sk-... && make run-ui

ROOT    = ../..
RUNTIME = $(ROOT)/runtime
VENDOR  = $(ROOT)/vendor

CC           = gcc
CXX          = g++
CFLAGS       = -Wall -Wextra -O2 -std=c11 -I$(RUNTIME) -I$(VENDOR)/mpack/src/mpack -I$(ROOT)
LIBS         = -lnng -llmdb
HANDLER_LIBS = -lcurl -llmdb -lsqlite3

NER_MODEL    ?= $(abspath ../../models/ner.onnx)
NER_INCS     = -I$(ROOT)/vendor/smhasher -I$(ROOT) -I/usr/include/onnxruntime
NER_LIBS     = -lonnxruntime -lstdc++
SMHASHER     = $(ROOT)/vendor/smhasher/MurmurHash2.cpp

MPACK_SRCS = $(VENDOR)/mpack/src/mpack/mpack-common.c $(VENDOR)/mpack/src/mpack/mpack-platform.c $(VENDOR)/mpack/src/mpack/mpack-reader.c $(VENDOR)/mpack/src/mpack/mpack-writer.c $(VENDOR)/mpack/src/mpack/mpack-expect.c $(VENDOR)/mpack/src/mpack/mpack-node.c
CJSON      = $(VENDOR)/cjson/cJSON.c
ACTOR      = $(ROOT)/actor
PROXY      = $(ROOT)/mesh-proxy

# ── Build ──────────────────────────────────────────────────────────────────

.PHONY: all handlers ui client tui clean

all: handlers

handlers: handlers/llm-agent handlers/sqlite-tool handlers/pii-filter

handlers/llm-agent: handlers/llm-agent.c $(MPACK_SRCS) $(CJSON)
	$(CC) $(CFLAGS) -I$(VENDOR)/cjson handlers/llm-agent.c $(MPACK_SRCS) $(CJSON) $(HANDLER_LIBS) -o $@

handlers/sqlite-tool: handlers/sqlite-tool.c $(MPACK_SRCS)
	$(CC) $(CFLAGS) handlers/sqlite-tool.c $(MPACK_SRCS) -lsqlite3 -o $@

handlers/pii-filter: handlers/pii-filter.c $(MPACK_SRCS) $(SMHASHER)
	$(CXX) $(CFLAGS) $(NER_INCS) handlers/pii-filter.c $(MPACK_SRCS) $(SMHASHER) $(HANDLER_LIBS) $(NER_LIBS) -o $@

client: client.c $(MPACK_SRCS)
	$(CC) $(CFLAGS) client.c $(MPACK_SRCS) $(LIBS) -o $@

tui: tui.c $(MPACK_SRCS)
	$(CC) $(CFLAGS) tui.c $(MPACK_SRCS) $(LIBS) -lpthread -lncursesw -o $@

ui:
	@cd ui && npm install && gcc -Wall -O2 -std=c11 bridge.c -lnng -o src-tauri/bridge && echo "[ui] ready"

clean:
	rm -f client tui handlers/llm-agent handlers/sqlite-tool handlers/pii-filter
	cd ui && rm -rf src-tauri/target src-tauri/bridge node_modules 2>/dev/null; true

# ── Config ─────────────────────────────────────────────────────────────────

LMDB_BASE   ?= /tmp/actor-mesh
EMPLOYEE_DB ?= $(abspath db/employee.db)
LOGDIR      ?= $(LMDB_BASE)/logs

BUS_SUB ?= tcp://127.0.0.1:5556
BUS_PUB ?= tcp://127.0.0.1:5557

LLM_BASE_URL ?= https://api.deepseek.com
LLM_API_KEY  ?=
LLM_MODEL    ?= deepseek-chat

COMMON_ENV = ACTOR_BUS_SUB=$(BUS_SUB) ACTOR_BUS_PUB=$(BUS_PUB) ACTOR_HEARTBEAT_MS=2000 ACTOR_RETRY_MAX=3

# ── Run ────────────────────────────────────────────────────────────────────

.PHONY: stop run-ui

stop:
	@pkill -9 mesh-proxy 2>/dev/null; true
	@pkill -9 actor 2>/dev/null; true
	@pkill -9 actor-mesh-desktop 2>/dev/null; true
	@rm -rf $(LMDB_BASE)
	@echo "[mesh] stopped"

run-ui: stop
	@# proxy
	@mkdir -p $(LMDB_BASE) $(LOGDIR)
	@PROXY_SUB_BIND=tcp://127.0.0.1:5557 PROXY_PUB_BIND=tcp://127.0.0.1:5556 $(PROXY) >>$(LOGDIR)/proxy.log 2>&1 &
	@sleep 0.3
	@# sqlite-tool
	@mkdir -p $(LMDB_BASE)/sqlite-tool
	@$(COMMON_ENV) ACTOR_ID=sqlite-tool-1 ACTOR_TOPIC=sql_query ACTOR_RESULT_TOPIC=sql_result ACTOR_HANDLER=$(abspath handlers/sqlite-tool) ACTOR_LMDB_PATH=$(LMDB_BASE)/sqlite-tool EMPLOYEE_DB=$(EMPLOYEE_DB) $(ACTOR) >>$(LOGDIR)/sqlite-tool.log 2>&1 &
	@sleep 0.2
	@# llm-agent
	@mkdir -p $(LMDB_BASE)/llm-agent
	@$(COMMON_ENV) ACTOR_ID=llm-agent-1 ACTOR_TOPIC=user_message,sql_result ACTOR_RESULT_TOPIC=agent_response ACTOR_HANDLER=$(abspath handlers/llm-agent) ACTOR_LMDB_PATH=$(LMDB_BASE)/llm-agent LLM_BASE_URL=$(LLM_BASE_URL) LLM_API_KEY=$(LLM_API_KEY) LLM_MODEL=$(LLM_MODEL) EMPLOYEE_DB=$(EMPLOYEE_DB) $(ACTOR) >>$(LOGDIR)/llm-agent.log 2>&1 &
	@sleep 0.2
	@# shell-exec
	@mkdir -p $(LMDB_BASE)/shell-exec
	@$(COMMON_ENV) ACTOR_ID=shell-exec-1 ACTOR_TOPIC=shell_exec ACTOR_RESULT_TOPIC=shell_result ACTOR_HANDLER=$(abspath handlers/shell-exec) ACTOR_LMDB_PATH=$(LMDB_BASE)/shell-exec $(ACTOR) >>$(LOGDIR)/shell-exec.log 2>&1 &
	@sleep 0.2
	@echo "[mesh] proxy + 3 actors running"
	@echo ""
	@cd ui && GSK_RENDERER=cairo GDK_BACKEND=x11 WEBKIT_DISABLE_COMPOSITING_MODE=1 npx tauri dev
