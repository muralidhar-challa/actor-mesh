# Makefile — employee mesh demo
#
# ── Quick start ────────────────────────────────────────────────────────────
#   make                build handlers
#   make ui             setup desktop UI deps
#   make run-ui         start proxy + actors + desktop UI (full PII pipeline)
#   make stop           kill everything
#
# ── LLM key ────────────────────────────────────────────────────────────────
#   export LLM_API_KEY=sk-... && make run-ui

ROOT    = ../..
RUNTIME = $(ROOT)/runtime
VENDOR  = $(ROOT)/vendor

CC           = gcc
CXX          = g++
CFLAGS       = -Wall -Wextra -O2 -std=c11 -I$(RUNTIME) -I$(VENDOR)/mpack -I$(ROOT)
LIBS         = -lnng -llmdb
HANDLER_LIBS = -lcurl -llmdb -lsqlite3

NER_MODEL    ?= $(abspath ../../models/ner.onnx)
NER_INCS     = -I$(ROOT)/vendor/smhasher -Ihandlers/lib -I$(ROOT) -I/usr/include/onnxruntime
NER_LIBS     = -lonnxruntime -lstdc++
SMHASHER     = $(ROOT)/vendor/smhasher/MurmurHash2.cpp

MPACK_SRCS = $(VENDOR)/mpack/mpack-common.c $(VENDOR)/mpack/mpack-platform.c $(VENDOR)/mpack/mpack-reader.c $(VENDOR)/mpack/mpack-writer.c $(VENDOR)/mpack/mpack-expect.c $(VENDOR)/mpack/mpack-node.c
CJSON      = $(VENDOR)/cjson/cJSON.c
ACTOR      = $(ROOT)/actor
PROXY      = $(ROOT)/mesh-proxy

# ── Build ──────────────────────────────────────────────────────────────────

.PHONY: all handlers ui client tui clean

all: handlers

handlers: handlers/agents/llm-agent handlers/tools/sqlite-tool handlers/tools/pii-filter

handlers/agents/llm-agent: handlers/agents/llm-agent.c $(MPACK_SRCS) $(CJSON)
	$(CC) $(CFLAGS) -I$(VENDOR)/cjson handlers/agents/llm-agent.c $(MPACK_SRCS) $(CJSON) $(HANDLER_LIBS) -o $@

handlers/tools/sqlite-tool: handlers/tools/sqlite-tool.c $(MPACK_SRCS)
	$(CC) $(CFLAGS) handlers/tools/sqlite-tool.c $(MPACK_SRCS) -lsqlite3 -o $@

handlers/tools/pii-filter: handlers/tools/pii-filter.c $(MPACK_SRCS) $(SMHASHER)
	$(CXX) $(CFLAGS) $(NER_INCS) handlers/tools/pii-filter.c $(MPACK_SRCS) $(SMHASHER) $(HANDLER_LIBS) $(NER_LIBS) -o $@

client: client.c $(MPACK_SRCS)
	$(CC) $(CFLAGS) client.c $(MPACK_SRCS) $(LIBS) -o $@

tui: tui.c $(MPACK_SRCS)
	$(CC) $(CFLAGS) tui.c $(MPACK_SRCS) $(LIBS) -lpthread -lncursesw -o $@

ui:
	@cd ui && npm install && gcc -Wall -O2 -std=c11 bridge.c -lnng -o src-tauri/bridge && echo "[ui] ready"

clean:
	rm -f client tui handlers/agents/llm-agent handlers/tools/sqlite-tool handlers/tools/pii-filter
	cd ui && rm -rf src-tauri/target src-tauri/bridge node_modules 2>/dev/null; true

# ── Config ─────────────────────────────────────────────────────────────────

LMDB_BASE    ?= /tmp/actor-mesh
TOKENMAP_PATH ?= $(LMDB_BASE)/tokenmap
EMPLOYEE_DB  ?= $(abspath db/employee.db)
LOGDIR       ?= $(LMDB_BASE)/logs

BUS_SUB ?= tcp://127.0.0.1:5556
BUS_PUB ?= tcp://127.0.0.1:5557

LLM_BASE_URL ?= https://api.deepseek.com
LLM_API_KEY  ?=
LLM_MODEL    ?= deepseek-chat

COMMON_ENV = ACTOR_BUS_SUB=$(BUS_SUB) ACTOR_BUS_PUB=$(BUS_PUB) ACTOR_HEARTBEAT_MS=2000 ACTOR_RETRY_MAX=3

# ── Run (full PII pipeline) ────────────────────────────────────────────────

.PHONY: stop run-ui

stop:
	@pkill -9 mesh-proxy 2>/dev/null; true
	@pkill -9 actor 2>/dev/null; true
	@pkill -9 actor-mesh-desktop 2>/dev/null; true
	@rm -rf $(LMDB_BASE)
	@echo "[mesh] stopped"

run-ui: stop
	@# proxy
	@mkdir -p $(LMDB_BASE) $(LOGDIR) $(TOKENMAP_PATH)
	@PROXY_SUB_BIND=tcp://127.0.0.1:5557 PROXY_PUB_BIND=tcp://127.0.0.1:5556 $(PROXY) >>$(LOGDIR)/proxy.log 2>&1 &
	@sleep 0.3
	@echo "[proxy] started"

	@# tokenise — subscribes user_message,sql_result → publishes *_masked
	@mkdir -p $(LMDB_BASE)/tokenise
	@$(COMMON_ENV) ACTOR_ID=tokenise-1 \
		ACTOR_TOPIC=user_message,sql_result ACTOR_RESULT_TOPIC=user_message_masked \
		ACTOR_HANDLER=$(abspath handlers/tools/pii-filter) ACTOR_LMDB_PATH=$(LMDB_BASE)/tokenise \
		PII_MODE=mask NER_MODEL_PATH=$(NER_MODEL) TOKENMAP_PATH=$(TOKENMAP_PATH) \
		$(ACTOR) >>$(LOGDIR)/tokenise.log 2>&1 &
	@sleep 0.2
	@echo "[actor] tokenise started"

	@# detokenise — subscribes *_masked → publishes restored names
	@mkdir -p $(LMDB_BASE)/detokenise
	@$(COMMON_ENV) ACTOR_ID=detokenise-1 \
		ACTOR_TOPIC=sql_query_masked,agent_response_masked ACTOR_RESULT_TOPIC=agent_response \
		ACTOR_HANDLER=$(abspath handlers/tools/pii-filter) ACTOR_LMDB_PATH=$(LMDB_BASE)/detokenise \
		PII_MODE=restore TOKENMAP_PATH=$(TOKENMAP_PATH) \
		$(ACTOR) >>$(LOGDIR)/detokenise.log 2>&1 &
	@sleep 0.2
	@echo "[actor] detokenise started"

	@# sqlite-tool
	@mkdir -p $(LMDB_BASE)/sqlite-tool
	@$(COMMON_ENV) ACTOR_ID=sqlite-tool-1 \
		ACTOR_TOPIC=sql_query ACTOR_RESULT_TOPIC=sql_result \
		ACTOR_HANDLER=$(abspath handlers/tools/sqlite-tool) ACTOR_LMDB_PATH=$(LMDB_BASE)/sqlite-tool \
		EMPLOYEE_DB=$(EMPLOYEE_DB) \
		$(ACTOR) >>$(LOGDIR)/sqlite-tool.log 2>&1 &
	@sleep 0.2
	@echo "[actor] sqlite-tool started"

	@# llm-agent — receives masked, emits masked
	@mkdir -p $(LMDB_BASE)/llm-agent
	@$(COMMON_ENV) ACTOR_ID=llm-agent-1 \
		ACTOR_TOPIC=user_message_masked,sql_result_masked ACTOR_RESULT_TOPIC=sql_query_masked \
		ACTOR_HANDLER=$(abspath handlers/agents/llm-agent) ACTOR_LMDB_PATH=$(LMDB_BASE)/llm-agent \
		LLM_BASE_URL=$(LLM_BASE_URL) LLM_API_KEY=$(LLM_API_KEY) LLM_MODEL=$(LLM_MODEL) \
		EMPLOYEE_DB=$(EMPLOYEE_DB) \
		$(ACTOR) >>$(LOGDIR)/llm-agent.log 2>&1 &
	@sleep 0.2
	@echo "[actor] llm-agent started"

	@# shell-exec
	@mkdir -p $(LMDB_BASE)/shell-exec
	@$(COMMON_ENV) ACTOR_ID=shell-exec-1 \
		ACTOR_TOPIC=shell_exec ACTOR_RESULT_TOPIC=shell_result \
		ACTOR_HANDLER=$(abspath handlers/tools/shell-exec) ACTOR_LMDB_PATH=$(LMDB_BASE)/shell-exec \
		$(ACTOR) >>$(LOGDIR)/shell-exec.log 2>&1 &
	@sleep 0.2
	@echo "[actor] shell-exec started"

	@echo "[mesh] proxy + 5 actors running (PII pipeline active)"
	@echo ""
	@cd ui && GSK_RENDERER=cairo GDK_BACKEND=x11 WEBKIT_DISABLE_COMPOSITING_MODE=1 npx tauri dev
