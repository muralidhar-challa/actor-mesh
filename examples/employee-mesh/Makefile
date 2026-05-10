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
CXX          = g++
CFLAGS       = -Wall -Wextra -O2 -std=c11 -I$(RUNTIME) -I$(VENDOR)/mpack/src/mpack -I$(ROOT)
LIBS         = -lzmq -llmdb
HANDLER_LIBS = -lcurl -llmdb -lsqlite3

NER_MODEL    ?= $(abspath ../../models/ner.onnx)
NER_INCS     = -I$(ROOT)/vendor/smhasher -I$(ROOT) -I/usr/include/onnxruntime
NER_LIBS     = -lonnxruntime -lstdc++
SMHASHER     = $(ROOT)/vendor/smhasher/MurmurHash2.cpp

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

handlers: handlers/llm-agent handlers/sqlite-tool handlers/pii-filter

handlers/llm-agent: handlers/llm-agent.c $(MPACK_SRCS) $(CJSON)
	$(CC) $(CFLAGS) -I$(VENDOR)/cjson handlers/llm-agent.c $(MPACK_SRCS) $(CJSON) $(HANDLER_LIBS) -o handlers/llm-agent

handlers/sqlite-tool: handlers/sqlite-tool.c $(MPACK_SRCS)
	$(CC) $(CFLAGS) handlers/sqlite-tool.c $(MPACK_SRCS) -lsqlite3 -o handlers/sqlite-tool

handlers/pii-filter: handlers/pii-filter.c $(MPACK_SRCS) $(SMHASHER)
	$(CXX) $(CFLAGS) $(NER_INCS) handlers/pii-filter.c $(MPACK_SRCS) $(SMHASHER) \
		$(HANDLER_LIBS) $(NER_LIBS) -o handlers/pii-filter

clean:
	rm -f client tui handlers/llm-agent handlers/sqlite-tool handlers/pii-filter

# ── Paths ──────────────────────────────────────────────────────────────────

LMDB_BASE    ?= /tmp/actor-mesh
TOKENMAP_PATH ?= $(LMDB_BASE)/tokenmap
EMPLOYEE_DB  ?= $(abspath db/employee.db)

# ── Bus ────────────────────────────────────────────────────────────────────

PROXY_XSUB_BIND ?= tcp://*:5557
PROXY_XPUB_BIND ?= tcp://*:5556
BUS_SUB         ?= tcp://localhost:5556
BUS_PUB         ?= tcp://localhost:5557

# ── LLM (OpenAI-compatible: DeepSeek, OpenAI, etc.) ───────────────────────

LLM_BASE_URL ?= https://api.deepseek.com
LLM_API_KEY  ?=
LLM_MODEL    ?= deepseek-chat

# ── Common actor env ───────────────────────────────────────────────────────

COMMON_ENV = \
	ACTOR_BUS_SUB=$(BUS_SUB) \
	ACTOR_BUS_PUB=$(BUS_PUB) \
	ACTOR_HEARTBEAT_MS=5000 \
	ACTOR_RETRY_MAX=3 \
	ACTOR_TTL_NS=300000000000

# ── Run ────────────────────────────────────────────────────────────────────

.PHONY: run stop run-proxy run-sqlite-tool run-llm-agent run-tokenise run-detokenise query chat

run: run-proxy run-tokenise run-detokenise run-sqlite-tool run-llm-agent
	@echo "[mesh] employee mesh running (PII tokenisation enabled)"
	@echo "[mesh] try: make query Q=\"list all engineers\""

stop:
	@pkill -9 zmq-proxy || true
	@pkill -9 actor     || true
	@rm -rf $(LMDB_BASE)
	@echo "[mesh] stopped  (logs were in $(LOGDIR))"

LOGDIR ?= /tmp/actor-mesh/logs

run-proxy:
	@mkdir -p $(LMDB_BASE) $(LOGDIR) $(TOKENMAP_PATH)
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
	ACTOR_TOPIC=user_message_masked,sql_result_masked \
	ACTOR_RESULT_TOPIC=sql_query_masked \
	ACTOR_HANDLER=$(abspath handlers/llm-agent) \
	ACTOR_LMDB_PATH=$(LMDB_BASE)/llm-agent \
	LLM_BASE_URL=$(LLM_BASE_URL) \
	LLM_API_KEY=$(LLM_API_KEY) \
	LLM_MODEL=$(LLM_MODEL) \
	EMPLOYEE_DB=$(EMPLOYEE_DB) \
	$(ACTOR) >>$(LOGDIR)/llm-agent.log 2>&1 &
	@echo "[actor] llm-agent started  (log: $(LOGDIR)/llm-agent.log)"

run-tokenise:
	@mkdir -p $(LMDB_BASE)/tokenise $(LOGDIR)
	@$(COMMON_ENV) \
	ACTOR_ID=tokenise-1 \
	ACTOR_TOPIC=user_message,sql_result \
	ACTOR_RESULT_TOPIC=user_message_masked \
	ACTOR_HANDLER=$(abspath handlers/pii-filter) \
	ACTOR_LMDB_PATH=$(LMDB_BASE)/tokenise \
	PII_MODE=mask \
	NER_MODEL_PATH=$(NER_MODEL) \
	TOKENMAP_PATH=$(TOKENMAP_PATH) \
	$(ACTOR) >>$(LOGDIR)/tokenise.log 2>&1 &
	@echo "[actor] tokenise started  (log: $(LOGDIR)/tokenise.log)"

run-detokenise:
	@mkdir -p $(LMDB_BASE)/detokenise $(LOGDIR)
	@$(COMMON_ENV) \
	ACTOR_ID=detokenise-1 \
	ACTOR_TOPIC=sql_query_masked,agent_response_masked \
	ACTOR_RESULT_TOPIC=agent_response \
	ACTOR_HANDLER=$(abspath handlers/pii-filter) \
	ACTOR_LMDB_PATH=$(LMDB_BASE)/detokenise \
	PII_MODE=restore \
	TOKENMAP_PATH=$(TOKENMAP_PATH) \
	$(ACTOR) >>$(LOGDIR)/detokenise.log 2>&1 &
	@echo "[actor] detokenise started  (log: $(LOGDIR)/detokenise.log)"

# ── Query / Chat ───────────────────────────────────────────────────────────

Q ?= list all employees

query:
	@./client "$(Q)"

chat:
	@./tui
