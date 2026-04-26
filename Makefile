# Makefile — actor mesh runtime
#
# ── Build ──────────────────────────────────────────────────────────────────
#   make          build actor + zmq-proxy
#   make actor    build actor only
#   make proxy    build proxy only
#   make clean    remove binaries
#
# ── Deps ───────────────────────────────────────────────────────────────────
#   Fedora/RHEL:  dnf install zeromq-devel lmdb-devel
#   Debian/Ubuntu: apt install libzmq3-dev liblmdb-dev

CC     = gcc
CFLAGS = -Wall -Wextra -O2 -std=c11 -Iruntime
LIBS   = -lzmq -llmdb

ifdef ACTOR_MAX_PAYLOAD
CFLAGS += -DACTOR_MAX_PAYLOAD=$(ACTOR_MAX_PAYLOAD)
endif

.PHONY: all actor proxy clean

all: actor proxy

actor: runtime/main.c runtime/actor.c runtime/actor.h \
       runtime/actor_tuple.h runtime/actor_uuid.h
	$(CC) $(CFLAGS) runtime/main.c runtime/actor.c $(LIBS) -o actor

proxy: proxy/proxy.c
	$(CC) $(CFLAGS) proxy/proxy.c $(LIBS) -o zmq-proxy

clean:
	rm -f actor zmq-proxy
