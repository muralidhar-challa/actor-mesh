# Makefile — actor mesh runtime
#
# ── Build ──────────────────────────────────────────────────────────────────
#   make          build actor + mesh-proxy
#   make clean    remove binaries
#
# ── Deps ───────────────────────────────────────────────────────────────────
#   Fedora/RHEL:  dnf install nng-devel lmdb-devel
#   Debian/Ubuntu: apt install libnng-dev liblmdb-dev

CC     = gcc
CFLAGS = -Wall -Wextra -O2 -std=c11 -Iruntime
LIBS   = -lnng -llmdb

ifdef ACTOR_MAX_PAYLOAD
CFLAGS += -DACTOR_MAX_PAYLOAD=$(ACTOR_MAX_PAYLOAD)
endif

.PHONY: all actor proxy clean

all: actor mesh-proxy

actor: runtime/main.c runtime/actor.c runtime/actor.h \
       runtime/actor_tuple.h runtime/actor_uuid.h
	$(CC) $(CFLAGS) runtime/main.c runtime/actor.c $(LIBS) -o actor

mesh-proxy: proxy/proxy.c
	$(CC) $(CFLAGS) proxy/proxy.c $(LIBS) -o mesh-proxy

clean:
	rm -f actor mesh-proxy
