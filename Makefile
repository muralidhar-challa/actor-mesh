# Makefile — actor mesh
#
# Dependencies:
#   apt install libzmq3-dev liblmdb-dev
#
# Targets:
#   make actor    — build actor binary
#   make proxy    — build proxy binary
#   make all      — build everything
#   make clean    — remove binaries

CC     = gcc
CFLAGS = -Wall -Wextra -O2 -std=c11 -Iruntime
LIBS   = -lzmq -llmdb

.PHONY: all actor proxy clean

all: actor proxy

actor: runtime/main.c runtime/actor.c runtime/actor.h \
       runtime/actor_tuple.h runtime/actor_uuid.h
	$(CC) $(CFLAGS) runtime/main.c runtime/actor.c $(LIBS) -o actor

proxy: proxy/proxy.c
	$(CC) $(CFLAGS) proxy/proxy.c $(LIBS) -o zmq-proxy

clean:
	rm -f actor zmq-proxy
