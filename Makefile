# Makefile — actor mesh runtime
#
# ── Build ──────────────────────────────────────────────────────────────────
#   make              build natively for current platform
#   make clean        remove binaries
#
# ── Cross-compile ──────────────────────────────────────────────────────────
#   Requires target libc + nng + lmdb. Use zig or musl-cross for easy setup:
#
#   make linux-arm64  CC="zig cc" TARGET=aarch64-linux-musl
#   make windows-x64  CC="zig cc" TARGET=x86_64-windows-gnu
#   make macos-x64    CC="zig cc" TARGET=x86_64-macos
#
# ── Deps ───────────────────────────────────────────────────────────────────
#   Fedora/RHEL:  dnf install nng-devel lmdb-devel
#   Debian/Ubuntu: apt install libnng-dev liblmdb-dev

CC      ?= gcc
TARGET  ?= native
ZIG     ?= zig

# Use zig cc if gcc not found — zig IS clang + cross-targets
ifeq ($(shell which $(CC) 2>/dev/null),)
  ifeq ($(shell which $(ZIG) 2>/dev/null),)
    $(error neither $(CC) nor $(ZIG) found. Install zig: https://ziglang.org/download)
  endif
  CC := $(ZIG) cc
endif
CFLAGS  = -Wall -Wextra -O2 -std=c11 -Iruntime
LDFLAGS =
LIBS    = -lnng -llmdb

ifdef ACTOR_MAX_PAYLOAD
CFLAGS += -DACTOR_MAX_PAYLOAD=$(ACTOR_MAX_PAYLOAD)
endif

ifneq ($(TARGET),native)
  CFLAGS  += -target $(TARGET)
endif

.PHONY: all actor mesh-proxy clean linux-arm64 macos-x64 macos-arm64 windows-x64

all: actor mesh-proxy

actor: runtime/main.c runtime/actor.c runtime/actor.h \
       runtime/actor_tuple.h runtime/actor_uuid.h
	$(CC) $(CFLAGS) $(LDFLAGS) runtime/main.c runtime/actor.c $(LIBS) -o actor

mesh-proxy: proxy/proxy.c
	$(CC) $(CFLAGS) $(LDFLAGS) proxy/proxy.c $(LIBS) -o mesh-proxy

clean:
	rm -f actor mesh-proxy

# Cross-compile convenience targets — use with CC=zig or CC=clang + sysroot
# Cross-compile targets (use CC=zig cc). Linux works out of box.
# Windows/macOS need extra SDK — use build.sh for those.
linux-arm64:
	$(MAKE) CC="$(CC)" TARGET=aarch64-linux-musl

linux-x64-static:
	$(MAKE) CC="$(CC)" TARGET=x86_64-linux-musl
