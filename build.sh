#!/bin/bash
# build.sh — cross-platform build for actor mesh
#
# Usage:
#   ./build.sh                          native build
#   ./build.sh --target aarch64-linux-musl   ARM64 static
#   ./build.sh --target x86_64-windows-gnu   Windows .exe
#   ./build.sh --target x86_64-macos         Intel Mac
#
# Requires: zig (https://ziglang.org/download)
# Zig bundles clang + libc headers for every target. No sysroot needed.

set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
TARGET="${1:-native}"
ZIG="${ZIG:-zig}"

# --- config ---
NNG_VERSION="1.9.0"
LMDB_VERSION="0.9.33"

CFLAGS="-Wall -Wextra -O2 -std=c11 -I$ROOT/runtime"
BUILD_DIR="$ROOT/build/$TARGET"
DEPS_DIR="$BUILD_DIR/deps"

if [ "$TARGET" != "native" ]; then
    CFLAGS="$CFLAGS -target $TARGET"
    LDFLAGS="-target $TARGET"
    # static linking for musl targets
    case "$TARGET" in
        *musl*) LDFLAGS="$LDFLAGS -static" ;;
    esac
fi

echo "=== actor mesh build ==="
echo "  target: $TARGET"
echo "  zig:    $($ZIG version)"
echo "  cc:     $ZIG cc"

mkdir -p "$BUILD_DIR" "$DEPS_DIR"

# --- lmdb (single C file, trivial to cross-compile) ---
LMDB_SRC="$ROOT/vendor/lmdb"
if [ ! -f "$DEPS_DIR/liblmdb.a" ]; then
    echo "=== building lmdb ==="
    mkdir -p "$LMDB_SRC"
    if [ ! -f "$LMDB_SRC/lmdb.h" ]; then
        # fetch lmdb source
        curl -sL "https://github.com/LMDB/lmdb/archive/refs/tags/LMDB_${LMDB_VERSION}.tar.gz" \
            | tar xz -C /tmp
        cp /tmp/lmdb-LMDB_${LMDB_VERSION}/libraries/liblmdb/lmdb.h "$LMDB_SRC/"
        cp /tmp/lmdb-LMDB_${LMDB_VERSION}/libraries/liblmdb/mdb.c  "$LMDB_SRC/"
        cp /tmp/lmdb-LMDB_${LMDB_VERSION}/libraries/liblmdb/midl.c "$LMDB_SRC/"
        cp /tmp/lmdb-LMDB_${LMDB_VERSION}/libraries/liblmdb/midl.h "$LMDB_SRC/"
        rm -rf /tmp/lmdb-LMDB_${LMDB_VERSION}
    fi
    $ZIG cc $CFLAGS -c "$LMDB_SRC/mdb.c"  -o "$DEPS_DIR/mdb.o"
    $ZIG cc $CFLAGS -c "$LMDB_SRC/midl.c" -o "$DEPS_DIR/midl.o"
    $ZIG ar crs "$DEPS_DIR/liblmdb.a" "$DEPS_DIR/mdb.o" "$DEPS_DIR/midl.o"
    echo "  -> $DEPS_DIR/liblmdb.a"
fi

# --- nng (needs cmake + zig for cross-compile) ---
NNG_SRC="$ROOT/vendor/nng"
NNG_BUILD="$BUILD_DIR/nng-build"
if [ ! -f "$DEPS_DIR/libnng.a" ]; then
    echo "=== building nng ==="
    if [ ! -f "$NNG_SRC/CMakeLists.txt" ]; then
        git clone --depth 1 --branch "v$NNG_VERSION" \
            https://github.com/nanomsg/nng.git "$NNG_SRC" 2>/dev/null || true
    fi
    mkdir -p "$NNG_BUILD"
    cd "$NNG_BUILD"
    cmake "$NNG_SRC" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DNNG_PROTOCOL_PAIR0=OFF \
        -DNNG_PROTOCOL_REQ0=OFF \
        -DNNG_PROTOCOL_REP0=OFF \
        -DNNG_PROTOCOL_PUSH0=OFF \
        -DNNG_PROTOCOL_PULL0=OFF \
        -DNNG_PROTOCOL_SURVEYOR0=OFF \
        -DNNG_PROTOCOL_RESPONDENT0=OFF \
        -DNNG_PROTOCOL_BUS0=OFF \
        -DNNG_TRANSPORT_INPROC=OFF \
        -DNNG_TRANSPORT_IPC=OFF \
        -DNNG_TRANSPORT_WS=OFF \
        -DNNG_TRANSPORT_WSS=OFF \
        -DNNG_TRANSPORT_ZEROTIER=OFF \
        -DNNG_ENABLE_TLS=OFF \
        -DNNG_TOOLS=OFF \
        -DNNG_TESTS=OFF \
        -DCMAKE_C_COMPILER="$ZIG cc" \
        $([ "$TARGET" != "native" ] && echo "-DCMAKE_C_FLAGS=-target $TARGET") \
        -DCMAKE_AR="$ZIG ar" \
        -DCMAKE_RANLIB="$ZIG ranlib"
    make -j$(nproc) nng
    cp libnng.a "$DEPS_DIR/"
    echo "  -> $DEPS_DIR/libnng.a"
    cd "$ROOT"
fi

# --- actor mesh ---
echo "=== building actor mesh ==="
INCLUDES="-I$NNG_SRC/include -I$LMDB_SRC"
LIBS="$DEPS_DIR/libnng.a $DEPS_DIR/liblmdb.a -lpthread"

$ZIG cc $CFLAGS $LDFLAGS $INCLUDES \
    "$ROOT/runtime/main.c" "$ROOT/runtime/actor.c" \
    $LIBS -o "$BUILD_DIR/actor"

$ZIG cc $CFLAGS $LDFLAGS $INCLUDES \
    "$ROOT/proxy/proxy.c" \
    $LIBS -o "$BUILD_DIR/mesh-proxy"

echo ""
echo "=== done ==="
ls -lh "$BUILD_DIR/actor" "$BUILD_DIR/mesh-proxy"
file "$BUILD_DIR/actor"
