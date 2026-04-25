/* proxy.c — ZMQ XPUB/XSUB dumb fanout proxy
 *
 * No logic. No routing. Just moves bytes.
 * All actors connect here.
 *
 * Env:
 *   PROXY_XSUB_BIND   actors publish here    e.g. tcp-star-5557
 *   PROXY_XPUB_BIND   actors subscribe here  e.g. tcp-star-5556
 *   PROXY_COMPRESS    zlib | zstd | none (default none)
 *                     Note: requires libzmq built with compression support.
 *                     Standard distro builds default to none.
 *                     Alternatively handle compression at the actor layer.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zmq.h>

/* ZMQ compression constants — defined in libzmq >= 4.3 custom builds.
 * Guard with ifdef so the binary still compiles on standard distro libzmq. */
#ifndef ZMQ_COMPRESS
#  define ZMQ_COMPRESS       86
#  define ZMQ_COMPRESS_NONE  0
#  define ZMQ_COMPRESS_ZLIB  1
#  define ZMQ_COMPRESS_ZSTD  2
#endif

static void set_compress(void* sock, int method, const char* label) {
    int rc = zmq_setsockopt(sock, ZMQ_COMPRESS, &method, sizeof(method));
    if (rc != 0) {
        fprintf(stderr, "[proxy] compression=%s not supported by this libzmq build"
                        " — running without compression\n", label);
    } else {
        fprintf(stderr, "[proxy] compression=%s\n", label);
    }
}

int main(void) {
    const char* xsub_bind = getenv("PROXY_XSUB_BIND");
    const char* xpub_bind = getenv("PROXY_XPUB_BIND");
    const char* compress  = getenv("PROXY_COMPRESS");

    if (!xsub_bind) xsub_bind = "tcp://*:5557";
    if (!xpub_bind) xpub_bind = "tcp://*:5556";
    if (!compress)  compress  = "none";

    void* ctx  = zmq_ctx_new();
    void* xsub = zmq_socket(ctx, ZMQ_XSUB);
    void* xpub = zmq_socket(ctx, ZMQ_XPUB);

    if (strcmp(compress, "zlib") == 0) {
        set_compress(xsub, ZMQ_COMPRESS_ZLIB, "zlib");
        set_compress(xpub, ZMQ_COMPRESS_ZLIB, "zlib");
    } else if (strcmp(compress, "zstd") == 0) {
        set_compress(xsub, ZMQ_COMPRESS_ZSTD, "zstd");
        set_compress(xpub, ZMQ_COMPRESS_ZSTD, "zstd");
    } else {
        fprintf(stderr, "[proxy] compression=none\n");
    }

    zmq_bind(xsub, xsub_bind);
    zmq_bind(xpub, xpub_bind);

    fprintf(stderr, "[proxy] xsub=%s xpub=%s\n", xsub_bind, xpub_bind);

    /* blocks forever — OS kills on SIGTERM */
    zmq_proxy(xsub, xpub, NULL);

    zmq_close(xsub);
    zmq_close(xpub);
    zmq_ctx_destroy(ctx);
    return 0;
}
