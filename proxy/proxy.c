// proxy.c — NNG pub/sub dumb fanout proxy
//
// No logic. No routing. Just moves bytes.
// All actors connect here.
//
// Env:
//   PROXY_SUB_BIND   where publishers dial  (default tcp://*:5557)
//   PROXY_PUB_BIND   where subscribers dial  (default tcp://*:5556)

#include <stdio.h>
#include <stdlib.h>
#include <nng/nng.h>
#include <nng/protocol/pubsub0/pub.h>
#include <nng/protocol/pubsub0/sub.h>

int main(void) {
    const char* sub_bind = getenv("PROXY_SUB_BIND");
    const char* pub_bind = getenv("PROXY_PUB_BIND");

    if (!sub_bind) sub_bind = "tcp://*:5557";
    if (!pub_bind) pub_bind = "tcp://*:5556";

    nng_socket sub, pub;
    int        rc;

    if ((rc = nng_sub0_open(&sub)) != 0) {
        fprintf(stderr, "[proxy] nng_sub0_open: %s\n", nng_strerror(rc));
        return 1;
    }
    if ((rc = nng_pub0_open(&pub)) != 0) {
        fprintf(stderr, "[proxy] nng_pub0_open: %s\n", nng_strerror(rc));
        nng_close(sub);
        return 1;
    }

    if ((rc = nng_listen(sub, sub_bind, NULL, 0)) != 0) {
        fprintf(stderr, "[proxy] sub listen %s: %s\n", sub_bind, nng_strerror(rc));
        nng_close(pub);
        nng_close(sub);
        return 1;
    }
    /* Subscribe to all topics */
    nng_socket_set(sub, NNG_OPT_SUB_SUBSCRIBE, "", 0);

    if ((rc = nng_listen(pub, pub_bind, NULL, 0)) != 0) {
        fprintf(stderr, "[proxy] pub listen %s: %s\n", pub_bind, nng_strerror(rc));
        nng_close(pub);
        nng_close(sub);
        return 1;
    }

    fprintf(stderr, "[proxy] sub=%s pub=%s\n", sub_bind, pub_bind);

    /* Manual forwarding loop */
    for (;;) {
        nng_msg* msg = NULL;
        if ((rc = nng_recvmsg(sub, &msg, 0)) != 0) {
            fprintf(stderr, "[proxy] recv err: %s\n", nng_strerror(rc));
            continue;
        }
        size_t len = nng_msg_len(msg);
        fprintf(stderr, "[proxy] fwd %zu bytes\n", len);
        rc = nng_sendmsg(pub, msg, 0);
        nng_msg_free(msg);
        if (rc != 0) {
            fprintf(stderr, "[proxy] send err: %s\n", nng_strerror(rc));
        }
    }

    nng_close(pub);
    nng_close(sub);
    return 0;
}
