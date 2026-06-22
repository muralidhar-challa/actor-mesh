/* sub.c — minimal one-shot subscriber for the mesh
 * Usage: ./sub topic timeout_ms
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <nng/nng.h>
#include <nng/protocol/pubsub0/sub.h>

#define BUS_SUB "tcp://127.0.0.1:5556"

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s topic timeout_ms\n", argv[0]);
        return 1;
    }
    const char *topic = argv[1];
    int timeout_ms = atoi(argv[2]);

    nng_socket sub;
    int rc;
    if ((rc = nng_sub0_open(&sub)) != 0) { fprintf(stderr, "open: %s\n", nng_strerror(rc)); return 1; }
    nng_socket_set(sub, NNG_OPT_SUB_SUBSCRIBE, topic, strlen(topic));
    nng_socket_set_ms(sub, NNG_OPT_RECVTIMEO, timeout_ms);
    if ((rc = nng_dial(sub, BUS_SUB, NULL, 0)) != 0) { fprintf(stderr, "dial: %s\n", nng_strerror(rc)); return 1; }

    nng_msg *msg;
    rc = nng_recvmsg(sub, &msg, 0);
    if (rc != 0) { fprintf(stderr, "recv: %s\n", nng_strerror(rc)); return 1; }
    fwrite(nng_msg_body(msg), 1, nng_msg_len(msg), stdout);
    printf("\n");
    nng_msg_free(msg);
    return 0;
}
