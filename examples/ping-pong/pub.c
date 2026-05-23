/* pub.c — minimal one-shot publisher for the mesh
 *
 * Usage: ./pub topic payload
 *
 * Sends a single tuple on the given topic and exits.
 * Useful for testing and scripting.
 */
#include "actor_tuple.h"
#include "actor_uuid.h"

#include <stdio.h>
#include <string.h>
#include <nng/nng.h>
#include <nng/protocol/pubsub0/pub.h>

#define BUS_PUB "tcp://127.0.0.1:5557"

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s topic payload\n", argv[0]);
        return 1;
    }

    const char *topic   = argv[1];
    const char *payload = argv[2];
    size_t      plen    = strlen(payload);

    nng_socket pub;
    int rc;
    if ((rc = nng_pub0_open(&pub)) != 0) {
        fprintf(stderr, "pub: nng_pub0_open: %s\n", nng_strerror(rc));
        return 1;
    }
    if ((rc = nng_dial(pub, BUS_PUB, NULL, 0)) != 0) {
        fprintf(stderr, "pub: dial %s: %s\n", BUS_PUB, nng_strerror(rc));
        return 1;
    }

    /* warm-up */
    struct timespec warm = { .tv_sec = 0, .tv_nsec = 200000000 };
    nanosleep(&warm, NULL);

    actor_header_t        hdr;
    static uint8_t        frame[sizeof(actor_header_t) + 4096];

    actor_tuple_init(&hdr, topic, "pub", NULL, NULL, (uint32_t)plen);
    actor_uuid_gen(hdr.id);

    memcpy(frame, &hdr, sizeof(actor_header_t));
    memcpy(frame + sizeof(actor_header_t), payload, plen);

    rc = nng_send(pub, frame, sizeof(actor_header_t) + plen, 0);
    if (rc != 0) {
        fprintf(stderr, "pub: send: %s\n", nng_strerror(rc));
        return 1;
    }

    fprintf(stderr, "[pub] sent topic=%s payload=%s\n", topic, payload);
    nng_close(pub);
    return 0;
}
