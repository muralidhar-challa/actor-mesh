/* client.c — send a user_message tuple, print agent_response
 *
 * Usage: ./client "list all engineers"
 *
 * Uses a unique session_id per invocation to skip stale responses.
 */

#define _GNU_SOURCE

#include "actor_tuple.h"
#include "actor_uuid.h"

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <zmq.h>

#define BUS_PUB    "tcp://localhost:5557"
#define BUS_SUB    "tcp://localhost:5556"
#define TIMEOUT_MS 120000

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s \"your query\"\n", argv[0]);
        return 1;
    }

    char session_id[32];
    snprintf(session_id, sizeof(session_id), "cli-%ld", (long)time(NULL));

    const char* query = argv[1];
    char payload[4096];
    int plen = snprintf(payload, sizeof(payload),
        "{\"type\":\"user_message\",\"query\":\"%s\",\"session_id\":\"%s\"}",
        query, session_id);

    void* ctx = zmq_ctx_new();

    void* sub = zmq_socket(ctx, ZMQ_SUB);
    zmq_connect(sub, BUS_SUB);
    zmq_setsockopt(sub, ZMQ_SUBSCRIBE, "agent_response", 14);

    void* pub = zmq_socket(ctx, ZMQ_PUB);
    zmq_connect(pub, BUS_PUB);

    struct timespec ts = { .tv_sec = 0, .tv_nsec = 500000000 };
    nanosleep(&ts, NULL);

    actor_header_t hdr;
    actor_tuple_init(&hdr, "user_message", "client", NULL, NULL, (uint32_t)plen);
    actor_uuid_gen(hdr.id);

    zmq_send(pub, &hdr,    sizeof(actor_header_t), ZMQ_SNDMORE);
    zmq_send(pub, payload, (size_t)plen,            0);
    fprintf(stderr, "[client] sent: %s\n", query);

    zmq_pollitem_t items[1];
    items[0].socket = sub;
    items[0].events = ZMQ_POLLIN;

    while (1) {
        if (zmq_poll(items, 1, TIMEOUT_MS) <= 0) {
            fprintf(stderr, "[client] timeout — no response\n");
            return 1;
        }

        zmq_msg_t hdr_msg, pay_msg;
        zmq_msg_init(&hdr_msg);
        zmq_msg_recv(&hdr_msg, sub, 0);
        zmq_msg_init(&pay_msg);
        if (zmq_msg_more(&hdr_msg))
            zmq_msg_recv(&pay_msg, sub, 0);

        char* body = (char*)zmq_msg_data(&pay_msg);
        int   blen = (int)zmq_msg_size(&pay_msg);

        char needle[64];
        snprintf(needle, sizeof(needle), "\"session_id\":\"%s\"", session_id);
        if (memmem(body, (size_t)blen, needle, strlen(needle)) == NULL) {
            zmq_msg_close(&hdr_msg);
            zmq_msg_close(&pay_msg);
            continue;
        }

        FILE* jq = popen("jq -r '.answer' | ~/go/bin/glow - 2>/dev/null || jq -r '.answer'", "w");
        if (jq) {
            fwrite(body, 1, (size_t)blen, jq);
            pclose(jq);
        } else {
            printf("%.*s\n", blen, body);
        }

        zmq_msg_close(&hdr_msg);
        zmq_msg_close(&pay_msg);
        break;
    }

    zmq_close(pub);
    zmq_close(sub);
    zmq_ctx_destroy(ctx);
    return 0;
}
