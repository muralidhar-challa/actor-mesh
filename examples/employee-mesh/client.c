/* client.c — send a user_message tuple, print agent_response
 *
 * Usage: ./client "list all engineers"
 *
 * Uses a unique session_id per invocation to skip stale responses.
 * Mesh wire format: mpack (MessagePack).
 */

#define _GNU_SOURCE

#include "actor_tuple.h"
#include "actor_uuid.h"
#include "mpack.h"

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <zmq.h>

#define BUS_PUB      "tcp://localhost:5557"
#define BUS_SUB      "tcp://localhost:5556"
#define TIMEOUT_MS   120000
#define MAX_OUT      4096
#define MAX_ANSWER   (256 * 1024)

static char g_out[MAX_OUT];
static char g_answer[MAX_ANSWER];
static char g_resp_session[64];

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s \"your query\"\n", argv[0]);
        return 1;
    }

    char session_id[32];
    snprintf(session_id, sizeof(session_id), "cli-%ld", (long)time(NULL));

    const char* query = argv[1];

    /* build mpack payload: {type, query, session_id} */
    mpack_writer_t pw;
    mpack_writer_init(&pw, g_out, sizeof(g_out));
    mpack_start_map(&pw, 3);
    mpack_write_cstr(&pw, "type");       mpack_write_cstr(&pw, "user_message");
    mpack_write_cstr(&pw, "query");      mpack_write_cstr(&pw, query);
    mpack_write_cstr(&pw, "session_id"); mpack_write_cstr(&pw, session_id);
    mpack_finish_map(&pw);
    size_t plen = mpack_writer_buffer_used(&pw);
    mpack_writer_destroy(&pw);

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

    zmq_send(pub, &hdr,   sizeof(actor_header_t), ZMQ_SNDMORE);
    zmq_send(pub, g_out,  plen,                   0);
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

        const char* body = (const char*)zmq_msg_data(&pay_msg);
        size_t      blen = zmq_msg_size(&pay_msg);

        /* decode mpack response to check session_id */
        g_answer[0] = g_resp_session[0] = '\0';

        mpack_reader_t r;
        mpack_reader_init_data(&r, body, blen);

        static const char* rkeys[] = { "type", "answer", "query", "session_id" };
        bool rfound[4] = { false, false, false, false };

        uint32_t rsz = mpack_expect_map_max(&r, 8);
        for (uint32_t i = 0; i < rsz && mpack_reader_error(&r) == mpack_ok; i++) {
            switch (mpack_expect_key_cstr(&r, rkeys, rfound, 4)) {
                case 0: mpack_discard(&r); break;
                case 1: mpack_expect_cstr(&r, g_answer,      sizeof(g_answer));      break;
                case 2: mpack_discard(&r); break;
                case 3: mpack_expect_cstr(&r, g_resp_session, sizeof(g_resp_session)); break;
                default: mpack_discard(&r); break;
            }
        }
        mpack_done_map(&r);
        mpack_reader_destroy(&r);

        zmq_msg_close(&hdr_msg);
        zmq_msg_close(&pay_msg);

        if (strcmp(g_resp_session, session_id) != 0)
            continue;

        FILE* glow = popen("glow - 2>/dev/null || cat", "w");
        if (glow) {
            fputs(g_answer, glow);
            fputc('\n', glow);
            pclose(glow);
        } else {
            printf("%s\n", g_answer);
        }
        break;
    }

    zmq_close(pub);
    zmq_close(sub);
    zmq_ctx_destroy(ctx);
    return 0;
}
