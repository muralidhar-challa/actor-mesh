/* client.c — send a user_message tuple, print agent_response
 *
 * Usage: ./client "list all engineers"
 *
 * Matches responses by correlation_id in the header (same as TUI).
 * Mesh wire format: mpack (MessagePack).
 */

#define _GNU_SOURCE

#include "actor_tuple.h"
#include "actor_uuid.h"
#include "mpack.h"

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <nng/nng.h>
#include <nng/protocol/pubsub0/pub.h>
#include <nng/protocol/pubsub0/sub.h>

#define BUS_PUB      "tcp://localhost:5557"
#define BUS_SUB      "tcp://localhost:5556"
#define TIMEOUT_MS   600000
#define MAX_OUT      4096
#define MAX_ANSWER   (256 * 1024)

static char    g_out[MAX_OUT];
static char    g_answer[MAX_ANSWER];
static uint8_t g_corr_id[16];

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s \"your query\"\n", argv[0]);
        return 1;
    }

    /* generate a correlation_id for this session (same approach as TUI) */
    actor_uuid_gen(g_corr_id);

    const char* query = argv[1];

    /* build mpack payload: {type, query} */
    mpack_writer_t pw;
    mpack_writer_init(&pw, g_out, sizeof(g_out));
    mpack_start_map(&pw, 2);
    mpack_write_cstr(&pw, "type");   mpack_write_cstr(&pw, "user_message");
    mpack_write_cstr(&pw, "query");  mpack_write_cstr(&pw, query);
    mpack_finish_map(&pw);
    size_t plen = mpack_writer_buffer_used(&pw);
    mpack_writer_destroy(&pw);

    nng_socket sub, pub;
    int        rc;

    if ((rc = nng_sub0_open(&sub)) != 0) {
        fprintf(stderr, "[client] nng_sub0_open: %s\n", nng_strerror(rc));
        return 1;
    }
    if ((rc = nng_dial(sub, BUS_SUB, NULL, 0)) != 0) {
        fprintf(stderr, "[client] sub dial: %s\n", nng_strerror(rc));
        return 1;
    }
    /* subscribe to agent_response — null-terminated for exact match */
    nng_socket_set(sub, NNG_OPT_SUB_SUBSCRIBE, "agent_response", 15);

    if ((rc = nng_pub0_open(&pub)) != 0) {
        fprintf(stderr, "[client] nng_pub0_open: %s\n", nng_strerror(rc));
        return 1;
    }
    if ((rc = nng_dial(pub, BUS_PUB, NULL, 0)) != 0) {
        fprintf(stderr, "[client] pub dial: %s\n", nng_strerror(rc));
        return 1;
    }

    /* warm-up: let NNG establish connections before sending */
    struct timespec ts = { .tv_sec = 0, .tv_nsec = 500000000 };
    nanosleep(&ts, NULL);

    actor_header_t hdr;
    actor_tuple_init(&hdr, "user_message", "client", g_corr_id, NULL, (uint32_t)plen);
    actor_uuid_gen(hdr.id);

    /* assemble header + payload into one contiguous message */
    size_t frame_len = sizeof(actor_header_t) + plen;
    static uint8_t frame_buf[sizeof(actor_header_t) + MAX_OUT];
    memcpy(frame_buf, &hdr, sizeof(actor_header_t));
    memcpy(frame_buf + sizeof(actor_header_t), g_out, plen);
    nng_send(pub, frame_buf, frame_len, 0);

    fprintf(stderr, "[client] sent: %s\n", query);

    /* set recv timeout */
    nng_socket_set_ms(sub, NNG_OPT_RECVTIMEO, TIMEOUT_MS);

    while (1) {
        nng_msg* msg = NULL;
        rc = nng_recvmsg(sub, &msg, 0);
        if (rc == NNG_ETIMEDOUT) {
            fprintf(stderr, "[client] timeout — no response\n");
            return 1;
        }
        if (rc != 0) {
            fprintf(stderr, "[client] recv error: %s\n", nng_strerror(rc));
            return 1;
        }

        void*  body     = nng_msg_body(msg);
        size_t body_len = nng_msg_len(msg);

        if (body_len < sizeof(actor_header_t)) {
            nng_msg_free(msg);
            continue;
        }

        const actor_header_t* hdr_in  = (const actor_header_t*)body;
        const char*           payload_ptr = (const char*)body + sizeof(actor_header_t);
        size_t                plen2  = body_len - sizeof(actor_header_t);

        /* match by correlation_id (same as TUI) */
        if (memcmp(hdr_in->correlation_id, g_corr_id, 16) != 0) {
            nng_msg_free(msg);
            continue;
        }

        /* decode mpack response */
        g_answer[0] = '\0';

        mpack_reader_t r;
        mpack_reader_init_data(&r, payload_ptr, plen2);

        static const char* rkeys[] = { "type", "answer" };
        bool rfound[2] = { false, false };

        uint32_t rsz = mpack_expect_map_max(&r, 8);
        for (uint32_t i = 0; i < rsz && mpack_reader_error(&r) == mpack_ok; i++) {
            switch (mpack_expect_key_cstr(&r, rkeys, rfound, 2)) {
                case 0: mpack_discard(&r); break;
                case 1: mpack_expect_cstr(&r, g_answer, sizeof(g_answer)); break;
                default: mpack_discard(&r); break;
            }
        }
        mpack_done_map(&r);
        mpack_reader_destroy(&r);

        nng_msg_free(msg);

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

    nng_close(pub);
    nng_close(sub);
    return 0;
}
