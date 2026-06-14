#include "state.h"
#include "net.h"
#include <SDL2/SDL.h>
#include <nng/nng.h>
#include <nng/protocol/pubsub0/pub.h>
#include <nng/protocol/pubsub0/sub.h>
#include <time.h>
#include <stdlib.h>

static nng_socket g_pub_sock;
static nng_socket g_sub_sock;
static uint8_t    g_client_id[16];

/* ── UUIDv7 ──────────────────────────────────────────────────────── */

static void uuid7(uint8_t d[16]) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    uint64_t ms = (uint64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
    for (int i = 0; i < 6; i++)  d[i] = (uint8_t)(ms >> (40 - i * 8));
    for (int i = 6; i < 16; i++) d[i] = (uint8_t)(rand() % 256);
    d[6] = (d[6] & 0x0f) | 0x70;
    d[8] = (d[8] & 0x3f) | 0x80;
}

/* ── net_init ────────────────────────────────────────────────────── */

void net_init(void) {
    const char *sub = getenv("ACTOR_BUS_SUB") ?: "tcp://127.0.0.1:5556";
    const char *pub = getenv("ACTOR_BUS_PUB") ?: "tcp://127.0.0.1:5557";

    nng_pub0_open(&g_pub_sock);
    nng_sub0_open(&g_sub_sock);
    nng_dial(g_pub_sock, pub, NULL, 0);
    nng_dial(g_sub_sock, sub, NULL, 0);
    nng_socket_set(g_sub_sock, "sub:subscribe", "", 0); /* subscribe to all */
    nng_socket_set_ms(g_sub_sock, "recv-timeout", 10);

    uuid7(g_client_id);
    g_connected = true;
}

/* ── net_send_frame ──────────────────────────────────────────────── */

void net_send_frame(const char *topic, const uint8_t *payload, size_t plen) {
    uint8_t frame[256 + 4096];
    memset(frame, 0, 256);

    int tlen = (int)strlen(topic);
    if (tlen > 31) tlen = 31;
    memcpy(frame, topic, tlen);

    uuid7(frame + 32);
    memcpy(frame + 48, g_client_id, 16);  /* correlation_id = client session */
    memcpy(frame + 80, "desktop", 7);     /* origin */

    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    int64_t ns = ts.tv_sec * 1000000000LL + ts.tv_nsec;
    memcpy(frame + 112, &ns, 8);

    if (plen > 4096) plen = 4096;
    uint32_t pl32 = (uint32_t)plen;
    memcpy(frame + 132, &pl32, 4);
    memcpy(frame + 256, payload, plen);

    nng_send(g_pub_sock, frame, 256 + plen, 0);
}

/* ── net_poll ────────────────────────────────────────────────────── */

void net_poll(void) {
    for (;;) {
        nng_msg *m = NULL;
        if (nng_recvmsg(g_sub_sock, &m, 0) != 0) break;

        const uint8_t *body = (const uint8_t *)nng_msg_body(m);
        size_t         blen = nng_msg_len(m);

        if (blen >= 256) {
            char topic[33] = {0};
            memcpy(topic, body, 32);

            /* ── heartbeat: update actor registry ─────────────────── */
            if (strncmp(topic, "heartbeat", 9) == 0) {
                const char *payload = (const char *)(body + 256);
                const char *p = strstr(payload, "\"id\":\"");
                if (p) {
                    p += 6;
                    char id[32] = {0};
                    int  ai = 0;
                    while (*p && *p != '"' && ai < 31) id[ai++] = *p++;

                    bool found = false;
                    for (int i = 0; i < g_nactors; i++) {
                        if (strcmp(g_actors[i].id, id) == 0) {
                            g_actors[i].last_hb_ticks = SDL_GetTicks();
                            const char *q;
                            if ((q = strstr(payload, "\"inbox\":")))
                                sscanf(q + 8,  "%d", &g_actors[i].inbox);
                            if ((q = strstr(payload, "\"outbox\":")))
                                sscanf(q + 9, "%d", &g_actors[i].outbox);
                            found = true;
                            break;
                        }
                    }
                    if (!found && g_nactors < MAX_ACTORS) {
                        Actor *a = &g_actors[g_nactors++];
                        memset(a, 0, sizeof(*a));
                        memcpy(a->id, id, 31); a->id[31] = '\0';
                        a->last_hb_ticks = SDL_GetTicks();
                    }
                }
            }

            /* ── all messages: append to bus ring buffer ──────────── */
            int slot = g_nevents % MAX_EVENTS;
            TupleEvent *ev = &g_events[slot];
            memset(ev, 0, sizeof(*ev));

            memcpy(ev->topic, body, 32);      ev->topic[32]  = '\0';
            memcpy(ev->origin, body + 80, 32); ev->origin[32] = '\0';

            const uint8_t *id_b = body + 32;
            snprintf(ev->id_hex, 9, "%02x%02x%02x%02x",
                     id_b[0], id_b[1], id_b[2], id_b[3]);

            const uint8_t *co_b = body + 48;
            snprintf(ev->corr_hex, 9, "%02x%02x%02x%02x",
                     co_b[0], co_b[1], co_b[2], co_b[3]);

            const uint8_t *ca_b = body + 64;
            snprintf(ev->caus_hex, 9, "%02x%02x%02x%02x",
                     ca_b[0], ca_b[1], ca_b[2], ca_b[3]);

            if (blen > 256) {
                const char *pb = (const char *)(body + 256);
                uint32_t plen = 0;
                memcpy(&plen, body + 132, 4);
                int avail = (int)(blen - 256);
                ev->payload_len = (int)plen < avail ? (int)plen : avail;

                int prev_len = ev->payload_len < 128 ? ev->payload_len : 128;
                for (int i = 0; i < prev_len; i++) {
                    unsigned char c = (unsigned char)pb[i];
                    ev->preview[i] = (c >= 0x20 && c < 0x7f) ? c : '.';
                }
                ev->preview[prev_len] = '\0';

                int full_len = ev->payload_len < 4096 ? ev->payload_len : 4096;
                for (int i = 0; i < full_len; i++) {
                    unsigned char c = (unsigned char)pb[i];
                    ev->full[i] = (c >= 0x20 && c < 0x7f) ? c : '.';
                }
                ev->full[full_len] = '\0';
            }

            ev->received_ticks = SDL_GetTicks();
            g_nevents++;
        }

        nng_msg_free(m);
    }
}
