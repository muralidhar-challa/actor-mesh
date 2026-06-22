// proxy.c — NNG mesh bridge + HTTP endpoint for browsers
//
// Backend: tcp://5556 (pub), tcp://5557 (sub)
// Browser: http://8082 POST binary frame → forked child → NNG mesh
//
// HTTP requests are forked — main loop never blocks.

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/wait.h>
#include <nng/nng.h>
#include <nng/protocol/pubsub0/pub.h>
#include <nng/protocol/pubsub0/sub.h>
#include "actor_tuple.h"
#include "actor_uuid.h"

static volatile sig_atomic_t g_stop = 0;
static void on_signal(int s) { (void)s; g_stop = 1; }
static nng_socket g_pub;

static void emit_heartbeat(const char* id) {
    char payload[128];
    size_t plen = snprintf(payload, sizeof(payload), "{\"id\":\"%s\"}", id);
    actor_header_t hdr;
    actor_tuple_init(&hdr, "heartbeat", id, NULL, NULL, (uint32_t)plen);
    actor_uuid_gen(hdr.id);
    uint8_t frame[sizeof(actor_header_t) + 128];
    memcpy(frame, &hdr, sizeof(actor_header_t));
    memcpy(frame + sizeof(actor_header_t), payload, plen);
    nng_send(g_pub, frame, sizeof(actor_header_t) + plen, 0);
}

static int listen_all(nng_socket sock, const char *urls) {
    char buf[256]; const char *start = urls, *p;
    for (p = urls; ; p++) {
        if (*p == ',' || *p == '\0') {
            size_t len = (size_t)(p - start);
            while (len > 0 && start[len-1] == ' ') len--;
            if (len >= sizeof(buf)) len = sizeof(buf)-1;
            memcpy(buf, start, len); buf[len] = '\0';
            int rc = nng_listen(sock, buf, NULL, 0);
            if (rc != 0) {
                fprintf(stderr, "[proxy] listen %s: %s\n", buf, nng_strerror(rc));
                if (*p == '\0') return rc;
            } else fprintf(stderr, "[proxy] listen %s: ok\n", buf);
            if (*p == '\0') return 0;
            start = p + 1; while (*start == ' ') start++; p = start - 1;
        }
    }
}

static void http_handle(int fd) {
    char buf[65536];
    ssize_t n = read(fd, buf, sizeof(buf)-1);
    if (n <= 0) { close(fd); _exit(0); }
    buf[n] = 0;
    char *cl = strstr(buf, "Content-Length:");
    char *body = strstr(buf, "\r\n\r\n");
    if (!cl || !body) { dprintf(fd, "HTTP/1.0 400\r\n\r\n"); close(fd); _exit(0); }
    body += 4;
    int blen = atoi(cl+15);
    ssize_t inbuf = n - (body - buf);
    while (inbuf < blen) { ssize_t r = read(fd, body+inbuf, blen-inbuf); if (r <= 0) break; inbuf += r; }
    if (blen < 256) { dprintf(fd, "HTTP/1.0 200\r\n\r\n{\"ok\":false}"); close(fd); _exit(0); }

    /* NNG is NOT fork-safe — must use nngcat via popen */
    /* 1. Publish via nngcat */
    FILE *p = popen("nngcat --pub --dial tcp://127.0.0.1:5557 --data - 2>/dev/null", "w");
    if (p) { fwrite(body, 1, blen, p); pclose(p); }

    /* 2. Subscribe to result via nngcat */
    char rt[64]; snprintf(rt, sizeof(rt), "%.32s.result", body);
    char cmd[512];
    snprintf(cmd, sizeof(cmd),
        "nngcat --sub --dial tcp://127.0.0.1:5556 --subscribe %s "
        "--count 1 --recv-timeout 5000 2>/dev/null", rt);
    FILE *s = popen(cmd, "r");
    if (s) {
        char rbuf[65536];
        size_t rlen = fread(rbuf, 1, sizeof(rbuf), s);
        pclose(s);
        if (rlen > 0) {
            dprintf(fd, "HTTP/1.0 200\r\nContent-Type: application/octet-stream\r\nContent-Length: %zu\r\n\r\n", rlen);
            write(fd, rbuf, rlen);
        } else {
            dprintf(fd, "HTTP/1.0 200\r\nContent-Type: application/json\r\n\r\n{\"ok\":false,\"error\":\"empty\"}");
        }
    } else {
        dprintf(fd, "HTTP/1.0 200\r\nContent-Type: application/json\r\n\r\n{\"ok\":false,\"error\":\"popen\"}");
    }
    close(fd);
    _exit(0);
}

int main(void) {
    const char* sub_bind = getenv("PROXY_SUB_BIND") ?: "tcp://*:5557";
    const char* pub_bind = getenv("PROXY_PUB_BIND") ?: "tcp://*:5556";
    const char* proxy_id = getenv("PROXY_ID") ?: "proxy";
    int hb_ms = getenv("PROXY_HEARTBEAT_MS") ? atoi(getenv("PROXY_HEARTBEAT_MS")) : 5000;

    signal(SIGTERM, on_signal);
    signal(SIGINT,  on_signal);
    signal(SIGCHLD, SIG_IGN);

    nng_socket sub; int rc;
    if ((rc = nng_sub0_open(&sub)) != 0) {
        fprintf(stderr, "[proxy] sub open: %s\n", nng_strerror(rc)); return 1;
    }
    if ((rc = nng_pub0_open(&g_pub)) != 0) {
        fprintf(stderr, "[proxy] pub open: %s\n", nng_strerror(rc)); nng_close(sub); return 1;
    }
    listen_all(sub, sub_bind);
    nng_socket_set(sub, NNG_OPT_SUB_SUBSCRIBE, "", 0);
    listen_all(g_pub, pub_bind);
    nng_socket_set_ms(sub, NNG_OPT_RECVTIMEO, 100);

    int http_fd = socket(AF_INET6, SOCK_STREAM, 0);
    { int v=1; setsockopt(http_fd, SOL_SOCKET, SO_REUSEADDR, &v, sizeof(v)); }
    struct sockaddr_in6 addr = { .sin6_family = AF_INET6, .sin6_port = htons(8082), .sin6_addr = in6addr_any };
    if (bind(http_fd, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
        listen(http_fd, 8);
        fcntl(http_fd, F_SETFL, O_NONBLOCK); /* accept() must not block the mesh forwarding loop */
        fprintf(stderr, "[proxy] http on :8082\n");
    } else { close(http_fd); http_fd = -1; }

    fprintf(stderr, "[proxy] id=%s sub=%s pub=%s hb=%d\n", proxy_id, sub_bind, pub_bind, hb_ms);

    int64_t last_hb = 0;
    while (!g_stop) {
        if (hb_ms > 0) {
            struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
            int64_t now_ms = ts.tv_sec * 1000LL + ts.tv_nsec / 1000000LL;
            if (now_ms - last_hb >= hb_ms) { emit_heartbeat(proxy_id); last_hb = now_ms; }
        }
        /* Accept HTTP connections (non-blocking — falls through to mesh forwarding below) */
        if (http_fd >= 0) {
            int cfd = accept(http_fd, NULL, NULL);
            if (cfd >= 0) {
                char buf[65536];
                ssize_t n = read(cfd, buf, sizeof(buf)-1);
                if (n > 0) {
                    buf[n] = 0;
                    char *cl = strstr(buf, "Content-Length:");
                    char *body = strstr(buf, "\r\n\r\n");
                    if (cl && body) {
                        body += 4;
                        int blen = atoi(cl+15);
                        if (blen >= 256) {
                            /* Publish via nngcat */
                            FILE *p = popen("nngcat --pub --dial tcp://127.0.0.1:5557 --data - 2>/dev/null", "w");
                            if (p) { fwrite(body, 1, blen, p); pclose(p); }
                            /* Subscribe to result */
                            char rt[64], cmd[512];
                            snprintf(rt, sizeof(rt), "%.32s.result", body);
                            snprintf(cmd, sizeof(cmd),
                                "nngcat --sub --dial tcp://127.0.0.1:5556 --subscribe %s --count 1 --recv-timeout 5000 2>/dev/null", rt);
                            FILE *s = popen(cmd, "r");
                            if (s) {
                                char rbuf[65536];
                                size_t rlen = fread(rbuf, 1, sizeof(rbuf), s);
                                pclose(s);
                                dprintf(cfd, "HTTP/1.0 200\r\nContent-Type: application/octet-stream\r\nContent-Length: %zu\r\n\r\n", rlen);
                                if (rlen > 0) write(cfd, rbuf, rlen);
                            }
                        }
                    }
                }
                close(cfd);
            }
        }
        nng_msg* msg = NULL;
        rc = nng_recvmsg(sub, &msg, 0);
        if (rc == NNG_ETIMEDOUT) continue;
        if (rc != 0) { if (g_stop) break; continue; }
        rc = nng_sendmsg(g_pub, msg, 0);
        nng_msg_free(msg);
        if (rc != 0) fprintf(stderr, "[proxy] send err: %s\n", nng_strerror(rc));
    }

    fprintf(stderr, "[proxy] shutting down\n");
    if (http_fd >= 0) close(http_fd);
    nng_close(g_pub); nng_close(sub);
    return 0;
}
