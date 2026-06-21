// proxy.c — NNG pub/sub mesh bridge + HTTP endpoint for browsers
//
// Actors connect via pub/sub (tcp://5556,5557).
// Browsers POST binary frames to tcp://8082 → gets result back.
//
// Env:
//   PROXY_SUB_BIND      where publishers dial  (default tcp://*:5557)
//   PROXY_PUB_BIND      where subscribers dial  (default tcp://*:5556)
//   PROXY_HTTP_BIND     browser HTTP endpoint   (default tcp://*:8082)
//   PROXY_ID            proxy identity          (default "proxy")

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <errno.h>

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
    char buf[256];
    const char *start = urls, *p;
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
            } else {
                fprintf(stderr, "[proxy] listen %s: ok\n", buf);
            }
            if (*p == '\0') return 0;
            start = p + 1; while (*start == ' ') start++; p = start - 1;
        }
    }
}

/* Minimal HTTP handler: POST binary frame → forward → wait → reply */
static void http_handle(int fd) {
    char buf[65536];
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    if (n <= 0) { close(fd); return; }
    buf[n] = '\0';

    /* Parse Content-Length */
    char *cl = strstr(buf, "Content-Length:");
    if (!cl) { dprintf(fd, "HTTP/1.0 400\r\n\r\n"); close(fd); return; }
    int body_len = atoi(cl + 15);
    char *body_start = strstr(buf, "\r\n\r\n");
    if (!body_start) { dprintf(fd, "HTTP/1.0 400\r\n\r\n"); close(fd); return; }
    body_start += 4;

    /* Read remaining body if needed */
    ssize_t in_buf = n - (body_start - buf);
    if (in_buf < body_len) {
        ssize_t r = read(fd, body_start + in_buf, body_len - in_buf);
        if (r < 0) { close(fd); return; }
    }
    uint8_t *body = (uint8_t*)body_start;

    if (body_len < 256) {
        dprintf(fd, "HTTP/1.0 200\r\nContent-Type: application/json\r\n\r\n"
                "{\"ok\":false,\"error\":\"short\"}");
        close(fd); return;
    }

    /* Extract topic */
    char topic[33] = {0};
    memcpy(topic, body, 32);
    char result_topic[64];
    snprintf(result_topic, sizeof(result_topic), "%s.result", topic);

    /* Subscribe FIRST (before forward, to not miss response) */
    nng_socket rsub;
    int rc = nng_sub0_open(&rsub);
    if (rc != 0) {
        dprintf(fd, "HTTP/1.0 500\r\n\r\n");
        close(fd); return;
    }
    nng_dial(rsub, "tcp://127.0.0.1:5556", NULL, 0);
    nng_socket_set(rsub, NNG_OPT_SUB_SUBSCRIBE, result_topic, strlen(result_topic)+1);
    nng_socket_set_ms(rsub, NNG_OPT_RECVTIMEO, 5000);

    /* Publish via inproc transport to g_pub */
    nng_socket ipub;
    int rc2 = nng_pub0_open(&ipub);
    fprintf(stderr, "[proxy] http: pub open=%d\n", rc2);
    if (rc2 == 0) {
        rc2 = nng_dial(ipub, "inproc://http-relay", NULL, 0);
        fprintf(stderr, "[proxy] http: inproc dial=%d\n", rc2);
        if (rc2 == 0) {
            rc2 = nng_send(ipub, body, body_len, 0);
            fprintf(stderr, "[proxy] http: send %zu bytes = %d\n", body_len, rc2);
        }
        nng_close(ipub);
    }
    if (rc2 != 0) {
        dprintf(fd, "HTTP/1.0 200\r\nContent-Type: application/json\r\n\r\n"
                "{\"ok\":false,\"error\":\"pub fail\"}");
        nng_close(rsub);
        close(fd); return;
    }

    nng_msg *res = NULL;
    rc = nng_recvmsg(rsub, &res, 0);
    nng_close(rsub);

    if (rc == 0) {
        void *rbody = nng_msg_body(res);
        size_t rlen = nng_msg_len(res);
        dprintf(fd, "HTTP/1.0 200\r\nContent-Type: application/octet-stream\r\n"
                "Content-Length: %zu\r\n\r\n", rlen);
        write(fd, rbody, rlen);
        nng_msg_free(res);
    } else {
        dprintf(fd, "HTTP/1.0 200\r\nContent-Type: application/json\r\n\r\n"
                "{\"ok\":false,\"error\":\"timeout\"}");
    }
    close(fd);
}

int main(void) {
    const char* sub_bind  = getenv("PROXY_SUB_BIND")  ?: "tcp://*:5557";
    const char* pub_bind  = getenv("PROXY_PUB_BIND")  ?: "tcp://*:5556";
    const char* http_bind = getenv("PROXY_HTTP_BIND");
    const char* proxy_id  = getenv("PROXY_ID")        ?: "proxy";
    int hb_ms = getenv("PROXY_HEARTBEAT_MS") ? atoi(getenv("PROXY_HEARTBEAT_MS")) : 5000;

    signal(SIGTERM, on_signal);
    signal(SIGINT,  on_signal);

    nng_socket sub;
    int rc;

    if ((rc = nng_sub0_open(&sub)) != 0) {
        fprintf(stderr, "[proxy] sub open: %s\n", nng_strerror(rc));
        return 1;
    }
    if ((rc = nng_pub0_open(&g_pub)) != 0) {
        fprintf(stderr, "[proxy] pub open: %s\n", nng_strerror(rc));
        nng_close(sub); return 1;
    }
    listen_all(sub, sub_bind);
    nng_socket_set(sub, NNG_OPT_SUB_SUBSCRIBE, "", 0);
    listen_all(g_pub, pub_bind);
    /* Additional inproc listener on g_pub for HTTP handler relay */
    nng_listen(g_pub, "inproc://http-relay", NULL, 0);
    nng_socket_set_ms(sub, NNG_OPT_RECVTIMEO, 100);

    /* HTTP listener */
    int http_fd = -1;
    if (http_bind) {
        struct sockaddr_in6 addr = {0};
        addr.sin6_family = AF_INET6;
        addr.sin6_port = htons(8082);
        addr.sin6_addr = in6addr_any;
        http_fd = socket(AF_INET6, SOCK_STREAM, 0);
        int v = 1;
        setsockopt(http_fd, SOL_SOCKET, SO_REUSEADDR, &v, sizeof(v));
        if (bind(http_fd, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
            listen(http_fd, 8);
            fprintf(stderr, "[proxy] HTTP listening on tcp://0.0.0.0:8082\n");
        } else {
            close(http_fd); http_fd = -1;
        }
    }

    fprintf(stderr, "[proxy] id=%s sub=%s pub=%s hb=%d\n", proxy_id, sub_bind, pub_bind, hb_ms);

    int64_t last_hb = 0;

    while (!g_stop) {
        if (hb_ms > 0) {
            struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
            int64_t now_ms = ts.tv_sec * 1000LL + ts.tv_nsec / 1000000LL;
            if (now_ms - last_hb >= hb_ms) { emit_heartbeat(proxy_id); last_hb = now_ms; }
        }

        /* Check HTTP connections (non-blocking) */
        if (http_fd >= 0) {
            struct sockaddr_in6 ca; socklen_t cl = sizeof(ca);
            int cfd = accept(http_fd, (struct sockaddr*)&ca, &cl);
            if (cfd >= 0) http_handle(cfd);
        }

        /* Mesh sub → pub relay */
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
    nng_close(g_pub);
    nng_close(sub);
    return 0;
}
