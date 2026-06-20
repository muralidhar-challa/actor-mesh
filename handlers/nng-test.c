/* nng-test.c — send a properly formatted test tuple through the proxy
 *
 * Usage: /handlers/nng-test <pub_url> <sub_url> <topic>
 *
 * Publishes a test tuple on <pub_url> then tries to receive
 * the echo result on <sub_url>.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <nng/nng.h>
#include <nng/protocol/pubsub0/pub.h>
#include <nng/protocol/pubsub0/sub.h>

static void uuid4(char buf[33]) {
    FILE *f = fopen("/dev/urandom", "rb");
    if (!f) { memset(buf, '0', 32); buf[32] = 0; return; }
    unsigned char u[16];
    fread(u, 1, 16, f);
    fclose(f);
    u[6] = (u[6] & 0x0f) | 0x40;
    u[8] = (u[8] & 0x3f) | 0x80;
    for (int i = 0; i < 16; i++) sprintf(buf + i*2, "%02x", u[i]);
    buf[32] = 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: nng-test <pub_url> <sub_url> [topic]\n");
        return 1;
    }
    const char *pub_url = argv[1];
    const char *sub_url = argv[2];
    const char *topic   = argc > 3 ? argv[3] : "test.echo";
    const char *result_topic = "test.echo.result";

    /* Build a proper actor_header_t + payload */
    unsigned char frame[2048];
    memset(frame, 0, 256);

    /* topic[32] */
    snprintf((char*)frame, 32, "%s", topic);

    /* id[16] */
    char id_str[33];
    uuid4(id_str);
    for (int i = 0; i < 16; i++) {
        unsigned int b;
        sscanf(id_str + i*2, "%2x", &b);
        frame[32 + i] = (unsigned char)b;
    }

    /* origin[32] */
    snprintf((char*)frame + 80, 32, "nng-test");

    /* emitted_at */
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    int64_t now = ts.tv_sec * 1000000000LL + ts.tv_nsec;
    memcpy(frame + 112, &now, 8);

    /* payload_len */
    const char *payload = "{\"test\":\"hello-mesh\"}";
    uint32_t plen = (uint32_t)strlen(payload);
    memcpy(frame + 136, &plen, 4);

    memcpy(frame + 256, payload, plen);
    size_t total = 256 + plen;

    fprintf(stderr, "[nng-test] sending %zu bytes on topic '%s' to %s\n",
            total, topic, pub_url);

    /* Open pub socket and send */
    nng_socket pub;
    int rc;
    if ((rc = nng_pub0_open(&pub)) != 0) {
        fprintf(stderr, "[nng-test] pub open: %s\n", nng_strerror(rc));
        return 1;
    }
    if ((rc = nng_dial(pub, pub_url, NULL, 0)) != 0) {
        fprintf(stderr, "[nng-test] pub dial %s: %s\n", pub_url, nng_strerror(rc));
        nng_close(pub);
        return 1;
    }

    /* NNG pub needs a moment for the dial to establish */
    usleep(100000);

    if ((rc = nng_send(pub, frame, total, 0)) != 0) {
        fprintf(stderr, "[nng-test] send: %s\n", nng_strerror(rc));
    } else {
        fprintf(stderr, "[nng-test] sent ok\n");
    }
    nng_close(pub);

    /* Now subscribe and try to receive the echo result */
    nng_socket sub;
    if ((rc = nng_sub0_open(&sub)) != 0) {
        fprintf(stderr, "[nng-test] sub open: %s\n", nng_strerror(rc));
        return 1;
    }
    if ((rc = nng_dial(sub, sub_url, NULL, 0)) != 0) {
        fprintf(stderr, "[nng-test] sub dial %s: %s\n", sub_url, nng_strerror(rc));
        nng_close(sub);
        return 1;
    }
    if ((rc = nng_socket_set(sub, NNG_OPT_SUB_SUBSCRIBE, result_topic,
                             strlen(result_topic) + 1)) != 0) {
        fprintf(stderr, "[nng-test] subscribe: %s\n", nng_strerror(rc));
        nng_close(sub);
        return 1;
    }
    nng_socket_set_ms(sub, NNG_OPT_RECVTIMEO, 3000);

    fprintf(stderr, "[nng-test] waiting for result on '%s'...\n", result_topic);

    nng_msg *msg = NULL;
    rc = nng_recvmsg(sub, &msg, 0);
    if (rc == NNG_ETIMEDOUT) {
        fprintf(stderr, "[nng-test] timeout — no result received\n");
    } else if (rc != 0) {
        fprintf(stderr, "[nng-test] recv error: %s\n", nng_strerror(rc));
    } else {
        void *body = nng_msg_body(msg);
        size_t len = nng_msg_len(msg);
        fprintf(stderr, "[nng-test] received %zu bytes\n", len);
        fwrite(body, 1, len, stderr);
        fprintf(stderr, "\n");
        nng_msg_free(msg);
    }
    nng_close(sub);

    return 0;
}
