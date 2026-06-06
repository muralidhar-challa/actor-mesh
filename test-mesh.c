#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <spawn.h>
#include <sys/wait.h>
#include <time.h>
#include <nng/nng.h>
#include <nng/protocol/pubsub0/pub.h>
#include <nng/protocol/pubsub0/sub.h>

static int failures = 0;
extern char **environ;
static void ms(int ms) { struct timespec t={ms/1000,(ms%1000)*1000000}; nanosleep(&t,NULL); }
static char sub_port[] = "tcp://127.0.0.1:55656";
static char pub_port[] = "tcp://127.0.0.1:55657";

#define TEST(n) printf("=== %s ===\n", n)
#define PASS() printf("  PASS\n")
#define FAIL(m) do { printf("  FAIL: %s\n", m); failures++; } while(0)
#define CHECK(c,m) do { if(c)PASS(); else FAIL(m); } while(0)

static pid_t spawn_proxy(void) {
    char *a[] = {"./mesh-proxy", NULL};
    char *e[] = {"PROXY_SUB_BIND=tcp://127.0.0.1:55657",
                 "PROXY_PUB_BIND=tcp://127.0.0.1:55656", NULL};
    pid_t p; posix_spawn(&p, a[0], NULL, NULL, a, e); return p;
}

static pid_t spawn_actor(const char *id, const char *topic, const char *result,
                          const char *handler, const char *lmdb) {
    char *a[] = {"./actor", NULL};
    char bus_sub[64], bus_pub[64], hb[32];
    snprintf(bus_sub, 64, "ACTOR_BUS_SUB=%s", sub_port);
    snprintf(bus_pub, 64, "ACTOR_BUS_PUB=%s", pub_port);
    char id_e[64], top[128], res[128], hdl[512], lmd[256];
    snprintf(id_e, 64, "ACTOR_ID=%s", id);
    snprintf(top, 128, "ACTOR_TOPIC=%s", topic);
    snprintf(res, 128, "ACTOR_RESULT_TOPIC=%s", result);
    snprintf(hdl, 512, "ACTOR_HANDLER=%s", handler);
    snprintf(lmd, 256, "ACTOR_LMDB_PATH=%s", lmdb);
    char *e[] = {bus_sub, bus_pub, "ACTOR_HEARTBEAT_MS=0", "ACTOR_RETRY_MAX=3",
                 id_e, top, res, hdl, lmd, NULL};
    pid_t p; posix_spawn(&p, a[0], NULL, NULL, a, e); return p;
}

static void send_msg(const char *topic, const uint8_t *payload, size_t plen) {
    nng_socket s; nng_pub0_open(&s);
    nng_dial(s, pub_port, NULL, 0); ms(50);
    uint8_t f[261]={0};
    memcpy(f, topic, strlen(topic));
    struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
    int64_t ns = ts.tv_sec*1000000000LL+ts.tv_nsec;
    memcpy(f+112, &ns, 8);
    uint32_t pl = plen; memcpy(f+138, &pl, 4);
    memcpy(f+256, payload, plen);
    nng_send(s, f, 256+plen, 0); nng_close(s);
}

static ssize_t recv_msg(const char *sub_topic, uint8_t *buf, size_t cap, int to_ms) {
    nng_socket s; nng_sub0_open(&s);
    nng_dial(s, sub_port, NULL, 0);
    nng_socket_set(s, NNG_OPT_SUB_SUBSCRIBE, sub_topic, strlen(sub_topic)+1);
    nng_socket_set_ms(s, "recv-timeout", to_ms);
    nng_msg *m = NULL;
    if (nng_recvmsg(s, &m, 0)) { nng_close(s); return -1; }
    size_t len = nng_msg_len(m);
    if (len > 256 && buf && len-256 <= cap)
        memcpy(buf, (uint8_t*)nng_msg_body(m)+256, len-256);
    ssize_t ret = len > 256 ? (ssize_t)(len-256) : 0;
    nng_msg_free(m); nng_close(s);
    return ret;
}

static void free_ports() { system("fuser -k 55656/tcp 55657/tcp 2>/dev/null"); ms(300); }
static void cleanup(void) {
    system("pkill -9 mesh-proxy actor 2>/dev/null"); ms(300);
}

/* ═══ TESTS ═══ */

static void t_proxy_forward(void) {
    TEST("proxy: forwards message");
    free_ports(); cleanup();
    pid_t pp = spawn_proxy(); ms(500);
    uint8_t p[] = "hello";
    send_msg("test", p, 5);
    ssize_t n = recv_msg("test", NULL, 0, 2000);
    CHECK(n == 5, "not forwarded");
    kill(pp, SIGKILL); waitpid(pp, NULL, 0);
}

static void t_proxy_filter(void) {
    TEST("proxy: topic filtering");
    free_ports(); cleanup();
    pid_t pp = spawn_proxy(); ms(500);
    send_msg("topic_A", (uint8_t*)"x", 1);
    ssize_t n = recv_msg("topic_B", NULL, 0, 1000);
    CHECK(n < 0, "received wrong topic");
    kill(pp, SIGKILL); waitpid(pp, NULL, 0);
}

static void t_actor_responds(void) {
    TEST("actor: receives and responds");
    free_ports(); cleanup();
    pid_t pp = spawn_proxy(); ms(500);
    system("mkdir -p /tmp/tm-a1");
    pid_t ap = spawn_actor("a1","ping","pong",
        "sh -c 'echo pong; echo ok'","/tmp/tm-a1");
    ms(500);
    send_msg("ping", (uint8_t*)"data", 4);
    ssize_t n = recv_msg("pong", NULL, 0, 3000);
    CHECK(n > 0, "no response");
    kill(pp,SIGKILL); kill(ap,SIGKILL);
    waitpid(pp,NULL,0); waitpid(ap,NULL,0);
}

static void t_actor_retry(void) {
    TEST("actor: retries on failure");
    free_ports(); cleanup();
    pid_t pp = spawn_proxy(); ms(500);
    system("rm -rf /tmp/tm-a2; mkdir -p /tmp/tm-a2");
    pid_t ap = spawn_actor("a2","rin","rout",
        "sh -c 'A=/tmp/tm-a2/c; C=$(cat $A 2>/dev/null||echo 0); C=$((C+1)); echo $C > $A; [ $C -ge 2 ] && echo rout && echo ok || exit 1'",
        "/tmp/tm-a2");
    ms(500);
    send_msg("rin", (uint8_t*)"x", 1);
    ssize_t n = recv_msg("rout", NULL, 0, 5000);
    CHECK(n > 0, "retry failed");
    kill(pp,SIGKILL); kill(ap,SIGKILL);
    waitpid(pp,NULL,0); waitpid(ap,NULL,0);
}

static void t_actor_multitopic(void) {
    TEST("actor: multi-topic subscription");
    free_ports(); cleanup();
    pid_t pp = spawn_proxy(); ms(500);
    system("mkdir -p /tmp/tm-a3");
    pid_t ap = spawn_actor("a3","topic_a,topic_b","out",
        "sh -c 'echo out; echo got'","/tmp/tm-a3");
    ms(500);
    send_msg("topic_b", (uint8_t*)"x", 1);
    ssize_t n = recv_msg("out", NULL, 0, 3000);
    CHECK(n > 0, "did not receive on 2nd topic");
    kill(pp,SIGKILL); kill(ap,SIGKILL);
    waitpid(pp,NULL,0); waitpid(ap,NULL,0);
}

static void t_handler_env(void) {
    TEST("handler: receives env vars");
    free_ports(); cleanup();
    pid_t pp = spawn_proxy(); ms(500);
    system("mkdir -p /tmp/tm-a4");
    pid_t ap = spawn_actor("a4","env_in","env_out",
        "sh -c 'echo env_out; echo T=\\$ACTOR_TUPLE_ID C=\\$ACTOR_CORRELATION_ID O=\\$ACTOR_TUPLE_ORIGIN A=\\$ACTOR_ATTEMPT'",
        "/tmp/tm-a4");
    ms(500);
    send_msg("env_in", (uint8_t*)"x", 1);
    uint8_t buf[512];
    ssize_t n = recv_msg("env_out", buf, sizeof(buf)-1, 3000);
    CHECK(n > 0, "no response from handler");
    if (n > 0) {
        buf[n] = 0;
        CHECK(strstr((char*)buf, "T=") != NULL, "TUPLE_ID missing");
        CHECK(strstr((char*)buf, "C=") != NULL, "CORRELATION_ID missing");
        CHECK(strstr((char*)buf, "O=") != NULL, "ORIGIN missing");
        CHECK(strstr((char*)buf, "A=") != NULL, "ATTEMPT missing");
    }
    kill(pp,SIGKILL); kill(ap,SIGKILL);
    waitpid(pp,NULL,0); waitpid(ap,NULL,0);
}

static void t_handler_route(void) {
    TEST("handler: dynamic topic routing");
    free_ports(); cleanup();
    pid_t pp = spawn_proxy(); ms(500);
    system("mkdir -p /tmp/tm-a5");
    pid_t ap = spawn_actor("a5","din","default",
        "sh -c 'echo custom_topic; echo result'","/tmp/tm-a5");
    ms(500);
    send_msg("din", (uint8_t*)"x", 1);
    ssize_t n = recv_msg("custom_topic", NULL, 0, 2000);
    CHECK(n > 0, "dynamic routing failed");
    kill(pp,SIGKILL); kill(ap,SIGKILL);
    waitpid(pp,NULL,0); waitpid(ap,NULL,0);
}

/* ═══ MAIN ═══ */

int main(void) {
    printf("Actor Mesh Test Suite\n\n");
    t_proxy_forward();
    t_proxy_filter();
    t_actor_responds();
    t_actor_retry();
    t_actor_multitopic();
    t_handler_env();
    t_handler_route();
    printf("\n%s (%d failures)\n", failures ? "FAIL" : "ALL PASSED", failures);
    free_ports(); cleanup();
    return failures;
}
