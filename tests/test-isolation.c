#define _POSIX_C_SOURCE 200809L
/* test-isolation.c — ACTOR_* confinement, phase 1.
 *
 * The property under test is not "isolation works". It is:
 *
 *     an actor must never run less isolated than it believes it is.
 *
 * A phase that silently degrades is worse than one that does not exist: a
 * deployment configures it, sees no error, and assumes a confinement it does
 * not have. So the tests that carry the weight here are the negative ones --
 * a requested primitive that cannot be applied must stop the actor before it
 * ever subscribes, not log a warning and carry on.
 *
 * Three cases per variable, all required:
 *   1. applied      -- set, and the restriction is OBSERVABLE in a handler.
 *                      "startup logged success" is not evidence.
 *   2. fails closed -- set to something unsatisfiable, actor exits, never
 *                      processes a tuple.
 *   3. absent       -- unset, behaviour identical to before this existed.
 *                      This is what protects existing deployments and is the
 *                      easiest to forget.
 *
 * Build:  make test-isolation && ./bin/test-isolation
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <spawn.h>
#include <signal.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <time.h>
#include <nng/nng.h>
#include <nng/protocol/pubsub0/pub.h>
#include <nng/protocol/pubsub0/sub.h>

static int failures = 0;
extern char **environ;
static void ms(int m) { struct timespec t = { m / 1000, (m % 1000) * 1000000L }; nanosleep(&t, NULL); }

#define SP "tcp://127.0.0.1:55766"
#define PP "tcp://127.0.0.1:55767"
#define TEST(n) printf("=== %s ===\n", n)
#define PASS()  printf("  PASS\n")
#define FAIL(m) do { printf("  FAIL: %s\n", m); failures++; } while (0)
#define CHECK(c,m) do { if (c) PASS(); else FAIL(m); } while (0)

#define MAXPL 1024

static void cleanup(void) {
    system("pkill -9 -f 'bin/mesh-proxy' 2>/dev/null");
    system("pkill -9 -f 'bin/actor' 2>/dev/null");
    ms(300);
}

static pid_t sp_(char **a, char **e) {
    pid_t p;
    if (posix_spawn(&p, a[0], NULL, NULL, a, e)) return -1;
    return p;
}

static void sendm(const char *topic, const char *payload) {
    nng_socket s; nng_pub0_open(&s); nng_dial(s, PP, NULL, 0); ms(60);
    size_t pl = strlen(payload);
    if (pl > MAXPL) { FAIL("payload too large"); nng_close(s); return; }
    uint8_t f[256 + MAXPL]; memset(f, 0, sizeof(f));
    size_t tl = strlen(topic); if (tl > 31) tl = 31;
    memcpy(f, topic, tl);
    memcpy(f + 80, "test", 4);
    struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
    int64_t ns = ts.tv_sec * 1000000000LL + ts.tv_nsec;
    memcpy(f + 112, &ns, 8);
    uint32_t pl2 = (uint32_t)pl; memcpy(f + 138, &pl2, 4);
    memcpy(f + 256, payload, pl);
    nng_send(s, f, 256 + pl, 0);
    nng_close(s);
}

static pid_t start_proxy(void) {
    char *a[] = { "./bin/mesh-proxy", NULL };
    char *e[] = { "PROXY_SUB_BIND=" PP, "PROXY_PUB_BIND=" SP, NULL };
    pid_t p = sp_(a, e); ms(600); return p;
}

/* Start an actor with extra env entries appended to the base set. */
static pid_t start_actor(const char *handler, const char *lmdb, char **extra, int nextra) {
    static char *e[32];
    static char h[512], l[512];
    snprintf(h, sizeof(h), "ACTOR_HANDLER=%s", handler);
    snprintf(l, sizeof(l), "ACTOR_LMDB_PATH=%s", lmdb);
    int n = 0;
    e[n++] = (char *)"ACTOR_BUS_SUB=" SP;
    e[n++] = (char *)"ACTOR_BUS_PUB=" PP;
    e[n++] = (char *)"ACTOR_ID=iso";
    e[n++] = (char *)"ACTOR_TOPIC=work";
    e[n++] = (char *)"ACTOR_RESULT_TOPIC=done";
    e[n++] = (char *)"ACTOR_HEARTBEAT_MS=0";
    e[n++] = h;
    e[n++] = l;
    for (int i = 0; i < nextra && n < 30; i++) e[n++] = extra[i];
    e[n] = NULL;
    char *a[] = { "./bin/actor", NULL };
    return sp_(a, e);
}

/* Collect result payloads for up to budget ms. Returns count; copies the first
   into `first` when given. */
static int drain(nng_socket s, int budget, char *first, size_t cap) {
    int got = 0; int waited = 0;
    while (waited < budget) {
        nng_msg *m = NULL;
        if (nng_recvmsg(s, &m, 0) == 0) {
            if (m) {
                size_t len = nng_msg_len(m);
                if (len > 256 && got == 0 && first && cap) {
                    size_t pl = len - 256; if (pl > cap - 1) pl = cap - 1;
                    memcpy(first, (uint8_t *)nng_msg_body(m) + 256, pl);
                    first[pl] = '\0';
                }
                nng_msg_free(m);
            }
            got++;
        }
        waited += 120;
    }
    return got;
}

static nng_socket sub_done(void) {
    nng_socket s; nng_sub0_open(&s); nng_dial(s, SP, NULL, 0);
    nng_socket_set(s, NNG_OPT_SUB_SUBSCRIBE, "done", 5);
    nng_socket_set_ms(s, NNG_OPT_RECVTIMEO, 120);
    ms(150);
    return s;
}

static void stop(pid_t proxy, pid_t actor) {
    if (actor > 0) { kill(actor, SIGKILL); waitpid(actor, NULL, 0); }
    if (proxy > 0) { kill(proxy, SIGKILL); waitpid(proxy, NULL, 0); }
    ms(200);
}

/* Did the actor exit on its own (fail closed), or is it still running? */
static int exited(pid_t p) {
    for (int i = 0; i < 25; i++) {
        int st;
        pid_t r = waitpid(p, &st, WNOHANG);
        if (r == p) return 1;
        ms(120);
    }
    return 0;
}

/* ── 1. absent is unchanged ──────────────────────────────────────────────── */

/* The constraint the entire plan rests on: an actor that sets none of the new
   variables behaves exactly as it did before this code existed. */
static void t_absent(void) {
    TEST("no isolation vars: behaviour unchanged");
    cleanup();
    pid_t pp = start_proxy();
    system("rm -rf /tmp/iso1; mkdir -p /tmp/iso1");
    pid_t ap = start_actor("sh -c 'echo done; echo ok'", "/tmp/iso1", NULL, 0);
    ms(700);
    nng_socket s = sub_done();
    sendm("work", "hello");
    int got = drain(s, 1500, NULL, 0);
    nng_close(s);
    CHECK(got > 0, "actor did not process a tuple with no isolation configured");
    stop(pp, ap);
}

/* ── 2. rlimits are observable ───────────────────────────────────────────── */

/* NOFILE is the cleanest to observe: the handler asks the shell what its own
   descriptor limit is, and the answer must be the number we set -- proving the
   limit was inherited across fork/exec into the handler, not merely set on the
   actor. */
static void t_rlimit_nofile_applied(void) {
    TEST("ACTOR_RLIMIT_NOFILE is inherited by the handler");
    cleanup();
    pid_t pp = start_proxy();
    system("rm -rf /tmp/iso2; mkdir -p /tmp/iso2");
    char *extra[] = { (char *)"ACTOR_RLIMIT_NOFILE=64" };
    pid_t ap = start_actor("sh -c 'echo done; ulimit -n'", "/tmp/iso2", extra, 1);
    ms(700);
    nng_socket s = sub_done();
    sendm("work", "x");
    char buf[256] = {0};
    int got = drain(s, 1500, buf, sizeof(buf));
    nng_close(s);
    CHECK(got > 0, "no result");
    CHECK(atoi(buf) == 64, "handler did not see the configured NOFILE limit");
    if (got > 0) printf("  handler saw ulimit -n = %d\n", atoi(buf));
    stop(pp, ap);
}

/* A limit that cannot be honoured must stop the actor. RLIMIT_NOFILE above the
   hard limit is the reliable way to provoke that as an unprivileged user. */
static void t_rlimit_fails_closed(void) {
    TEST("an unsatisfiable rlimit fails closed");
    cleanup();
    pid_t pp = start_proxy();
    system("rm -rf /tmp/iso3; mkdir -p /tmp/iso3");
    /* Far above any hard limit an unprivileged process may raise to. */
    char *extra[] = { (char *)"ACTOR_RLIMIT_NOFILE=1000000000" };
    pid_t ap = start_actor("sh -c 'echo done; echo ok'", "/tmp/iso3", extra, 1);
    CHECK(exited(ap), "actor kept running despite an rlimit it could not apply");

    /* And it must not have processed anything on the way out. */
    nng_socket s = sub_done();
    sendm("work", "x");
    int got = drain(s, 900, NULL, 0);
    nng_close(s);
    CHECK(got == 0, "actor processed a tuple despite failing to isolate");
    stop(pp, -1);
}

/* A malformed value is a configuration error, not a 0. Silently reading
   ACTOR_UID=root as uid 0 is the worst possible reading of a typo. */
static void t_garbage_fails_closed(void) {
    TEST("a non-numeric isolation value fails closed");
    cleanup();
    pid_t pp = start_proxy();
    system("rm -rf /tmp/iso4; mkdir -p /tmp/iso4");
    char *extra[] = { (char *)"ACTOR_UID=root" };
    pid_t ap = start_actor("sh -c 'echo done; echo ok'", "/tmp/iso4", extra, 1);
    CHECK(exited(ap), "actor kept running with a non-numeric ACTOR_UID");
    stop(pp, -1);
}

/* ── 3. privilege drop ───────────────────────────────────────────────────── */

/* Dropping uid needs privilege we do not have in an ordinary test run. Rather
   than skip the case, assert the half that IS observable unprivileged: asking
   for a uid we cannot reach must fail closed rather than continue as the
   launching user. That is the same silent-degradation failure the module
   exists to prevent, and it is the more important direction. */
static void t_uid_unreachable_fails_closed(void) {
    TEST("an unreachable ACTOR_UID fails closed");
    cleanup();
    if (geteuid() == 0) {
        printf("  SKIP: running as root, this case needs an unprivileged run\n");
        return;
    }
    pid_t pp = start_proxy();
    system("rm -rf /tmp/iso5; mkdir -p /tmp/iso5");
    /* A uid this process certainly cannot become. */
    char *extra[] = { (char *)"ACTOR_UID=65123" };
    pid_t ap = start_actor("sh -c 'echo done; echo ok'", "/tmp/iso5", extra, 1);
    CHECK(exited(ap), "actor kept running after failing to drop uid");

    nng_socket s = sub_done();
    sendm("work", "x");
    int got = drain(s, 900, NULL, 0);
    nng_close(s);
    CHECK(got == 0, "actor processed a tuple after failing to drop privilege");
    stop(pp, -1);
}

/* ── 4. isolation does not break concurrency ─────────────────────────────── */

/* The properties test-concurrency.c defends -- each worker gets its own
   child's status, a failed handler is never reported as a success -- must
   still hold with isolation active. This is the regression the plan calls for:
   run the dangerous case again, with the new code switched on. */
static void t_failing_handler_still_fails(void) {
    TEST("a failing handler still fails with isolation active");
    cleanup();
    pid_t pp = start_proxy();
    system("rm -rf /tmp/iso6; mkdir -p /tmp/iso6");
    char *extra[] = {
        (char *)"ACTOR_RLIMIT_NOFILE=256",
        (char *)"ACTOR_CONCURRENCY=4",
        (char *)"ACTOR_RETRY_MAX=0",
    };
    pid_t ap = start_actor("sh -c 'exit 3'", "/tmp/iso6", extra, 3);
    ms(700);

    nng_socket r; nng_sub0_open(&r); nng_dial(r, SP, NULL, 0);
    nng_socket_set(r, NNG_OPT_SUB_SUBSCRIBE, "tuple_rejected", 15);
    nng_socket_set_ms(r, NNG_OPT_RECVTIMEO, 120);
    nng_socket d = sub_done();
    ms(150);

    for (int i = 0; i < 4; i++) sendm("work", "x");
    int rejected = drain(r, 2500, NULL, 0);
    int succeeded = drain(d, 300, NULL, 0);
    nng_close(r); nng_close(d);

    printf("  rejected=%d succeeded=%d (expect 4/0)\n", rejected, succeeded);
    CHECK(rejected == 4, "failing handlers were not all rejected");
    CHECK(succeeded == 0, "a failing handler was reported as a success");
    stop(pp, ap);
}

/* RLIMIT_NPROC is the one limit whose correct value depends on what else the
   REAL uid is already running: the kernel counts every process and thread for
   that uid system wide, not just this actor's. On a machine where the uid also
   owns a desktop session the current count can already exceed a limit that
   looks generous, and the actor then dies on its first pthread_create -- which
   NNG reports as "Out of memory", pointing nowhere near the cause.

   So the test asserts what is actually true rather than a comfortable number:
   a limit above the uid's current usage works, and the actor serves tuples. It
   reads the current count instead of hardcoding one, because the right value
   is a property of the machine, not of this test. */
static void t_nproc_ordering(void) {
    TEST("ACTOR_RLIMIT_NPROC above current usage: actor still serves tuples");
    cleanup();

    /* Current processes+threads for this uid, plus generous headroom. */
    int cur = 0;
    {
        char cmd[128];
        snprintf(cmd, sizeof(cmd), "ps -L -u %u --no-headers 2>/dev/null | wc -l", (unsigned)getuid());
        FILE *f = popen(cmd, "r");
        if (f) { if (fscanf(f, "%d", &cur) != 1) cur = 0; pclose(f); }
    }
    if (cur <= 0) { printf("  SKIP: could not read current process count\n"); return; }
    int limit = cur + 2048;
    printf("  uid has %d processes/threads; setting NPROC=%d\n", cur, limit);

    pid_t pp = start_proxy();
    system("rm -rf /tmp/iso7; mkdir -p /tmp/iso7");
    char nproc[64];
    snprintf(nproc, sizeof(nproc), "ACTOR_RLIMIT_NPROC=%d", limit);
    char *extra[] = { nproc };
    pid_t ap = start_actor("sh -c 'echo done; echo ok'", "/tmp/iso7", extra, 1);
    ms(700);
    nng_socket s = sub_done();
    sendm("work", "x");
    int got = drain(s, 1500, NULL, 0);
    nng_close(s);
    CHECK(got > 0, "actor with a headroom NPROC limit did not serve a tuple");
    stop(pp, ap);
}

/* The other direction, and the one that bites in practice: a limit BELOW what
   the uid is already using. The actor must not reach the poll loop pretending
   to be healthy -- it must die. This documents the hazard as an assertion so
   nobody has to rediscover it from an "Out of memory" from NNG. */
static void t_nproc_below_usage_fails(void) {
    TEST("ACTOR_RLIMIT_NPROC below current usage: actor does not serve tuples");
    cleanup();
    pid_t pp = start_proxy();
    system("rm -rf /tmp/iso8; mkdir -p /tmp/iso8");
    /* 1 is below any live uid's usage by construction. */
    char *extra[] = { (char *)"ACTOR_RLIMIT_NPROC=1" };
    pid_t ap = start_actor("sh -c 'echo done; echo ok'", "/tmp/iso8", extra, 1);
    ms(900);

    nng_socket s = sub_done();
    sendm("work", "x");
    int got = drain(s, 1200, NULL, 0);
    nng_close(s);
    CHECK(got == 0, "actor served a tuple despite an NPROC limit it cannot satisfy");
    stop(pp, ap);
}

int main(void) {
    printf("actor isolation tests (phase 1)\n\n");
    t_absent();
    t_rlimit_nofile_applied();
    t_rlimit_fails_closed();
    t_garbage_fails_closed();
    t_uid_unreachable_fails_closed();
    t_failing_handler_still_fails();
    t_nproc_ordering();
    t_nproc_below_usage_fails();
    printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED",
           failures, failures == 1 ? "" : "s");
    cleanup();
    return failures ? 1 : 0;
}
