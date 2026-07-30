#define _POSIX_C_SOURCE 200809L
/* actor.c — generic actor runtime
 *
 * Zero malloc. Fixed static buffers. Capped payload.
 * Loop: poll NNG → check TTL → write inbox LMDB → fork handler
 *       → write payload stdin → read result stdout → build result header
 *       → write outbox LMDB → publish → clear LMDB
 *
 * Handler sees: payload bytes on stdin
 * Handler emits: result bytes on stdout
 * Handler knows: nothing else
 */

#include "actor.h"
#include "actor_tuple.h"
#include "actor_uuid.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <errno.h>
#include <time.h>
#include <sys/wait.h>

#ifdef _WIN32
#  include <windows.h>
#endif

#include <nng/nng.h>
#include <nng/protocol/pubsub0/pub.h>
#include <nng/protocol/pubsub0/sub.h>

#include <lmdb.h>

/* ── Payload cap ─────────────────────────────────────────────────────────── */

/* ACTOR_MAX_PAYLOAD — hard ceiling on payload size.
 * Override at build time: -DACTOR_MAX_PAYLOAD=4194304 for 4MB.
 * Payloads exceeding this are dropped — never malloc to accommodate. */
#ifndef ACTOR_MAX_PAYLOAD
#  define ACTOR_MAX_PAYLOAD (1024 * 1024)   /* 1MB default */
#endif

#define ACTOR_MAX_FRAME (ACTOR_MAX_PAYLOAD + sizeof(actor_header_t))

/* ── Static buffers — zero heap ──────────────────────────────────────────── */

/* g_result_buf: handler stdout is read into here — fixed, never grows */
static uint8_t g_result_buf[ACTOR_MAX_PAYLOAD];

/* g_frame_buf: header + payload assembled here for LMDB writes and NNG sends */
static uint8_t g_frame_buf[ACTOR_MAX_FRAME];

/* ── Config ──────────────────────────────────────────────────────────────── */

typedef struct {
    const char* id;
    const char* topic;
    const char* result_topic;
    const char* bus_sub;
    const char* bus_pub;
    const char* handler;
    const char* lmdb_path;
    int64_t     ttl_ns;
    int         heartbeat_ms;
    int         retry_max;
} actor_cfg_t;

static actor_cfg_t cfg;

static int cfg_load(void) {
    cfg.id           = getenv("ACTOR_ID");
    cfg.topic        = getenv("ACTOR_TOPIC");
    cfg.result_topic = getenv("ACTOR_RESULT_TOPIC");
    cfg.bus_sub      = getenv("ACTOR_BUS_SUB");
    cfg.bus_pub      = getenv("ACTOR_BUS_PUB");
    cfg.handler      = getenv("ACTOR_HANDLER");
    cfg.lmdb_path    = getenv("ACTOR_LMDB_PATH");

    if (!cfg.id || !cfg.topic || !cfg.result_topic ||
        !cfg.bus_sub || !cfg.bus_pub ||
        !cfg.handler || !cfg.lmdb_path) {
        fprintf(stderr, "[actor] missing required env vars\n");
        fprintf(stderr, "  required: ACTOR_ID ACTOR_TOPIC ACTOR_RESULT_TOPIC"
                        " ACTOR_BUS_SUB ACTOR_BUS_PUB ACTOR_HANDLER"
                        " ACTOR_LMDB_PATH\n");
        return -1;
    }

    const char* ttl = getenv("ACTOR_TTL_NS");
    cfg.ttl_ns = ttl ? atoll(ttl) : 0;

    const char* hb = getenv("ACTOR_HEARTBEAT_MS");
    cfg.heartbeat_ms = hb ? atoi(hb) : 5000;

    const char* rm = getenv("ACTOR_RETRY_MAX");
    cfg.retry_max = rm ? atoi(rm) : 3;

    return 0;
}

/* ── State ───────────────────────────────────────────────────────────────── */

static nng_socket nng_pub;
static nng_socket nng_sub;
static MDB_env*   mdb_env;
static MDB_dbi    dbi_inbox;
static MDB_dbi    dbi_outbox;
static MDB_dbi    dbi_state;

static volatile sig_atomic_t g_stop = 0;

/* ── Signal ──────────────────────────────────────────────────────────────── */

static void on_signal(int s) { (void)s; g_stop = 1; }

/* ── NNG ─────────────────────────────────────────────────────────────────── */

static int nng_setup(void) {
    int rc;

    if ((rc = nng_pub0_open(&nng_pub)) != 0) {
        fprintf(stderr, "[actor] nng_pub0_open: %s\n", nng_strerror(rc));
        return -1;
    }
    /* retry dial — proxy may not be listening yet */
    for (int i = 0; i < 30; i++) {
        rc = nng_dial(nng_pub, cfg.bus_pub, NULL, 0);
        if (rc == 0) break;
        fprintf(stderr, "[actor] pub dial %s: %s (retry %d)\n",
                cfg.bus_pub, nng_strerror(rc), i);
        sleep(1);
    }
    if (rc != 0) return -1;

    if ((rc = nng_sub0_open(&nng_sub)) != 0) {
        fprintf(stderr, "[actor] nng_sub0_open: %s\n", nng_strerror(rc));
        return -1;
    }
    for (int i = 0; i < 30; i++) {
        rc = nng_dial(nng_sub, cfg.bus_sub, NULL, 0);
        if (rc == 0) break;
        fprintf(stderr, "[actor] sub dial %s: %s (retry %d)\n",
                cfg.bus_sub, nng_strerror(rc), i);
        sleep(1);
    }
    if (rc != 0) return -1;

    /* ACTOR_TOPIC supports comma-separated list: "user_message,sql_result"
     * NNG sub0 does prefix matching on the message body. The topic field
     * is at offset 0 of actor_header_t, so subscribe with null-terminated
     * topic string for exact (non-prefix) match. */
    char topic_buf[256];
    strncpy(topic_buf, cfg.topic, sizeof(topic_buf) - 1);
    topic_buf[sizeof(topic_buf) - 1] = '\0';
    char *saveptr; char* tok = strtok_r(topic_buf, ",", &saveptr);
    while (tok) {
        while (*tok == ' ') tok++;
        size_t tlen = strlen(tok) + 1;  /* include null for exact match */
        if ((rc = nng_socket_set(nng_sub, NNG_OPT_SUB_SUBSCRIBE, tok, tlen)) != 0) {
            fprintf(stderr, "[actor] sub subscribe %s: %s\n", tok, nng_strerror(rc));
            return -1;
        }
        tok = strtok_r(NULL, ",", &saveptr);
    }

    /* set recv timeout for heartbeat check granularity (100ms) */
    nng_socket_set_ms(nng_sub, NNG_OPT_RECVTIMEO, 100);

    return 0;
}

/* ── LMDB ────────────────────────────────────────────────────────────────── */

static int lmdb_setup(void) {
    mdb_env_create(&mdb_env);
    mdb_env_set_mapsize(mdb_env, 64UL * 1024 * 1024);
    mdb_env_set_maxdbs(mdb_env, 3);

    int rc = mdb_env_open(mdb_env, cfg.lmdb_path, 0, 0664);
    if (rc) {
        fprintf(stderr, "[actor] lmdb open failed: %s\n", mdb_strerror(rc));
        return -1;
    }

    MDB_txn* txn;
    mdb_txn_begin(mdb_env, NULL, 0, &txn);
    mdb_dbi_open(txn, "inbox",  MDB_CREATE, &dbi_inbox);
    mdb_dbi_open(txn, "outbox", MDB_CREATE, &dbi_outbox);
    mdb_dbi_open(txn, "state",  MDB_CREATE, &dbi_state);
    mdb_txn_commit(txn);
    return 0;
}

static void lmdb_put(MDB_dbi dbi,
                     const void* key, size_t klen,
                     const void* val, size_t vlen) {
    MDB_txn* txn;
    if (mdb_txn_begin(mdb_env, NULL, 0, &txn)) return;
    MDB_val k = { klen, (void*)key };
    MDB_val v = { vlen, (void*)val };
    mdb_put(txn, dbi, &k, &v, 0);
    mdb_txn_commit(txn);
}

static void lmdb_del(MDB_dbi dbi, const void* key, size_t klen) {
    MDB_txn* txn;
    if (mdb_txn_begin(mdb_env, NULL, 0, &txn)) return;
    MDB_val k = { klen, (void*)key };
    mdb_del(txn, dbi, &k, NULL);
    mdb_txn_commit(txn);
}

static size_t lmdb_count(MDB_dbi dbi) {
    MDB_txn*  txn;
    MDB_stat  stat;
    if (mdb_txn_begin(mdb_env, NULL, MDB_RDONLY, &txn)) return 0;
    mdb_stat(txn, dbi, &stat);
    mdb_txn_abort(txn);
    return stat.ms_entries;
}

/* ── Handler spawn — cross-platform ──────────────────────────────────────── */

/* platform_spawn launches the handler, pipes payload to its stdin,
 * reads result from its stdout, returns the length or error.
 * Unix: fork/exec   Windows: CreateProcess */
#ifdef _WIN32

static ssize_t platform_spawn(const uint8_t* in,  size_t in_len,
                              uint8_t*       out, size_t out_cap) {
    HANDLE stdin_rd  = NULL, stdin_wr  = NULL;
    HANDLE stdout_rd = NULL, stdout_wr = NULL;
    SECURITY_ATTRIBUTES sa = { sizeof(sa), NULL, TRUE };
    ssize_t result = -1;

    if (!CreatePipe(&stdin_rd,  &stdin_wr,  &sa, 0)) return -1;
    if (!CreatePipe(&stdout_rd, &stdout_wr, &sa, 0)) goto fail_stdin;
    SetHandleInformation(stdin_wr,  HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(stdout_rd, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOA si = { sizeof(si) };
    si.hStdInput  = stdin_rd;
    si.hStdOutput = stdout_wr;
    si.hStdError  = GetStdHandle(STD_ERROR_HANDLE);
    si.dwFlags    = STARTF_USESTDHANDLES;

    char cmdline[1024];
    snprintf(cmdline, sizeof(cmdline), "cmd.exe /c %s", cfg.handler);

    PROCESS_INFORMATION pi = {0};
    if (!CreateProcessA(NULL, cmdline, NULL, NULL, TRUE,
                        0, NULL, NULL, &si, &pi))
        goto fail_stdout;

    CloseHandle(stdin_rd);   stdin_rd  = NULL;
    CloseHandle(stdout_wr);  stdout_wr = NULL;

    DWORD written = 0;
    if (!WriteFile(stdin_wr, in, (DWORD)in_len, &written, NULL) || written != in_len)
        goto fail_proc;
    CloseHandle(stdin_wr); stdin_wr = NULL;

    size_t len = 0;
    for (;;) {
        DWORD n = 0;
        if (!ReadFile(stdout_rd, out + len, (DWORD)(out_cap - len), &n, NULL) || n == 0)
            break;
        len += n;
        if (len == out_cap) {
            uint8_t drain[256]; DWORD d;
            while (ReadFile(stdout_rd, drain, sizeof(drain), &d, NULL) && d > 0) {}
            result = -2;
            goto fail_proc;
        }
    }
    result = (ssize_t)len;

    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD ec;
    if (GetExitCodeProcess(pi.hProcess, &ec) && ec != 0) result = -1;

fail_proc:
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
fail_stdout:
    if (stdout_rd) CloseHandle(stdout_rd);
    if (stdout_wr) CloseHandle(stdout_wr);
fail_stdin:
    if (stdin_rd)  CloseHandle(stdin_rd);
    if (stdin_wr)  CloseHandle(stdin_wr);
    return result;
}

#else /* Unix — fork/exec */

static ssize_t platform_spawn(const uint8_t* in,  size_t in_len,
                              uint8_t*       out, size_t out_cap) {
    int to_child[2], from_child[2];

    if (pipe(to_child)   < 0) return -1;
    if (pipe(from_child) < 0) { close(to_child[0]); close(to_child[1]); return -1; }

    pid_t pid = fork();
    if (pid < 0) {
        close(to_child[0]); close(to_child[1]);
        close(from_child[0]); close(from_child[1]);
        return -1;
    }

    if (pid == 0) {
        dup2(to_child[0],   STDIN_FILENO);
        dup2(from_child[1], STDOUT_FILENO);
        close(to_child[0]); close(to_child[1]);
        close(from_child[0]); close(from_child[1]);
        execl("/bin/sh", "sh", "-c", cfg.handler, NULL);
        _exit(1);
    }

    close(to_child[0]);
    close(from_child[1]);

    if (write(to_child[1], in, in_len) < 0) {
        close(to_child[1]);
        close(from_child[0]);
        waitpid(pid, NULL, 0);
        return -1;
    }
    close(to_child[1]);

    size_t  len = 0;
    ssize_t n;
    while ((n = read(from_child[0], out + len, out_cap - len)) > 0) {
        len += (size_t)n;
        if (len == out_cap) {
            uint8_t drain[256];
            while (read(from_child[0], drain, sizeof(drain)) > 0) {}
            close(from_child[0]);
            waitpid(pid, NULL, 0);
            fprintf(stderr, "[actor] result exceeded ACTOR_MAX_PAYLOAD=%d, dropping\n",
                    ACTOR_MAX_PAYLOAD);
            return -2;
        }
    }
    close(from_child[0]);

    int status;
    waitpid(pid, &status, 0);
    if (WIFEXITED(status) && WEXITSTATUS(status) != 0) return -1;

    return (ssize_t)len;
}

#endif /* _WIN32 */

/* invoke_handler: set env vars, spawn handler, return result length */
static ssize_t invoke_handler(const actor_header_t* hdr,
                              const uint8_t*        payload,
                              size_t                payload_len) {
    /* Set env vars before spawn — child inherits parent's environment.
     * On Unix (fork): child gets a copy. On Windows: child gets current env. */
    char id_hex[33], corr_hex[33], caus_hex[33], attempt_str[12];
    actor_uuid_hex(hdr->id,             id_hex);
    actor_uuid_hex(hdr->correlation_id, corr_hex);
    actor_uuid_hex(hdr->causation_id,   caus_hex);
    snprintf(attempt_str, sizeof(attempt_str), "%d", hdr->attempt);

#ifdef _WIN32
    SetEnvironmentVariableA("ACTOR_TUPLE_ID",       id_hex);
    SetEnvironmentVariableA("ACTOR_CORRELATION_ID", corr_hex);
    SetEnvironmentVariableA("ACTOR_CAUSATION_ID",   caus_hex);
    SetEnvironmentVariableA("ACTOR_TUPLE_ORIGIN",   hdr->origin);
    SetEnvironmentVariableA("ACTOR_ATTEMPT",        attempt_str);
#else
    setenv("ACTOR_TUPLE_ID",       id_hex,      1);
    setenv("ACTOR_CORRELATION_ID", corr_hex,    1);
    setenv("ACTOR_CAUSATION_ID",   caus_hex,    1);
    setenv("ACTOR_TUPLE_ORIGIN",   hdr->origin, 1);
    setenv("ACTOR_ATTEMPT",        attempt_str, 1);
#endif

    return platform_spawn(payload, payload_len, g_result_buf, ACTOR_MAX_PAYLOAD);
}

/* ── Publish ─────────────────────────────────────────────────────────────── */

/* If handler output begins with a bare topic name on its own line
 * (e.g. "sql_query\n{...}"), use that topic and skip the prefix line.
 * Otherwise fall back to cfg.result_topic and use the full buffer.
 * A valid topic prefix: only [a-zA-Z0-9_] chars followed immediately by '\n'. */
static const char* result_topic(size_t result_len, size_t* payload_off) {
    *payload_off = 0;
    const char* p = (const char*)g_result_buf;
    size_t i = 0;
    while (i < result_len && i < 31) {
        char c = p[i];
        if (c == '\n') {
            if (i == 0) break;           /* empty first line — no override   */
            *payload_off = i + 1;
            return p;                    /* caller uses p[0..i-1] as topic   */
        }
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') ||  c == '_')) {
            break;                       /* non-identifier char — no override */
        }
        i++;
    }
    return cfg.result_topic;            /* default */
}

static void publish_result(const actor_header_t* in_hdr, size_t result_len) {
    size_t        payload_off  = 0;
    const char*   raw_topic    = result_topic(result_len, &payload_off);

    /* if the handler wrote a topic prefix, null-terminate it in the buffer */
    char topic_buf[32] = {0};
    if (payload_off > 0) {
        size_t tlen = payload_off - 1;   /* exclude the '\n' */
        if (tlen >= sizeof(topic_buf)) tlen = sizeof(topic_buf) - 1;
        memcpy(topic_buf, raw_topic, tlen);
    } else {
        strncpy(topic_buf, raw_topic, sizeof(topic_buf) - 1);
    }

    const uint8_t* payload     = g_result_buf + payload_off;
    size_t         payload_len = result_len   - payload_off;

    actor_header_t out_hdr;
    actor_tuple_init(&out_hdr,
                     topic_buf,
                     cfg.id,
                     in_hdr->correlation_id,
                     in_hdr->id,
                     (uint32_t)payload_len);
    actor_uuid_gen(out_hdr.id);
    out_hdr.ttl = cfg.ttl_ns;

    /* assemble header + payload into frame buffer, write to LMDB outbox */
    size_t frame_len = sizeof(actor_header_t) + payload_len;
    memcpy(g_frame_buf, &out_hdr, sizeof(actor_header_t));
    memcpy(g_frame_buf + sizeof(actor_header_t), payload, payload_len);
    lmdb_put(dbi_outbox, out_hdr.id, 16, g_frame_buf, frame_len);

    /* publish as single contiguous message — NNG sub0 prefix-matches on topic */
    nng_send(nng_pub, g_frame_buf, frame_len, 0);

    lmdb_del(dbi_outbox, out_hdr.id, 16);
}

/* ── Rejection ───────────────────────────────────────────────────────────── */

static void publish_rejection(const actor_header_t* in_hdr, const char* reason) {
    char   id_hex[33], corr_hex[33];
    actor_uuid_hex(in_hdr->id,             id_hex);
    actor_uuid_hex(in_hdr->correlation_id, corr_hex);

    char payload[512];
    size_t plen = (size_t)snprintf(payload, sizeof(payload),
        "{\"tuple_id\":\"%s\",\"correlation_id\":\"%s\","
        "\"origin\":\"%s\",\"topic\":\"%s\",\"reason\":\"%s\"}",
        id_hex, corr_hex,
        in_hdr->origin, in_hdr->topic, reason);

    actor_header_t hdr;
    actor_tuple_init(&hdr, "tuple_rejected", cfg.id,
                     in_hdr->correlation_id, in_hdr->id, (uint32_t)plen);
    actor_uuid_gen(hdr.id);

    size_t frame_len = sizeof(actor_header_t) + plen;
    memcpy(g_frame_buf, &hdr, sizeof(actor_header_t));
    memcpy(g_frame_buf + sizeof(actor_header_t), payload, plen);
    nng_send(nng_pub, g_frame_buf, frame_len, 0);

    fprintf(stderr, "[actor] rejected tuple %s reason=%s\n", id_hex, reason);
}

/* ── Heartbeat ───────────────────────────────────────────────────────────── */

static void emit_heartbeat(void) {
    char   payload[256];
    size_t plen = (size_t)snprintf(payload, sizeof(payload),
                                   "{\"id\":\"%s\",\"inbox\":%zu,\"outbox\":%zu}",
                                   cfg.id,
                                   lmdb_count(dbi_inbox),
                                   lmdb_count(dbi_outbox));

    actor_header_t hdr;
    actor_tuple_init(&hdr, "heartbeat", cfg.id, NULL, NULL, (uint32_t)plen);
    actor_uuid_gen(hdr.id);
    hdr.ttl = (int64_t)cfg.heartbeat_ms * 3 * 1000000LL;

    /* assemble header + payload into frame buffer and send */
    size_t frame_len = sizeof(actor_header_t) + plen;
    memcpy(g_frame_buf, &hdr, sizeof(actor_header_t));
    memcpy(g_frame_buf + sizeof(actor_header_t), payload, plen);
    nng_send(nng_pub, g_frame_buf, frame_len, 0);
}

/* ── Process one tuple ───────────────────────────────────────────────────── */

static void process_tuple(const actor_header_t* hdr,
                          const uint8_t*        payload,
                          size_t                payload_len) {
    /* hard cap on incoming payload */
    if (payload_len > ACTOR_MAX_PAYLOAD) {
        publish_rejection(hdr, "payload_cap_exceeded");
        return;
    }

    /* write inbox LMDB — assemble frame into static buffer */
    size_t frame_len = sizeof(actor_header_t) + payload_len;
    memcpy(g_frame_buf, hdr, sizeof(actor_header_t));
    memcpy(g_frame_buf + sizeof(actor_header_t), payload, payload_len);
    lmdb_put(dbi_inbox, hdr->id, 16, g_frame_buf, frame_len);

    /* exponential backoff retry loop */
    int attempt = 0;
    while (attempt <= cfg.retry_max) {
        ssize_t result_len = invoke_handler(hdr, payload, payload_len);

        if (result_len > 0) {
            publish_result(hdr, (size_t)result_len);
            break;
        }
        if (result_len == 0) {
            /* handler produced no output — nothing to publish, treat as done */
            break;
        }

        if (result_len == -2) {
            /* result payload cap exceeded — no point retrying */
            publish_rejection(hdr, "result_cap_exceeded");
            break;
        }

        attempt++;
        if (attempt > cfg.retry_max) {
            publish_rejection(hdr, "max_retries_exceeded");
            break;
        }

        struct timespec backoff = {
            .tv_sec  = 0,
            .tv_nsec = (long)(100000000LL << (attempt - 1))
        };
        nanosleep(&backoff, NULL);
        fprintf(stderr, "[actor] retry %d/%d\n", attempt, cfg.retry_max);
    }

    lmdb_del(dbi_inbox, hdr->id, 16);
}

/* ── Main loop ───────────────────────────────────────────────────────────── */

/* Reap orphaned descendants.
 *
 * platform_spawn already waits for the shell it forks, so its own child never
 * lingers. The leak is one generation further down: the handler's shell
 * spawns its own children, and any that outlive it are reparented to this
 * process. In a container that is PID 1, whose kernel-assigned duty is to
 * reap them -- and nothing here did, so they accumulated as zombies for the
 * lifetime of the pod. Observed live: 21 of them (gpg, gpgconf, sh) aged up
 * to four hours, in a container whose only live process was this one.
 *
 * Zombies hold a PID table slot each. Left long enough the cgroup's pids.max
 * is reached, and the next fork in ANY process in that container fails with
 * EAGAIN -- which surfaces as an unrelated handler dying for no visible
 * reason, far from the cause.
 *
 * Deliberately a sweep here rather than a SIGCHLD handler or SIG_IGN. Both of
 * those reap indiscriminately and would race platform_spawn's blocking
 * waitpid for its own child: the handler's exit status is what decides
 * success or retry, and losing it to ECHILD would silently turn every failed
 * run into a successful one. This runs between messages, where no tracked
 * waitpid is outstanding, so it can only ever collect genuine orphans.
 *
 * WNOHANG keeps it non-blocking: with nothing to reap it is one syscall, and
 * the loop below already wakes ~10x/sec on the receive timeout.
 */
#ifndef _WIN32
static void reap_orphans(void) {
    while (waitpid(-1, NULL, WNOHANG) > 0) { }
}
#else
/* Windows has no fork/orphan model to clean up after; handler processes are
   waited for explicitly in the _WIN32 branch of platform_spawn. */
static void reap_orphans(void) { }
#endif

int actor_run(void) {
    if (cfg_load() < 0) return -1;

    signal(SIGTERM, on_signal);
    signal(SIGINT,  on_signal);

    if (nng_setup()  < 0) return -1;
    if (lmdb_setup() < 0) return -1;

    fprintf(stderr, "[actor] id=%s topic(s)=%s handler=%s max_payload=%d\n",
            cfg.id, cfg.topic, cfg.handler, ACTOR_MAX_PAYLOAD);

    int64_t last_hb = 0;

    while (!g_stop) {
        /* Collect any descendants orphaned by the previous handler run. Safe
           here and only here: platform_spawn's waitpid has already returned,
           so nothing this process still needs a status for is outstanding. */
        reap_orphans();

        /* heartbeat */
        if (cfg.heartbeat_ms > 0) {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            int64_t now_ms = ts.tv_sec * 1000LL + ts.tv_nsec / 1000000LL;
            if (now_ms - last_hb >= cfg.heartbeat_ms) {
                emit_heartbeat();
                last_hb = now_ms;
            }
        }

        /* receive with 100ms timeout — unblocks for heartbeat check */
        nng_msg* msg = NULL;
        int rc = nng_recvmsg(nng_sub, &msg, 0);
        if (rc == NNG_ETIMEDOUT) {
            continue;
        }
        if (rc != 0) {
            if (g_stop) break;
            fprintf(stderr, "[actor] recv error: %s\n", nng_strerror(rc));
            break;
        }

        void*  body      = nng_msg_body(msg);
        size_t body_len  = nng_msg_len(msg);

        if (body_len < sizeof(actor_header_t)) {
            fprintf(stderr, "[actor] short message %zu bytes, dropping\n", body_len);
            nng_msg_free(msg);
            continue;
        }

        const actor_header_t* hdr         = (const actor_header_t*)body;
        const uint8_t*        payload     = (const uint8_t*)body + sizeof(actor_header_t);
        size_t                payload_len = body_len - sizeof(actor_header_t);

        /* TTL check */
        if (actor_tuple_expired(hdr)) {
            publish_rejection(hdr, "ttl_expired");
            nng_msg_free(msg);
            continue;
        }

        process_tuple(hdr, payload, payload_len);
        nng_msg_free(msg);
    }

    fprintf(stderr, "[actor] shutting down\n");
    nng_close(nng_pub);
    nng_close(nng_sub);
    mdb_env_close(mdb_env);
    return 0;
}
