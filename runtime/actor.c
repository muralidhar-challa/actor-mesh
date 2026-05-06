#define _POSIX_C_SOURCE 200809L
/* actor.c — generic actor runtime
 *
 * Zero malloc. Fixed static buffers. Capped payload.
 * Loop: poll ZMQ → check TTL → write inbox LMDB → fork handler
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

#include <zmq.h>
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

/* g_frame_buf: header + payload assembled here for LMDB writes */
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

static void*    zmq_ctx;
static void*    zmq_pub;
static void*    zmq_sub;
static MDB_env* mdb_env;
static MDB_dbi  dbi_inbox;
static MDB_dbi  dbi_outbox;
static MDB_dbi  dbi_state;

static volatile sig_atomic_t g_stop = 0;

/* ── Signal ──────────────────────────────────────────────────────────────── */

static void on_signal(int s) { (void)s; g_stop = 1; }

/* ── ZMQ ─────────────────────────────────────────────────────────────────── */

static int zmq_setup(void) {
    zmq_ctx = zmq_ctx_new();
    if (!zmq_ctx) return -1;

    zmq_pub = zmq_socket(zmq_ctx, ZMQ_PUB);
    if (zmq_connect(zmq_pub, cfg.bus_pub) < 0) {
        fprintf(stderr, "[actor] pub connect failed: %s\n", zmq_strerror(errno));
        return -1;
    }

    zmq_sub = zmq_socket(zmq_ctx, ZMQ_SUB);
    if (zmq_connect(zmq_sub, cfg.bus_sub) < 0) {
        fprintf(stderr, "[actor] sub connect failed: %s\n", zmq_strerror(errno));
        return -1;
    }

    /* ACTOR_TOPIC supports comma-separated list: "user_message,sql_result" */
    char topic_buf[256];
    strncpy(topic_buf, cfg.topic, sizeof(topic_buf) - 1);
    topic_buf[sizeof(topic_buf) - 1] = '\0';
    char *saveptr; char* tok = strtok_r(topic_buf, ",", &saveptr);
    while (tok) {
        while (*tok == ' ') tok++;
        zmq_setsockopt(zmq_sub, ZMQ_SUBSCRIBE, tok, strlen(tok) + 1);
        tok = strtok_r(NULL, ",", &saveptr);
    }
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

/* ── Handler ─────────────────────────────────────────────────────────────── */

/* invoke_handler forks the handler, writes payload to stdin,
 * reads result into g_result_buf (static, no malloc).
 * Returns result length, -1 on error, -2 if result exceeds cap. */
static ssize_t invoke_handler(const actor_header_t* hdr,
                              const uint8_t*        payload,
                              size_t                payload_len) {
    int to_handler[2];
    int from_handler[2];

    if (pipe(to_handler)   < 0) return -1;
    if (pipe(from_handler) < 0) { close(to_handler[0]); close(to_handler[1]); return -1; }

    pid_t pid = fork();
    if (pid < 0) return -1;

    if (pid == 0) {
        /* child — becomes the handler */
        dup2(to_handler[0],   STDIN_FILENO);
        dup2(from_handler[1], STDOUT_FILENO);
        close(to_handler[0]);  close(to_handler[1]);
        close(from_handler[0]); close(from_handler[1]);

        /* expose header fields as env vars — read-only visibility for handler */
        char id_hex[33], corr_hex[33], caus_hex[33], attempt_str[12];
        actor_uuid_hex(hdr->id,             id_hex);
        actor_uuid_hex(hdr->correlation_id, corr_hex);
        actor_uuid_hex(hdr->causation_id,   caus_hex);
        snprintf(attempt_str, sizeof(attempt_str), "%d", hdr->attempt);
        setenv("ACTOR_TUPLE_ID",       id_hex,      1);
        setenv("ACTOR_CORRELATION_ID", corr_hex,    1);
        setenv("ACTOR_CAUSATION_ID",   caus_hex,    1);
        setenv("ACTOR_TUPLE_ORIGIN",   hdr->origin, 1);
        setenv("ACTOR_ATTEMPT",        attempt_str, 1);

        execl("/bin/sh", "sh", "-c", cfg.handler, NULL);
        _exit(1);
    }

    /* parent */
    close(to_handler[0]);
    close(from_handler[1]);

    /* write payload → handler stdin, signal EOF */
    if (write(to_handler[1], payload, payload_len) < 0) {
        close(to_handler[1]);
        close(from_handler[0]);
        waitpid(pid, NULL, 0);
        return -1;
    }
    close(to_handler[1]);

    /* read result into fixed static buffer — hard cap enforced */
    size_t  len = 0;
    ssize_t n;
    while ((n = read(from_handler[0],
                     g_result_buf + len,
                     ACTOR_MAX_PAYLOAD - len)) > 0) {
        len += (size_t)n;
        if (len == ACTOR_MAX_PAYLOAD) {
            /* cap hit — drain and discard remainder, signal overflow */
            uint8_t drain[256];
            while (read(from_handler[0], drain, sizeof(drain)) > 0) {}
            close(from_handler[0]);
            waitpid(pid, NULL, 0);
            fprintf(stderr, "[actor] result exceeded ACTOR_MAX_PAYLOAD=%d, dropping\n",
                    ACTOR_MAX_PAYLOAD);
            return -2;
        }
    }
    close(from_handler[0]);

    int status;
    waitpid(pid, &status, 0);

    if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
        return -1;
    }

    return (ssize_t)len;
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

    size_t frame_len = sizeof(actor_header_t) + payload_len;
    memcpy(g_frame_buf, &out_hdr, sizeof(actor_header_t));
    memcpy(g_frame_buf + sizeof(actor_header_t), payload, payload_len);
    lmdb_put(dbi_outbox, out_hdr.id, 16, g_frame_buf, frame_len);

    zmq_send(zmq_pub, &out_hdr, sizeof(actor_header_t), ZMQ_SNDMORE);
    zmq_send(zmq_pub, payload,  payload_len,             0);

    lmdb_del(dbi_outbox, out_hdr.id, 16);
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

    zmq_send(zmq_pub, &hdr,    sizeof(actor_header_t), ZMQ_SNDMORE);
    zmq_send(zmq_pub, payload, plen,                   0);
}

/* ── Process one tuple ───────────────────────────────────────────────────── */

static void process_tuple(const actor_header_t* hdr,
                          const uint8_t*        payload,
                          size_t                payload_len) {
    /* hard cap on incoming payload — drop silently if exceeded */
    if (payload_len > ACTOR_MAX_PAYLOAD) {
        fprintf(stderr, "[actor] incoming payload %zu exceeds cap %d, dropping\n",
                payload_len, ACTOR_MAX_PAYLOAD);
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
            /* payload cap exceeded — no point retrying */
            break;
        }

        attempt++;
        if (attempt > cfg.retry_max) {
            fprintf(stderr, "[actor] handler failed after %d attempts, dropping\n",
                    cfg.retry_max);
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

int actor_run(void) {
    if (cfg_load() < 0) return -1;

    signal(SIGTERM, on_signal);
    signal(SIGINT,  on_signal);

    if (zmq_setup()  < 0) return -1;
    if (lmdb_setup() < 0) return -1;

    fprintf(stderr, "[actor] id=%s topic(s)=%s handler=%s max_payload=%d\n",
            cfg.id, cfg.topic, cfg.handler, ACTOR_MAX_PAYLOAD);

    zmq_pollitem_t items[1];
    items[0].socket = zmq_sub;
    items[0].events = ZMQ_POLLIN;

    int64_t last_hb = 0;

    while (!g_stop) {
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

        int rc = zmq_poll(items, 1, 100);
        if (rc < 0) break;
        if (rc == 0) continue;

        /* receive header frame */
        zmq_msg_t hdr_msg;
        zmq_msg_init(&hdr_msg);
        zmq_msg_recv(&hdr_msg, zmq_sub, 0);

        if (zmq_msg_size(&hdr_msg) < sizeof(actor_header_t)) {
            fprintf(stderr, "[actor] short frame, dropping\n");
            zmq_msg_close(&hdr_msg);
            continue;
        }

        const actor_header_t* hdr = zmq_msg_data(&hdr_msg);

        /* receive payload frame */
        zmq_msg_t pay_msg;
        zmq_msg_init(&pay_msg);
        if (zmq_msg_more(&hdr_msg)) {
            zmq_msg_recv(&pay_msg, zmq_sub, 0);
        }

        /* TTL check */
        if (actor_tuple_expired(hdr)) {
            fprintf(stderr, "[actor] tuple expired, dropping (topic=%.32s)\n", hdr->topic);
            zmq_msg_close(&hdr_msg);
            zmq_msg_close(&pay_msg);
            continue;
        }

        /* zero copy — payload pointer direct from ZMQ buffer */
        const uint8_t* payload     = zmq_msg_data(&pay_msg);
        size_t         payload_len = zmq_msg_size(&pay_msg);

        process_tuple(hdr, payload, payload_len);

        zmq_msg_close(&hdr_msg);
        zmq_msg_close(&pay_msg);
    }

    fprintf(stderr, "[actor] shutting down\n");
    zmq_close(zmq_pub);
    zmq_close(zmq_sub);
    zmq_ctx_destroy(zmq_ctx);
    mdb_env_close(mdb_env);
    return 0;
}
