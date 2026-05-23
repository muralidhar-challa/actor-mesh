/* actor_tuple.h
 *
 * Fixed binary header for every tuple on the mesh.
 * No serialization library. Cast and read.
 *
 * Wire format (single contiguous buffer):
 *   [ actor_header_t 256 bytes ][ payload bytes ]
 *
 * NNG sub0 prefix-matches on topic[32] at offset 0.
 * Payload is opaque bytes. Runtime never inspects it.
 * Handler receives only payload. Handler emits only payload.
 * Runtime owns the header completely.
 */

#ifndef ACTOR_TUPLE_H
#define ACTOR_TUPLE_H

#define _POSIX_C_SOURCE 200809L

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

/* ── Header ──────────────────────────────────────────────────────────────── */

typedef struct __attribute__((packed)) {
    char     topic[32];           /* subscription prefix (at offset 0)   */
    uint8_t  id[16];              /* uuidv7 binary                       */
    uint8_t  correlation_id[16];  /* trace chain end to end              */
    uint8_t  causation_id[16];    /* direct parent tuple id              */
    char     origin[32];          /* actor id who emitted                */
    int64_t  emitted_at;          /* unix nanoseconds                    */
    int64_t  ttl;                 /* nanoseconds, 0 = no expiry          */
    int32_t  attempt;             /* retry count, 0 = first emission     */
    uint32_t payload_len;         /* bytes following this header         */
    uint8_t  _reserved[120];      /* pad to 256 bytes, zeroed            */
} actor_header_t;

_Static_assert(sizeof(actor_header_t) == 256,
               "actor_header_t must be exactly 256 bytes");

/* ── Helpers ─────────────────────────────────────────────────────────────── */

static inline int actor_tuple_expired(const actor_header_t* h) {
    if (h->ttl == 0) return 0;
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    int64_t now = (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
    return now > h->emitted_at + h->ttl;
}

static inline const uint8_t* actor_tuple_payload(const actor_header_t* h) {
    return (const uint8_t*)(h + 1);
}

static inline void actor_tuple_init(actor_header_t*  h,
                                    const char*      topic,
                                    const char*      origin,
                                    const uint8_t*   correlation_id,
                                    const uint8_t*   causation_id,
                                    uint32_t         payload_len) {
    memset(h, 0, sizeof(actor_header_t));
    snprintf(h->topic,  sizeof(h->topic),  "%s", topic);
    snprintf(h->origin, sizeof(h->origin), "%s", origin);
    if (correlation_id) memcpy(h->correlation_id, correlation_id, 16);
    if (causation_id)   memcpy(h->causation_id,   causation_id,   16);
    h->payload_len = payload_len;
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    h->emitted_at = (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

#endif /* ACTOR_TUPLE_H */
