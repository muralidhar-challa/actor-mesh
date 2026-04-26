#define _POSIX_C_SOURCE 200809L
/* actor_uuid.h
 *
 * Minimal UUIDv7 generation.
 * No dependency. Single header.
 *
 * UUIDv7 layout:
 *   [ 48-bit unix ms ][ 4-bit version=7 ][ 12-bit rand ]
 *   [ 2-bit variant  ][ 62-bit rand ]
 */

#ifndef ACTOR_UUID_H
#define ACTOR_UUID_H

#include <stdint.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>

#ifdef __linux__
#  include <sys/random.h>
#endif

/* actor_uuid_gen fills dst with a 16-byte UUIDv7. */
static inline void actor_uuid_gen(uint8_t dst[16]) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    uint64_t ms = (uint64_t)ts.tv_sec * 1000ULL
                + (uint64_t)ts.tv_nsec / 1000000ULL;

    uint8_t rnd[10];
#ifdef __linux__
    if (getrandom(rnd, sizeof(rnd), 0) != (ssize_t)sizeof(rnd)) {
        for (int i = 0; i < 10; i++) rnd[i] = (uint8_t)rand();
    }
#else
    /* fallback — not cryptographically strong but sufficient for mesh ids */
    for (int i = 0; i < 10; i++) rnd[i] = (uint8_t)rand();
#endif

    /* bytes 0-5: 48-bit ms timestamp big endian */
    dst[0] = (ms >> 40) & 0xFF;
    dst[1] = (ms >> 32) & 0xFF;
    dst[2] = (ms >> 24) & 0xFF;
    dst[3] = (ms >> 16) & 0xFF;
    dst[4] = (ms >>  8) & 0xFF;
    dst[5] = (ms      ) & 0xFF;

    /* byte 6: version 7 + 4 rand bits */
    dst[6] = 0x70 | (rnd[0] & 0x0F);

    /* byte 7: 8 rand bits */
    dst[7] = rnd[1];

    /* byte 8: variant 10xx + 6 rand bits */
    dst[8] = 0x80 | (rnd[2] & 0x3F);

    /* bytes 9-15: 56 rand bits */
    memcpy(dst + 9, rnd + 3, 7);
}

/* actor_uuid_hex formats a 16-byte uuid as a 32-char lowercase hex string (no dashes).
 * dst must be at least 33 bytes. */
static inline void actor_uuid_hex(const uint8_t src[16], char dst[33]) {
    static const char h[] = "0123456789abcdef";
    for (int i = 0; i < 16; i++) {
        dst[i*2]   = h[src[i] >> 4];
        dst[i*2+1] = h[src[i] & 0xf];
    }
    dst[32] = '\0';
}

/* actor_uuid_str formats a 16-byte uuid as a 36-char null-terminated string.
 * dst must be at least 37 bytes. */
static inline void actor_uuid_str(const uint8_t src[16], char dst[37]) {
    static const char hex[] = "0123456789abcdef";
    int i = 0, j = 0;
    for (; i < 4;  i++, j+=2) { dst[j]=hex[src[i]>>4]; dst[j+1]=hex[src[i]&0xF]; }
    dst[j++] = '-';
    for (; i < 6;  i++, j+=2) { dst[j]=hex[src[i]>>4]; dst[j+1]=hex[src[i]&0xF]; }
    dst[j++] = '-';
    for (; i < 8;  i++, j+=2) { dst[j]=hex[src[i]>>4]; dst[j+1]=hex[src[i]&0xF]; }
    dst[j++] = '-';
    for (; i < 10; i++, j+=2) { dst[j]=hex[src[i]>>4]; dst[j+1]=hex[src[i]&0xF]; }
    dst[j++] = '-';
    for (; i < 16; i++, j+=2) { dst[j]=hex[src[i]>>4]; dst[j+1]=hex[src[i]&0xF]; }
    dst[j] = '\0';
}

#endif /* ACTOR_UUID_H */
