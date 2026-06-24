// actor_wasm.c — WASM bindings for actor_tuple.h
// Exposes buildFrame() and parseTopic()/parsePayload() to JavaScript

#include <stdint.h>
#include <string.h>

// Import actor_tuple.h definitions manually (avoid NNG deps)
typedef struct __attribute__((packed)) {
    char     topic[32];
    uint8_t  id[16];
    uint8_t  correlation_id[16];
    uint8_t  causation_id[16];
    char     origin[32];
    int64_t  emitted_at;
    int64_t  ttl;
    int32_t  attempt;
    uint32_t payload_len;
    uint8_t  _reserved[120];
} actor_header_t;

// Emit a placeholder UUID (caller should overwrite with crypto.getRandomValues in JS)
static void uuid4(uint8_t *out) {
    for (int i = 0; i < 16; i++) out[i] = (uint8_t)(i * 7 + 13);
}

// buildFrame: topic (string ptr), topicLen, origin (string ptr), originLen,
//             payload (string ptr), payloadLen, output buffer (256+payloadLen bytes)
// Returns: total frame length
int buildFrame(char *topic, int topicLen, char *origin, int originLen,
               char *payload, int payloadLen, uint8_t *output) {
    actor_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));

    memcpy(hdr.topic, topic, topicLen < 32 ? topicLen : 31);
    uuid4(hdr.id);
    memcpy(hdr.origin, origin, originLen < 32 ? originLen : 31);
    hdr.payload_len = (uint32_t)payloadLen;

    memcpy(output, &hdr, 256);
    memcpy(output + 256, payload, payloadLen);
    return 256 + payloadLen;
}

// parseTopic: read topic from a received frame
// frame bytes, frameLen, output buffer (at least 33 bytes)
void parseTopic(uint8_t *frame, int frameLen, char *topicOut) {
    if (frameLen < 256) { topicOut[0] = 0; return; }
    memcpy(topicOut, frame, 32);
    topicOut[32] = 0;
}

// parsePayload: read payload length and copy payload from frame
// frame bytes, frameLen, output buffer
// Returns: payload length copied
int parsePayload(uint8_t *frame, int frameLen, uint8_t *payloadOut) {
    if (frameLen < 256) return 0;
    uint32_t plen = *(uint32_t*)(frame + 132); /* offsetof(actor_header_t, payload_len) */
    if (plen > (uint32_t)(frameLen - 256)) plen = (uint32_t)(frameLen - 256);
    memcpy(payloadOut, frame + 256, plen);
    return (int)plen;
}
