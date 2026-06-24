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

// buildFrame: topic (string ptr), topicLen, origin (string ptr), originLen,
//             id (16 raw bytes, caller-generated via crypto.getRandomValues),
//             payload (string ptr), payloadLen, output buffer (256+payloadLen bytes)
// Returns: total frame length
int buildFrame(char *topic, int topicLen, char *origin, int originLen,
               uint8_t *id, char *payload, int payloadLen, uint8_t *output) {
    actor_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));

    memcpy(hdr.topic, topic, topicLen < 32 ? topicLen : 31);
    memcpy(hdr.id, id, 16);
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

// parseId: read this frame's own 16-byte id (output buffer >= 16 bytes)
void parseId(uint8_t *frame, int frameLen, uint8_t *idOut) {
    if (frameLen < 256) { memset(idOut, 0, 16); return; }
    memcpy(idOut, frame + 32, 16); /* offsetof(actor_header_t, id) */
}

// parseCausationId: read the id of the request this frame is a reply to
// (output buffer >= 16 bytes) -- used to correlate a worker's result frame
// back to the request frame that caused it.
void parseCausationId(uint8_t *frame, int frameLen, uint8_t *idOut) {
    if (frameLen < 256) { memset(idOut, 0, 16); return; }
    memcpy(idOut, frame + 64, 16); /* offsetof(actor_header_t, causation_id) */
}
