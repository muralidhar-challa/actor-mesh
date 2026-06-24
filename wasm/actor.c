// actor.c — WASM browser actor (NNG SP over WebSocket)
// Speaks same protocol as backend C actor runtime.
// No NNG needed — raw SP framing over WebSocket.
#include <string.h>
#include <stdint.h>
#include <emscripten/websocket.h>
#include <emscripten/emscripten.h>

// ── actor_header_t (same as actor_tuple.h) ────────────────────────
typedef struct __attribute__((packed)) {
    char     topic[32];  uint8_t  id[16];  uint8_t  corr[16];
    uint8_t  caus[16];   char     origin[32]; int64_t  emitted_at;
    int64_t  ttl;        int32_t  attempt; uint32_t payload_len;
    uint8_t  _pad[120];
} actor_header_t;
_Static_assert(sizeof(actor_header_t) == 256, "bad size");

// ── SP framing ────────────────────────────────────────────────────
// NNG SP frame: [0x00 0x00][payload_len BE u32][payload bytes]
static void add_sp_framing(uint8_t *buf, size_t payload_len) {
    buf[0] = 0; buf[1] = 0;
    buf[2] = (payload_len >> 24) & 0xFF;
    buf[3] = (payload_len >> 16) & 0xFF;
    buf[4] = (payload_len >> 8) & 0xFF;
    buf[5] = payload_len & 0xFF;
}
static size_t strip_sp_framing(uint8_t *buf, size_t frame_len) {
    if (frame_len < 6) return 0;
    size_t plen = ((size_t)buf[2] << 24) | ((size_t)buf[3] << 16) | 
                  ((size_t)buf[4] << 8) | buf[5];
    if (plen > frame_len - 6) return 0;
    memmove(buf, buf + 6, plen);
    return plen;
}

// ── State ─────────────────────────────────────────────────────────
static EMSCRIPTEN_WEBSOCKET_T g_pub_ws = 0;
static EMSCRIPTEN_WEBSOCKET_T g_sub_ws = 0;

// ── Helpers ───────────────────────────────────────────────────────
static void uuid4(uint8_t *out) {
    EM_ASM({ crypto.getRandomValues(new Uint8Array(Module.HEAPU8.buffer, $0, 16)); }, out);
}
static int64_t now_ns(void) { return (int64_t)(emscripten_get_now() * 1e6); }

static void build_header(actor_header_t *h, const char *topic, const char *payload) {
    memset(h, 0, sizeof(*h));
    strncpy(h->topic, topic, 31);
    uuid4(h->id);
    strncpy(h->origin, "browser", 31);
    h->emitted_at = now_ns();
    h->payload_len = (uint32_t)strlen(payload);
}

// ── Send (pub ws://8081) ──────────────────────────────────────────
static void ws_send(EMSCRIPTEN_WEBSOCKET_T ws, const uint8_t *data, size_t len) {
    // NNG SP framing over WebSocket: the ws:// listener expects SP frames
    uint8_t framed[65536 + 6];
    add_sp_framing(framed, len);
    memcpy(framed + 6, data, len);
    emscripten_websocket_send_binary(ws, framed, len + 6);
}

// ── Receive callback (sub ws://8080) ──────────────────────────────
static EM_BOOL onmessage(int et, const EmscriptenWebSocketMessageEvent *e, void *ud) {
    (void)et; (void)ud;
    if (e->isText || e->numBytes < 6) return EM_TRUE;
    
    uint8_t buf[65536];
    memcpy(buf, e->data, e->numBytes);
    size_t len = strip_sp_framing(buf, e->numBytes);
    if (len < 256) return EM_TRUE;
    
    actor_header_t *h = (actor_header_t*)buf;
    uint32_t plen = h->payload_len;
    if (plen > len - 256) plen = (uint32_t)(len - 256);
    
    char topic[33] = {0};
    memcpy(topic, h->topic, 32);
    
    EM_ASM({
        var topic = UTF8ToString($0);
        var payload = UTF8ToString($1, $2);
        if (window.__actorCallbacks && window.__actorCallbacks[topic]) {
            try { window.__actorCallbacks[topic](JSON.parse(payload)); } catch(e) {}
        }
    }, topic, buf + 256, plen);
    return EM_TRUE;
}

// ── Public API ────────────────────────────────────────────────────
EMSCRIPTEN_KEEPALIVE int actor_init(const char *pub_url, const char *sub_url) {
    EmscriptenWebSocketCreateAttributes a;
    emscripten_websocket_init_create_attributes(&a);
    
    a.url = sub_url;
    g_sub_ws = emscripten_websocket_new(&a);
    emscripten_websocket_set_onmessage_callback(g_sub_ws, NULL, onmessage);
    
    a.url = pub_url;
    g_pub_ws = emscripten_websocket_new(&a);
    return (g_pub_ws > 0 && g_sub_ws > 0) ? 0 : -1;
}

EMSCRIPTEN_KEEPALIVE void actor_request(const char *topic, const char *payload) {
    actor_header_t hdr;
    build_header(&hdr, topic, payload);
    uint8_t frame[sizeof(hdr) + 65536];
    memcpy(frame, &hdr, sizeof(hdr));
    size_t plen = strlen(payload);
    memcpy(frame + sizeof(hdr), payload, plen);
    ws_send(g_pub_ws, frame, sizeof(hdr) + plen);
}

int main(void) { return 0; }
