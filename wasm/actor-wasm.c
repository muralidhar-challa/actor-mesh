// actor-wasm.c — WASM browser actor
//
// Same protocol as C actor runtime, compiled to WASM.
// Uses Emscripten WebSocket API instead of NNG.
// No fork/exec — handlers are function pointers (for now, just relay).
//
// Exports to JS:
//   actor_init(url)      — connect to proxy ws:// endpoint
//   actor_request(topic, payload_json, callback_id)
//   actor_poll()         — process incoming messages, call JS callbacks
//
// The React UI just calls actor_request("task.crud", "{...}") and
// gets the result via a JS callback. Zero JS logic.

#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <emscripten/websocket.h>
#include <emscripten/emscripten.h>

// ── actor_tuple_t (same as actor_tuple.h) ─────────────────────────

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

_Static_assert(sizeof(actor_header_t) == 256, "header must be 256 bytes");

// ── WebSocket state ───────────────────────────────────────────────

static EMSCRIPTEN_WEBSOCKET_T g_ws = 0;
static uint8_t g_recv_buf[65536];
static size_t  g_recv_len = 0;

// Pending request callbacks
#define MAX_PENDING 8
static int g_pending_ids[MAX_PENDING];
static int g_pending_count = 0;

// ── Helpers ───────────────────────────────────────────────────────

static void uuid4(uint8_t *out) {
    EM_ASM({
        crypto.getRandomValues(new Uint8Array(Module.HEAPU8.buffer, $0, 16));
    }, out);
}

static int64_t now_ns(void) {
    return (int64_t)(emscripten_get_now() * 1e6);
}

// ── Frame builder ─────────────────────────────────────────────────

static void build_frame(actor_header_t *hdr, const char *topic, 
                        const char *origin, const char *payload) {
    memset(hdr, 0, sizeof(*hdr));
    strncpy(hdr->topic, topic, 31);
    uuid4(hdr->id);
    strncpy(hdr->origin, origin, 31);
    hdr->emitted_at = now_ns();
    hdr->payload_len = (uint32_t)strlen(payload);
}

// ── Send ──────────────────────────────────────────────────────────

static void send_frame(const char *topic, const char *payload) {
    actor_header_t hdr;
    build_frame(&hdr, topic, "browser", payload);
    
    uint8_t frame[sizeof(hdr) + 65536];
    memcpy(frame, &hdr, sizeof(hdr));
    memcpy(frame + sizeof(hdr), payload, hdr.payload_len);
    
    emscripten_websocket_send_binary(g_ws, frame, sizeof(hdr) + hdr.payload_len);
}

// ── WebSocket callbacks ───────────────────────────────────────────

static EM_BOOL onopen(int eventType, const EmscriptenWebSocketOpenEvent *e, void *ud) {
    (void)eventType; (void)e; (void)ud;
    return EM_TRUE;
}

static EM_BOOL onmessage(int eventType, const EmscriptenWebSocketMessageEvent *e, void *ud) {
    (void)eventType; (void)ud;
    if (!e->isText && e->numBytes >= 256) {
        const uint8_t *data = e->data;
        actor_header_t *hdr = (actor_header_t*)data;
        uint32_t plen = hdr->payload_len;
        if (plen > e->numBytes - 256) plen = (uint32_t)(e->numBytes - 256);
        
        // Extract topic for callback routing
        char result_topic[64];
        strncpy(result_topic, hdr->topic, 32);
        result_topic[32] = 0;
        
        // Call JS callback with result
        EM_ASM({
            var topic = UTF8ToString($0);
            var payload = UTF8ToString($1, $2);
            if (window.__actorCallbacks && window.__actorCallbacks[topic]) {
                window.__actorCallbacks[topic](JSON.parse(payload));
            }
        }, result_topic, data + 256, plen);
    }
    return EM_TRUE;
}

static EM_BOOL onerror(int eventType, const EmscriptenWebSocketErrorEvent *e, void *ud) {
    (void)eventType; (void)e; (void)ud;
    return EM_TRUE;
}

// ── Public API (exported to JS) ───────────────────────────────────

EMSCRIPTEN_KEEPALIVE
int actor_init(const char *url) {
    EmscriptenWebSocketCreateAttributes attr;
    emscripten_websocket_init_create_attributes(&attr);
    attr.url = url;
    
    g_ws = emscripten_websocket_new(&attr);
    emscripten_websocket_set_onopen_callback(g_ws, NULL, onopen);
    emscripten_websocket_set_onmessage_callback(g_ws, NULL, onmessage);
    emscripten_websocket_set_onerror_callback(g_ws, NULL, onerror);
    return g_ws > 0 ? 0 : -1;
}

EMSCRIPTEN_KEEPALIVE
void actor_request(const char *topic, const char *payload) {
    send_frame(topic, payload);
}

EMSCRIPTEN_KEEPALIVE
void actor_poll(void) {
    // No-op — WebSocket events are async
}

int main(void) { return 0; }
