# Actor Mesh Control Plane UI — Plan

## Purpose

A generic, topology-agnostic desktop control plane for observing, tuning, and
injecting into any Actor Mesh. Not a chat client. The employee demo is one
topology that happens to run on top.

The UI *is itself a subscriber* — it connects to the mesh bus, subscribes to
every topic, and displays the raw truth.

---

## File Structure

```
ui/
  main.c                  # SDL2 init, Clay init, main event/render loop
  state.h                 # shared types, limits, design tokens, helpers (SPACER, cs, rule)
  state.c                 # global state definitions + prepare_strings()
  net.h                   # net_init / net_poll / net_send_frame declarations
  net.c                   # NNG pub/sub implementation, tuple parsing, uuid7
  components/
    header.h / header.c   # top bar: logo, connection pill, tab switcher
    actors.h / actors.c   # tab 0: live actor registry table
    bus.h    / bus.c      # tab 1: live tuple stream, expandable rows
    publish.h / publish.c # tab 2: inject arbitrary tuples onto the mesh
```

**Compilation unit rule**: only `main.c` defines `CLAY_IMPLEMENTATION` and
includes `clay_renderer_SDL2.c`. All other files include `clay.h` without the
define.

---

## Shared State (`state.h` / `state.c`)

### Types

```c
// Actor registered from heartbeat tuples
typedef struct {
    char     id[32];
    int      inbox, outbox;
    bool     alive;
    uint32_t last_hb_ticks;   // SDL_GetTicks() at last heartbeat
} Actor;

// One entry in the bus ring buffer
typedef struct {
    char     topic[33];
    char     origin[33];      // header bytes 80-111
    char     id_hex[9];       // first 4 bytes of tuple id as 8 hex chars
    char     corr_hex[9];     // first 4 bytes of correlation_id
    char     caus_hex[9];     // first 4 bytes of causation_id
    char     preview[129];    // first 128 printable payload bytes
    char     full[4097];      // first 4096 printable payload bytes (expanded view)
    uint32_t received_ticks;
    int      payload_len;
} TupleEvent;
```

### Limits

```c
#define MAX_ACTORS    16
#define MAX_EVENTS   256    // ring buffer size
#define UI_MAX_INPUT 2048
```

### Global state (defined in state.c, extern in state.h)

```c
// Actors
Actor  g_actors[MAX_ACTORS];
int    g_nactors;

// Bus ring buffer  (g_nevents % MAX_EVENTS = next write slot)
TupleEvent g_events[MAX_EVENTS];
int        g_nevents;       // total ever received
int        g_expanded;      // bus display-index currently expanded (-1 = none)

// Tabs: 0=Actors 1=Bus 2=Publish
int  g_tab;
bool g_connected;

// Publish fields
char g_pub_topic[128];
int  g_pub_topic_len;
char g_pub_payload[UI_MAX_INPUT];
int  g_pub_payload_len;
int  g_pub_focus;           // 0=topic field, 1=payload field

// Frame counter (spinner, cursor blink)
int g_tick;
```

### Per-frame display string buffers (written by prepare_strings before BeginLayout)

These are plain `char[]` globals in state.c. Components read them as `const char*`.

```c
// Actors tab
char s_alive_count[8];
char s_event_count[8];
char s_actor_label[MAX_ACTORS][48];   // "● tokenise-1"
char s_actor_inbox[MAX_ACTORS][16];   // "4 B"
char s_actor_outbox[MAX_ACTORS][16];  // "12 B"
char s_actor_age[MAX_ACTORS][16];     // "0.4s" / "8s" / "—"

// Bus tab  (display order: 0=newest)
int  s_disp_count;                    // how many rows to show
char s_evt_topic[MAX_EVENTS][48];
char s_evt_origin[MAX_EVENTS][40];
char s_evt_age[MAX_EVENTS][16];
char s_evt_preview[MAX_EVENTS][80];
// Expanded detail (for g_expanded row)
char s_exp_topic[48];
char s_exp_origin[40];
char s_exp_id[20];
char s_exp_corr[20];
char s_exp_caus[20];
char s_exp_full[512];                 // truncated full payload for display

// Publish tab
char s_pub_topic_disp[132];           // topic + cursor
char s_pub_payload_disp[UI_MAX_INPUT+4];
```

`prepare_strings()` runs once per frame before `Clay_BeginLayout()`.
It fills every buffer above from live state. Clay stores pointers, not copies,
so these buffers must remain valid until after render.

---

## Networking (`net.h` / `net.c`)

### Tuple wire format (256-byte header)

| Offset | Field          | Size |
|--------|----------------|------|
| 0      | topic          | 32   |
| 32     | id (UUIDv7)    | 16   |
| 48     | correlation_id | 16   |
| 64     | causation_id   | 16   |
| 80     | origin         | 32   |
| 112    | emitted_at ns  | 8    |
| 120    | ttl ns         | 8    |
| 128    | attempt        | 4    |
| 132    | payload_len    | 4    |
| 136    | _reserved      | 120  |
| 256    | payload…       |      |

### net_init()

Opens NNG pub+sub sockets. Subscribes to `""` (all topics). Sets recv timeout 10ms.

### net_poll()

Called once per frame. Drains the NNG receive queue.

For every valid message (blen ≥ 256):
1. Parse heartbeat → update `g_actors[]`, set `alive=true`, store `last_hb_ticks`
2. All messages → append to `g_events[]` ring buffer, extract topic/origin/id/payload

Reset `alive=false` for all actors at start of poll so actors that miss a
heartbeat go offline naturally (TTL = 3× heartbeat interval ≈ 6s).

### net_send_frame(topic, payload, plen)

Builds a 256-byte header + payload, sends via NNG pub socket.

---

## Components

### `header.c` — `build_header(float sc)`

```
┌─────────────────────────────────────────────────────────────────┐
│  Actor Mesh        ● 3 actors · 47 events    Actors  Bus  Publish│
└─────────────────────────────────────────────────────────────────┘
```

- Logo text left
- Connection status pill center (green dot + count when connected)
- Tab buttons right: Actors | Bus | Publish
- Active tab: ACCENT background; inactive: ghost on hover
- Click tab → set `g_tab`

### `actors.c` — `build_actors(float sc)`

```
┌──────────────────────────────────────────────────────────────────┐
│  3  ACTIVE     47  EVENTS     ●  CONNECTED                       │
├──────────────────────────────────────────────────────────────────┤
│  Actor                Inbox      Outbox    Last Beat   Status    │
├──────────────────────────────────────────────────────────────────┤
│  ● tokenise-1         4 B        12 B      0.4s        alive     │
│  ● llm-agent-1        8 B         4 B      1.2s        alive     │
│  ○ detokenise-1       0 B         0 B      8.3s        offline   │
└──────────────────────────────────────────────────────────────────┘
```

- Three stat cards at top (GROW → fill width equally)
- Table: dark header row, alternating white/BG rows, hover highlight
- Status pill: green "alive" / gray "offline"
- "Waiting for actors..." empty state

### `bus.c` — `build_bus(float sc)`

```
┌──────────────────────────────────────────────────────────────────┐
│  256 events    [pause]  [clear]                                  │
├──────────────────────────────────────────────────────────────────┤
│  0ms   [heartbeat]   tokenise-1    {"id":"tokenise-1","inbox":0} │
│  0.3s  [user_msg]    desktop       {"type":"user_message",...}   │
│▶ 0.4s  [sql_query]   llm-agent-1   {"sql":"SELECT * FROM em...   │
│  ├─ id:    018f2a3b                                              │
│  ├─ corr:  018f1234                                              │
│  ├─ caus:  018f0001                                              │
│  └─ payload: {"sql":"SELECT * FROM employees WHERE dept=..."}    │
└──────────────────────────────────────────────────────────────────┘
```

- Scrollable ring buffer display, newest first
- Each row: age | topic pill (color-coded by hash) | origin | payload preview
- Click a row → expand inline to show id/corr/caus + full payload (wrapping)
- Click again → collapse
- Topic pill: colored border + text, semi-transparent background
- Empty state: "Bus is quiet — no tuples received yet"

**Topic color palette** (hash of topic string → index 0-7):
blue, emerald, amber, violet, red, cyan, orange, lime.
Special case: `heartbeat` → gray (muted, de-emphasized).

### `publish.c` — `build_publish(float sc)`

```
┌──────────────────────────────────────────────────────────────────┐
│  Inject a tuple onto the mesh                                    │
│                                                                  │
│  Topic                                                           │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │ user_message█                                             │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Payload                                                         │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │ {"type":"user_message","query":"how many employees?"}     │   │
│  │                                                           │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                     [  Send  ]   │
└──────────────────────────────────────────────────────────────────┘
```

- Two input fields: topic (single line) + payload (multi-line wrapping)
- Active field shown with ACCENT border + blinking cursor
- Click field → focus it; Tab key → cycle focus between fields
- Enter on either field → send (if both non-empty)
- Send button → same action
- On send: payload cleared, topic kept (send again to same topic is common)
- Sends raw payload bytes via `net_send_frame`

---

## Main Loop (`main.c`)

```
SDL event drain
  SDL_QUIT → exit
  SDL_TEXTINPUT → route to focused input (publish tab only)
  SDL_KEYDOWN
    Backspace → delete from focused input
    Enter     → send (publish tab) or nothing
    Tab       → cycle g_tab (0→1→2→0)
  SDL_MOUSEBUTTONDOWN → set mouse_clicked=true
  SDL_MOUSEWHEEL → accumulate wheel_y

Clay_SetPointerState(mouse_pos, false)    // hover detection
Clay_UpdateScrollContainers(false, wheel, dt)

net_poll()                                // update state from mesh

prepare_strings()                         // fill all s_* buffers

Clay_SetLayoutDimensions(win_size)
Clay_BeginLayout()
  build_header(sc)
  if tab==0: build_actors(sc)
  if tab==1: build_bus(sc)
  if tab==2: build_publish(sc)
cmds = Clay_EndLayout(0)

if mouse_clicked:
  check Clay_PointerOver(TabActors/TabBus/TabPublish) → set g_tab
  if tab==1: check Clay_PointerOver(CLAY_SIDI("BusRow", i)) → toggle g_expanded
  if tab==2: check clicks on PubTopicField, PubPayloadField, PubSendBtn

SDL_RenderClear → Clay_SDL2_Render → SDL_RenderPresent
SDL_Delay(16)
```

---

## Fonts

| ID | File             | Size | Usage                          |
|----|------------------|------|--------------------------------|
| 0  | IBMPlexSans.ttf  | 64pt | Stat card large numbers        |
| 1  | IBMPlexSans.ttf  | 32pt | Body text, labels, buttons     |
| 2  | IBMPlexMono.ttf  | 28pt | Actor IDs, topic names, UUIDs, payloads |

---

## Design Tokens

```
BG       #f9f9f8   page background
DARK     #0a0a0a   header, table header, dark elements
ACCENT   #c41e3a   active tab, send button, highlights
ACCENT_H #e02a46   accent hover
DIM      #2a2a2a   primary text
MUTED    #888888   secondary text, inactive states
WHITE    #ffffff   card backgrounds
GREEN    #22c55e   connection indicator, alive status
SURFACE  #f0f0ee   hover row background
BORDER   #e2e2e2   table borders, card borders
```

---

## Makefile Change

```makefile
UI_SRCS = ui/main.c ui/state.c ui/net.c \
          ui/components/header.c \
          ui/components/actors.c \
          ui/components/bus.c \
          ui/components/publish.c

ui/clay-ui: $(UI_SRCS)
    $(CC) -Wall -O2 -std=c11 \
        -Wno-unused-variable -Wno-format-truncation -Wno-parentheses \
        $(UI_SRCS) \
        $(shell sdl2-config --cflags --libs) -lSDL2_ttf -lnng \
        -o ui/clay-ui
```

---

## What Gets Removed

- All chat-specific state: `g_msgs`, `g_waiting`, `g_total`, `Message` struct
- `build_chat()`, `maybe_autoscroll()` (replaced by bus autoscroll)
- `net_send_query()` (replace with generic `net_send_frame`)
- "Ask about employees" placeholder text
- Employee-specific topic parsing (`agent_response` → chat message)
