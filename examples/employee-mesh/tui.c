/* tui.c — ncurses chat interface for employee mesh
 *
 * Tuple usage:
 *   correlation_id  — one UUID per session, carried on every sent tuple
 *   causation_id    — set to the last agent_response tuple id on each send
 *   sql_query/sql_result subscriptions show per-turn thinking status
 *   agent_response matched by hdr->correlation_id, not payload session_id
 *
 * Layout:
 *   ┌────────────────────────────────────────────────┐
 *   │ conversation history (scrollable)              │
 *   ├────────────────────────────────────────────────┤
 *   │ mesh  ● llm-agent-1  ● sqlite-tool-1           │
 *   ├────────────────────────────────────────────────┤
 *   │ > input_                                       │
 *   └────────────────────────────────────────────────┘
 *
 * Keys: Enter=send  PgUp/PgDn or ↑↓=scroll  Ctrl-C=quit
 * Deps: libncursesw libnng libpthread mpack (vendored)
 */

#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "actor_tuple.h"
#include "actor_uuid.h"
#include "mpack.h"

#include <ncursesw/curses.h>
#include <locale.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <nng/nng.h>
#include <nng/protocol/pubsub0/pub.h>
#include <nng/protocol/pubsub0/sub.h>

/* ── Limits ──────────────────────────────────────────────────────────────── */

#define BUS_PUB       "tcp://localhost:5557"
#define BUS_SUB       "tcp://localhost:5556"

#define MAX_HISTORY   128
#define MAX_MSG       (32 * 1024)
#define MAX_DETAIL    64
#define TUI_MAX_INPUT 256
#define MAX_SEND      (512 * 1024)
#define MAX_RECV      (256 * 1024)
#define MAX_ACTORS    8
#define ACTOR_TTL_S   15
#define MAX_SLINES    4096
#define MAX_SLINE_W   512

/* ── Types ───────────────────────────────────────────────────────────────── */

typedef enum { ROLE_USER, ROLE_AGENT, ROLE_THINKING } role_t;

typedef struct {
    role_t role;
    char   text[MAX_MSG];
    char   detail[MAX_DETAIL]; /* ROLE_THINKING: current step label */
} message_t;

typedef struct {
    char   id[32];
    time_t last_seen;
} actor_t;

typedef struct {
    int  mi;
    int  attr;
    char text[MAX_SLINE_W];
} sline_t;

/* ── Global state ────────────────────────────────────────────────────────── */

static message_t g_msgs[MAX_HISTORY];
static int       g_nmsgs   = 0;
static int       g_scroll  = 0;

static actor_t   g_actors[MAX_ACTORS];
static int       g_nactors = 0;

/* one correlation_id for the whole session; causation chains each turn */
static uint8_t   g_corr_id[16];   /* set once at startup                  */
static uint8_t   g_last_id[16];   /* id of last agent_response received   */

static bool      g_thinking = false;
static int       g_tick     = 0;

static pthread_mutex_t g_lock  = PTHREAD_MUTEX_INITIALIZER;
static bool            g_dirty = false;
static volatile bool   g_run   = true;

static nng_socket g_pub;
static nng_socket g_sub;

static char g_input[TUI_MAX_INPUT + 1];
static int  g_ilen = 0;

static char g_send_buf[MAX_SEND];
static char g_recv_buf[MAX_RECV];

/* ── Windows ─────────────────────────────────────────────────────────────── */

static WINDOW *g_whist   = NULL;
static WINDOW *g_wstatus = NULL;
static WINDOW *g_winput  = NULL;

/* ── Color pair IDs ──────────────────────────────────────────────────────── */

#define CP_USER   1
#define CP_AGENT  2
#define CP_THINK  3
#define CP_STATUS 4
#define CP_INPUT  5

/* ── Spinner ─────────────────────────────────────────────────────────────── */

static const char *SPIN[] = { "⠋", "⠙", "⠸", "⠴", "⠦", "⠇" };
#define SPIN_N 6

/* ── Screen-line buffer ──────────────────────────────────────────────────── */

static sline_t g_slines[MAX_SLINES];
static int     g_nslines = 0;

/* ── Build wrapped screen lines from g_msgs (called under g_lock) ────────── */

static void build_slines(int width) {
    g_nslines = 0;
    int tw = width - 4;
    if (tw < 8) tw = 8;

    for (int i = 0; i < g_nmsgs && g_nslines < MAX_SLINES; i++) {
        message_t *m = &g_msgs[i];

        if (m->role == ROLE_THINKING) {
            sline_t *sl = &g_slines[g_nslines++];
            sl->mi   = i;
            sl->attr = COLOR_PAIR(CP_THINK) | A_DIM;
            const char *step = m->detail[0] ? m->detail : "thinking...";
            snprintf(sl->text, MAX_SLINE_W, "  %s %s", SPIN[g_tick % SPIN_N], step);
            goto gap;
        }

        /* label */
        {
            sline_t *sl = &g_slines[g_nslines++];
            sl->mi = i;
            if (m->role == ROLE_USER) {
                sl->attr = COLOR_PAIR(CP_USER) | A_BOLD;
                snprintf(sl->text, MAX_SLINE_W, "  you");
            } else {
                sl->attr = COLOR_PAIR(CP_AGENT) | A_BOLD;
                snprintf(sl->text, MAX_SLINE_W, "  agent");
            }
        }

        /* word-wrapped body */
        {
            int msg_attr = (m->role == ROLE_USER)
                           ? COLOR_PAIR(CP_USER)
                           : COLOR_PAIR(CP_AGENT);
            const char *p = m->text;
            while (*p && g_nslines < MAX_SLINES) {
                int n = 0, last_sp = -1;
                const char *q = p;
                while (*q && *q != '\n' && n < tw) {
                    if (*q == ' ') last_sp = n;
                    q++; n++;
                }
                if (*q && *q != '\n' && last_sp > 0) {
                    n = last_sp;
                    q = p + n;
                }
                sline_t *sl = &g_slines[g_nslines++];
                sl->mi   = i;
                sl->attr = msg_attr;
                snprintf(sl->text, MAX_SLINE_W, "    %.*s", n, p);
                p = q;
                if (*p == '\n') p++;
                else if (*p == ' ' && last_sp > 0) p++;
            }
        }

gap:
        if (i < g_nmsgs - 1 && g_nslines < MAX_SLINES) {
            sline_t *sl = &g_slines[g_nslines++];
            sl->mi   = i;
            sl->attr = 0;
            sl->text[0] = '\0';
        }
    }
}

/* ── Drawing ─────────────────────────────────────────────────────────────── */

static void draw_history(void) {
    werase(g_whist);
    int h, w;
    getmaxyx(g_whist, h, w);

    pthread_mutex_lock(&g_lock);
    build_slines(w);
    int nslines = g_nslines;
    pthread_mutex_unlock(&g_lock);

    int start = nslines - h - g_scroll;
    if (start < 0) start = 0;

    for (int row = 0; row < h && start + row < nslines; row++) {
        sline_t *sl = &g_slines[start + row];
        wmove(g_whist, row, 0);
        if (sl->attr) wattron(g_whist, sl->attr);
        waddstr(g_whist, sl->text);
        if (sl->attr) wattroff(g_whist, sl->attr);
    }

    wrefresh(g_whist);
}

static void draw_status(void) {
    werase(g_wstatus);
    wbkgd(g_wstatus, COLOR_PAIR(CP_STATUS));

    int cols = getmaxx(g_wstatus);

    wattron(g_wstatus, COLOR_PAIR(CP_STATUS) | A_BOLD);
    mvwaddstr(g_wstatus, 0, 1, "mesh");
    wattroff(g_wstatus, A_BOLD);

    time_t now = time(NULL);
    int x = 7;

    pthread_mutex_lock(&g_lock);
    for (int i = 0; i < g_nactors && x < cols - 2; i++) {
        bool alive = (now - g_actors[i].last_seen) < ACTOR_TTL_S;
        char buf[48];
        snprintf(buf, sizeof(buf), "  %s %.28s",
                 alive ? "●" : "○", g_actors[i].id);
        wattron(g_wstatus, alive ? A_BOLD : A_DIM);
        mvwaddstr(g_wstatus, 0, x, buf);
        wattroff(g_wstatus, A_BOLD | A_DIM);
        x += (int)strlen(buf);
    }
    pthread_mutex_unlock(&g_lock);

    wrefresh(g_wstatus);
}

static void draw_input(void) {
    werase(g_winput);
    int cols = getmaxx(g_winput);
    mvwhline(g_winput, 0, 0, ACS_HLINE, cols);

    wattron(g_winput, COLOR_PAIR(CP_INPUT) | A_BOLD);
    mvwaddstr(g_winput, 1, 0, "> ");
    wattroff(g_winput, A_BOLD);

    mvwaddnstr(g_winput, 1, 2, g_input, g_ilen);
    wmove(g_winput, 1, 2 + g_ilen);

    wrefresh(g_winput);
}

static void redraw(void) {
    draw_history();
    draw_status();
    draw_input();
}

/* ── Window layout ───────────────────────────────────────────────────────── */

static void create_windows(void) {
    int rows, cols;
    getmaxyx(stdscr, rows, cols);
    if (rows < 5) rows = 5;

    if (g_whist)   { delwin(g_whist);   g_whist   = NULL; }
    if (g_wstatus) { delwin(g_wstatus); g_wstatus = NULL; }
    if (g_winput)  { delwin(g_winput);  g_winput  = NULL; }

    g_whist   = newwin(rows - 3, cols, 0,        0);
    g_wstatus = newwin(1,        cols, rows - 3, 0);
    g_winput  = newwin(2,        cols, rows - 2, 0);

    keypad(g_winput, TRUE);
}

/* ── Heartbeat JSON parser ────────────────────────────────────────────────── */

static void parse_heartbeat(const char *json) {
    /* payload: {"id":"llm-agent-1","inbox":0,"outbox":0} */
    const char *p = strstr(json, "\"id\":\"");
    if (!p) return;
    p += 6;
    const char *end = strchr(p, '"');
    if (!end) return;
    size_t len = (size_t)(end - p);
    if (len == 0 || len >= 32) return;

    char id[32];
    memcpy(id, p, len);
    id[len] = '\0';

    time_t now = time(NULL);
    pthread_mutex_lock(&g_lock);
    bool found = false;
    for (int i = 0; i < g_nactors; i++) {
        if (strcmp(g_actors[i].id, id) == 0) {
            g_actors[i].last_seen = now;
            found = true;
            break;
        }
    }
    if (!found && g_nactors < MAX_ACTORS) {
        memcpy(g_actors[g_nactors].id, id, len);
        g_actors[g_nactors].id[len] = '\0';
        g_actors[g_nactors].last_seen = now;
        g_nactors++;
    }
    g_dirty = true;
    pthread_mutex_unlock(&g_lock);
}

/* ── Send: simple {type, query} payload, proper tuple header chain ───────── */

static void send_message(const char *query) {
    mpack_writer_t pw;
    mpack_writer_init(&pw, g_send_buf, sizeof(g_send_buf));
    mpack_start_map(&pw, 2);
    mpack_write_cstr(&pw, "type");  mpack_write_cstr(&pw, "user_message");
    mpack_write_cstr(&pw, "query"); mpack_write_cstr(&pw, query);
    mpack_finish_map(&pw);
    size_t plen = mpack_writer_buffer_used(&pw);
    mpack_writer_destroy(&pw);

    /* g_corr_id = session identity; g_last_id = causation (last response id) */
    actor_header_t hdr;
    actor_tuple_init(&hdr, "user_message", "tui",
                     g_corr_id,
                     g_last_id,
                     (uint32_t)plen);
    actor_uuid_gen(hdr.id);
    hdr.ttl = 60LL * 1000000000LL; /* drop if unprocessed after 60s */

    /* assemble header + payload into one contiguous message */
    static uint8_t frame_buf[sizeof(actor_header_t) + MAX_SEND];
    memcpy(frame_buf, &hdr, sizeof(actor_header_t));
    memcpy(frame_buf + sizeof(actor_header_t), g_send_buf, plen);
    nng_send(g_pub, frame_buf, sizeof(actor_header_t) + plen, 0);
}

/* ── NNG receive thread ──────────────────────────────────────────────────── */

static void *recv_thread(void *arg) {
    (void)arg;

    static char ans[MAX_RECV];

    nng_socket_set_ms(g_sub, NNG_OPT_RECVTIMEO, 200);

    while (g_run) {
        nng_msg* msg = NULL;
        int rc = nng_recvmsg(g_sub, &msg, 0);
        if (rc == NNG_ETIMEDOUT) continue;
        if (rc != 0) break;

        void*  body      = nng_msg_body(msg);
        size_t body_len  = nng_msg_len(msg);

        if (body_len < sizeof(actor_header_t)) {
            nng_msg_free(msg);
            continue;
        }

        const actor_header_t *hdr  = (const actor_header_t *)body;
        const char           *payload_ptr = (const char *)body + sizeof(actor_header_t);
        size_t                plen = body_len - sizeof(actor_header_t);

        char topic[33] = {0};
        memcpy(topic, hdr->topic, 32);

        if (strcmp(topic, "heartbeat") == 0) {
            if (plen < sizeof(g_recv_buf)) {
                memcpy(g_recv_buf, payload_ptr, plen);
                g_recv_buf[plen] = '\0';
                parse_heartbeat(g_recv_buf);
            }

        } else if (strcmp(topic, "sql_query") == 0 ||
                   strcmp(topic, "sql_result") == 0) {
            /* update thinking status for our in-flight turn */
            if (memcmp(hdr->correlation_id, g_corr_id, 16) != 0) goto next;

            pthread_mutex_lock(&g_lock);
            for (int i = g_nmsgs - 1; i >= 0; i--) {
                if (g_msgs[i].role == ROLE_THINKING) {
                    snprintf(g_msgs[i].detail, MAX_DETAIL, "%s",
                             strcmp(topic, "sql_query") == 0
                             ? "querying database..."
                             : "processing results...");
                    break;
                }
            }
            g_dirty = true;
            pthread_mutex_unlock(&g_lock);

        } else if (strcmp(topic, "agent_response") == 0) {
            /* match by correlation_id — no mpack session_id needed */
            if (memcmp(hdr->correlation_id, g_corr_id, 16) != 0) goto next;

            ans[0] = '\0';
            mpack_reader_t r;
            mpack_reader_init_data(&r, payload_ptr, plen);
            static const char *keys[] = { "type", "answer" };
            bool found[2] = { false, false };
            uint32_t sz = mpack_expect_map_max(&r, 4);
            for (uint32_t k = 0; k < sz && mpack_reader_error(&r) == mpack_ok; k++) {
                switch (mpack_expect_key_cstr(&r, keys, found, 2)) {
                    case 0: mpack_discard(&r); break;
                    case 1: mpack_expect_cstr(&r, ans, sizeof(ans)); break;
                    default: mpack_discard(&r); break;
                }
            }
            mpack_done_map(&r);
            mpack_reader_destroy(&r);

            pthread_mutex_lock(&g_lock);

            /* this response id becomes the causation_id for the next send */
            memcpy(g_last_id, hdr->id, 16);

            /* replace the THINKING placeholder */
            bool replaced = false;
            for (int i = g_nmsgs - 1; i >= 0; i--) {
                if (g_msgs[i].role == ROLE_THINKING) {
                    g_msgs[i].role = ROLE_AGENT;
                    memcpy(g_msgs[i].text, ans, MAX_MSG - 1);
                    g_msgs[i].text[MAX_MSG - 1] = '\0';
                    replaced = true;
                    break;
                }
            }
            if (!replaced && g_nmsgs < MAX_HISTORY) {
                g_msgs[g_nmsgs].role = ROLE_AGENT;
                memcpy(g_msgs[g_nmsgs].text, ans, MAX_MSG - 1);
                g_msgs[g_nmsgs++].text[MAX_MSG - 1] = '\0';
            }
            g_thinking = false;
            g_scroll   = 0;
            g_dirty    = true;
            pthread_mutex_unlock(&g_lock);
        }

next:
        nng_msg_free(msg);
    }

    return NULL;
}

/* ── Signal handling ─────────────────────────────────────────────────────── */

static volatile bool g_resize = false;

static void on_signal(int sig) {
    if (sig == SIGWINCH)                 g_resize = true;
    if (sig == SIGINT || sig == SIGTERM) g_run    = false;
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(void) {
    /* session correlation_id: one UUID for the lifetime of this TUI instance */
    actor_uuid_gen(g_corr_id);
    memset(g_last_id, 0, 16); /* no causation for the first message */

    int rc;

    if ((rc = nng_sub0_open(&g_sub)) != 0) {
        fprintf(stderr, "[tui] nng_sub0_open: %s\n", nng_strerror(rc));
        return 1;
    }
    if ((rc = nng_dial(g_sub, BUS_SUB, NULL, 0)) != 0) {
        fprintf(stderr, "[tui] sub dial: %s\n", nng_strerror(rc));
        return 1;
    }
    nng_socket_set(g_sub, NNG_OPT_SUB_SUBSCRIBE, "agent_response", 15);
    nng_socket_set(g_sub, NNG_OPT_SUB_SUBSCRIBE, "heartbeat",       9);
    nng_socket_set(g_sub, NNG_OPT_SUB_SUBSCRIBE, "sql_query",       9);
    nng_socket_set(g_sub, NNG_OPT_SUB_SUBSCRIBE, "sql_result",     10);

    if ((rc = nng_pub0_open(&g_pub)) != 0) {
        fprintf(stderr, "[tui] nng_pub0_open: %s\n", nng_strerror(rc));
        return 1;
    }
    if ((rc = nng_dial(g_pub, BUS_PUB, NULL, 0)) != 0) {
        fprintf(stderr, "[tui] pub dial: %s\n", nng_strerror(rc));
        return 1;
    }

    /* warm-up: let NNG establish connections before sending */
    struct timespec warm = { .tv_sec = 0, .tv_nsec = 500000000 };
    nanosleep(&warm, NULL);

    setlocale(LC_ALL, "");
    initscr();
    cbreak();
    noecho();
    curs_set(1);
    start_color();
    use_default_colors();

    init_pair(CP_USER,   COLOR_CYAN,   -1);
    init_pair(CP_AGENT,  COLOR_GREEN,  -1);
    init_pair(CP_THINK,  COLOR_YELLOW, -1);
    init_pair(CP_STATUS, COLOR_BLACK,  COLOR_WHITE);
    init_pair(CP_INPUT,  COLOR_WHITE,  -1);

    create_windows();
    halfdelay(1);

    signal(SIGWINCH, on_signal);
    signal(SIGINT,   on_signal);
    signal(SIGTERM,  on_signal);

    pthread_t tid;
    pthread_create(&tid, NULL, recv_thread, NULL);

    redraw();

    while (g_run) {
        if (g_resize) {
            g_resize = false;
            endwin();
            refresh();
            clear();
            create_windows();
            redraw();
        }

        pthread_mutex_lock(&g_lock);
        bool dirty    = g_dirty;
        bool thinking = g_thinking;
        g_dirty = false;
        pthread_mutex_unlock(&g_lock);

        if (dirty) {
            redraw();
        } else if (thinking) {
            g_tick++;
            pthread_mutex_lock(&g_lock);
            bool still = g_thinking;
            pthread_mutex_unlock(&g_lock);
            if (still) {
                draw_history();
                draw_input();
            } else {
                redraw();
            }
        }

        int ch = wgetch(g_winput);
        if (ch == ERR) continue;

        switch (ch) {
        case '\n':
        case KEY_ENTER: {
            if (g_ilen == 0) break;
            pthread_mutex_lock(&g_lock);
            if (g_thinking) { pthread_mutex_unlock(&g_lock); break; }
            g_input[g_ilen] = '\0';
            if (g_nmsgs < MAX_HISTORY) {
                g_msgs[g_nmsgs].role     = ROLE_USER;
                g_msgs[g_nmsgs].detail[0] = '\0';
                snprintf(g_msgs[g_nmsgs++].text, MAX_MSG, "%s", g_input);
            }
            if (g_nmsgs < MAX_HISTORY) {
                g_msgs[g_nmsgs].role     = ROLE_THINKING;
                g_msgs[g_nmsgs].text[0]  = '\0';
                g_msgs[g_nmsgs].detail[0] = '\0';
                g_nmsgs++;
            }
            g_thinking = true;
            g_scroll   = 0;
            g_dirty    = true;
            pthread_mutex_unlock(&g_lock);

            send_message(g_input);
            g_input[0] = '\0';
            g_ilen     = 0;
            break;
        }

        case KEY_BACKSPACE:
        case 127:
        case '\b':
            if (g_ilen > 0) { g_ilen--; draw_input(); }
            break;

        case KEY_PPAGE:
            g_scroll += getmaxy(g_whist) / 2;
            if (g_scroll > g_nslines) g_scroll = g_nslines;
            draw_history(); draw_input();
            break;

        case KEY_NPAGE:
            g_scroll -= getmaxy(g_whist) / 2;
            if (g_scroll < 0) g_scroll = 0;
            draw_history(); draw_input();
            break;

        case KEY_UP:
            g_scroll += 3;
            if (g_scroll > g_nslines) g_scroll = g_nslines;
            draw_history(); draw_input();
            break;

        case KEY_DOWN:
            g_scroll -= 3;
            if (g_scroll < 0) g_scroll = 0;
            draw_history(); draw_input();
            break;

        default:
            if (ch >= 32 && ch < 127 && g_ilen < TUI_MAX_INPUT) {
                g_input[g_ilen++] = (char)ch;
                draw_input();
            }
            break;
        }
    }

    g_run = false;
    pthread_join(tid, NULL);

    delwin(g_whist);
    delwin(g_wstatus);
    delwin(g_winput);
    endwin();

    nng_close(g_pub);
    nng_close(g_sub);

    return 0;
}
