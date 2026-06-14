#pragma once
#define _POSIX_C_SOURCE 200809L

#include "../../../vendor/clay.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* ── Limits ──────────────────────────────────────────────────────── */

#define MAX_ACTORS    16
#define MAX_EVENTS   256
#define UI_MAX_INPUT 2048

/* ── Types ───────────────────────────────────────────────────────── */

typedef struct {
    char     id[32];
    int      inbox, outbox;
    bool     alive;
    uint32_t last_hb_ticks;
} Actor;

typedef struct {
    char     topic[33];
    char     origin[33];
    char     id_hex[9];    /* first 4 bytes as 8 hex chars */
    char     corr_hex[9];
    char     caus_hex[9];
    char     preview[129]; /* first 128 printable payload bytes */
    char     full[4097];   /* first 4096 printable payload bytes */
    uint32_t received_ticks;
    int      payload_len;
} TupleEvent;

/* ── Design tokens ───────────────────────────────────────────────── */

static const Clay_Color BG       = {0xf9, 0xf9, 0xf8, 255};
static const Clay_Color DARK     = {0x0a, 0x0a, 0x0a, 255};
static const Clay_Color ACCENT   = {0xc4, 0x1e, 0x3a, 255};
static const Clay_Color ACCENT_H = {0xe0, 0x2a, 0x46, 255};
static const Clay_Color DIM      = {0x2a, 0x2a, 0x2a, 255};
static const Clay_Color MUTED    = {0x88, 0x88, 0x88, 255};
static const Clay_Color WHITE    = {255,  255,  255,  255};
static const Clay_Color GREEN    = {0x22, 0xc5, 0x5e, 255};
static const Clay_Color TRANS    = {0,    0,    0,    0  };
static const Clay_Color SURFACE  = {0xf0, 0xf0, 0xee, 255};
static const Clay_Color BORDER   = {0xe2, 0xe2, 0xe2, 255};

/* ── Shared global state ─────────────────────────────────────────── */

extern Actor      g_actors[MAX_ACTORS];
extern int        g_nactors;

extern TupleEvent g_events[MAX_EVENTS];
extern int        g_nevents;
extern int        g_expanded;   /* bus display-index expanded, -1 = none */

extern int        g_tab;        /* 0=Actors 1=Bus 2=Publish */
extern bool       g_connected;

extern char       g_pub_topic[128];
extern int        g_pub_topic_len;
extern char       g_pub_payload[UI_MAX_INPUT];
extern int        g_pub_payload_len;
extern int        g_pub_focus;  /* 0=topic field, 1=payload field */

extern int        g_tick;       /* frame counter */

/* ── Per-frame display string buffers ────────────────────────────── */
/* Written by prepare_strings() before Clay_BeginLayout each frame.  */
/* Clay stores pointers — these must stay valid through SDL render.   */

extern char s_alive_count[8];
extern char s_event_count[8];

extern char s_actor_label[MAX_ACTORS][48];
extern char s_actor_inbox[MAX_ACTORS][16];
extern char s_actor_outbox[MAX_ACTORS][16];
extern char s_actor_age[MAX_ACTORS][16];

extern int  s_disp_count;
extern char s_evt_topic[MAX_EVENTS][48];
extern char s_evt_origin[MAX_EVENTS][40];
extern char s_evt_age[MAX_EVENTS][16];
extern char s_evt_preview[MAX_EVENTS][80];

extern char s_exp_topic[48];
extern char s_exp_origin[40];
extern char s_exp_id[20];
extern char s_exp_corr[20];
extern char s_exp_caus[20];
extern char s_exp_full[512];

extern char s_pub_topic_disp[132];
extern char s_pub_payload_disp[UI_MAX_INPUT + 4];

/* ── Helpers ─────────────────────────────────────────────────────── */

static inline Clay_String cs(const char *s) {
    return (Clay_String){ .length = (int32_t)strlen(s), .chars = s };
}

static inline void fmt_bytes(int n, char buf[16]) {
    if (n < 1024) snprintf(buf, 16, "%d B", n);
    else          snprintf(buf, 16, "%.1fK", n / 1024.0f);
}

static inline Clay_Color topic_color(const char *topic) {
    if (strcmp(topic, "heartbeat") == 0)
        return (Clay_Color){0x88, 0x88, 0x88, 255};
    int h = 0;
    for (const char *p = topic; *p; p++) h += (unsigned char)*p;
    static const Clay_Color pal[8] = {
        {0x3b, 0x82, 0xf6, 255},
        {0x10, 0xb9, 0x81, 255},
        {0xf5, 0x9e, 0x0b, 255},
        {0x8b, 0x5c, 0xf6, 255},
        {0xef, 0x44, 0x44, 255},
        {0x06, 0xb6, 0xd4, 255},
        {0xf9, 0x73, 0x16, 255},
        {0x84, 0xcc, 0x16, 255},
    };
    return pal[((unsigned)h) % 8];
}

/* ── Layout macros ───────────────────────────────────────────────── */

/* Grow-to-fill spacer — auto ID is unique per call site via __COUNTER__ */
#define SPACER() \
    CLAY_AUTO_ID({ .layout = { .sizing = { CLAY_SIZING_GROW(0) } } }) {}

/* 1px horizontal rule */
#define HRULE(sc) \
    CLAY_AUTO_ID({ \
        .layout = { .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_FIXED(1*(sc)) } }, \
        .backgroundColor = BORDER, \
    }) {}

/* ── prepare_strings ─────────────────────────────────────────────── */

void prepare_strings(void);
