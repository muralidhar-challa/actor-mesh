#include "state.h"
#include <SDL2/SDL.h>

/* ── Global state definitions ────────────────────────────────────── */

Actor      g_actors[MAX_ACTORS];
int        g_nactors   = 0;

TupleEvent g_events[MAX_EVENTS];
int        g_nevents   = 0;
int        g_expanded  = -1;

int        g_tab       = 0;
bool       g_connected = false;

char       g_pub_topic[128]       = {0};
int        g_pub_topic_len        = 0;
char       g_pub_payload[UI_MAX_INPUT] = {0};
int        g_pub_payload_len      = 0;
int        g_pub_focus            = 0;

int        g_tick      = 0;

/* ── Per-frame display string buffer definitions ─────────────────── */

char s_alive_count[8];
char s_event_count[8];

char s_actor_label[MAX_ACTORS][48];
char s_actor_inbox[MAX_ACTORS][16];
char s_actor_outbox[MAX_ACTORS][16];
char s_actor_age[MAX_ACTORS][16];

int  s_disp_count = 0;
char s_evt_topic[MAX_EVENTS][48];
char s_evt_origin[MAX_EVENTS][40];
char s_evt_age[MAX_EVENTS][16];
char s_evt_preview[MAX_EVENTS][80];

char s_exp_topic[48];
char s_exp_origin[40];
char s_exp_id[20];
char s_exp_corr[20];
char s_exp_caus[20];
char s_exp_full[512];

char s_pub_topic_disp[132];
char s_pub_payload_disp[UI_MAX_INPUT + 4];

/* ── prepare_strings ─────────────────────────────────────────────── */

void prepare_strings(void) {
    int alive = 0;
    for (int i = 0; i < g_nactors; i++)
        if (g_actors[i].alive) alive++;

    snprintf(s_alive_count, sizeof(s_alive_count), "%d", alive);
    snprintf(s_event_count, sizeof(s_event_count), "%d", g_nevents);

    uint32_t now = SDL_GetTicks();

    /* actors — alive = heartbeat received within 3× heartbeat interval (6s) */
    for (int i = 0; i < g_nactors; i++) {
        uint32_t hb_age = (g_actors[i].last_hb_ticks > 0)
                          ? (now - g_actors[i].last_hb_ticks)
                          : UINT32_MAX;
        g_actors[i].alive = hb_age < 6000;

        snprintf(s_actor_label[i], sizeof(s_actor_label[i]), "%.31s", g_actors[i].id);
        fmt_bytes(g_actors[i].inbox,  s_actor_inbox[i]);
        fmt_bytes(g_actors[i].outbox, s_actor_outbox[i]);

        if (g_actors[i].last_hb_ticks == 0) {
            s_actor_age[i][0] = '\xe2'; s_actor_age[i][1] = '\x80';
            s_actor_age[i][2] = '\x94'; s_actor_age[i][3] = '\0'; /* — */
        } else {
            uint32_t ms = now - g_actors[i].last_hb_ticks;
            if (ms < 1000)
                snprintf(s_actor_age[i], 16, "%ums",  ms);
            else
                snprintf(s_actor_age[i], 16, "%.1fs", ms / 1000.0f);
        }
    }

    /* bus ring buffer — display order: 0 = newest */
    s_disp_count = g_nevents < MAX_EVENTS ? g_nevents : MAX_EVENTS;
    for (int di = 0; di < s_disp_count; di++) {
        int ri = ((g_nevents - 1 - di) % MAX_EVENTS + MAX_EVENTS) % MAX_EVENTS;
        TupleEvent *e = &g_events[ri];

        uint32_t ms = now - e->received_ticks;
        if (ms < 1000)
            snprintf(s_evt_age[di], 16, "%ums", ms);
        else if (ms < 60000)
            snprintf(s_evt_age[di], 16, "%.1fs", ms / 1000.0f);
        else
            snprintf(s_evt_age[di], 16, "%um",  ms / 60000);

        strncpy(s_evt_topic[di],   e->topic,   47); s_evt_topic[di][47]   = '\0';
        strncpy(s_evt_origin[di],  e->origin,  39); s_evt_origin[di][39]  = '\0';
        strncpy(s_evt_preview[di], e->preview, 79); s_evt_preview[di][79] = '\0';

        if (s_evt_origin[di][0] == '\0')
            strcpy(s_evt_origin[di], "\xe2\x80\x94"); /* — */
    }

    /* expanded row detail */
    if (g_expanded >= 0 && g_expanded < s_disp_count) {
        int ri = ((g_nevents - 1 - g_expanded) % MAX_EVENTS + MAX_EVENTS) % MAX_EVENTS;
        TupleEvent *e = &g_events[ri];
        strncpy(s_exp_topic,  e->topic,   47); s_exp_topic[47]  = '\0';
        strncpy(s_exp_origin, e->origin,  39); s_exp_origin[39] = '\0';
        snprintf(s_exp_id,   sizeof(s_exp_id),   "%s\xe2\x80\xa6", e->id_hex);
        snprintf(s_exp_corr, sizeof(s_exp_corr), "%s\xe2\x80\xa6", e->corr_hex);
        snprintf(s_exp_caus, sizeof(s_exp_caus), "%s\xe2\x80\xa6", e->caus_hex);
        snprintf(s_exp_full, sizeof(s_exp_full), "%.511s", e->full);
    }

    /* publish fields */
    bool blink = (g_tick / 20) % 2 == 0;

    if (g_pub_focus == 0) {
        snprintf(s_pub_topic_disp, sizeof(s_pub_topic_disp), "%s%s",
                 g_pub_topic, blink ? "|" : " ");
    } else {
        snprintf(s_pub_topic_disp, sizeof(s_pub_topic_disp), "%s", g_pub_topic);
    }

    if (g_pub_focus == 1) {
        snprintf(s_pub_payload_disp, sizeof(s_pub_payload_disp), "%s%s",
                 g_pub_payload, blink ? "|" : " ");
    } else {
        snprintf(s_pub_payload_disp, sizeof(s_pub_payload_disp), "%s", g_pub_payload);
    }
}
