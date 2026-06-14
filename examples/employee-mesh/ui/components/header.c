#include "../state.h"
#include "header.h"

void build_header(float sc) {
    int alive = 0;
    for (int i = 0; i < g_nactors; i++)
        if (g_actors[i].alive) alive++;

    static char s_status[64];
    if (g_connected) {
        snprintf(s_status, sizeof(s_status),
                 "%d actor%s  \xc2\xb7  %s event%s",
                 alive,  alive == 1    ? "" : "s",
                 s_event_count,
                 g_nevents == 1 ? "" : "s");
    } else {
        snprintf(s_status, sizeof(s_status), "disconnected");
    }

    static const struct { const char *label; int idx; } tabs[] = {
        { "Actors",  0 },
        { "Bus",     1 },
        { "Publish", 2 },
    };

    CLAY(CLAY_ID("Header"), {
        .layout = {
            .sizing          = { CLAY_SIZING_GROW(0), CLAY_SIZING_FIT(0) },
            .layoutDirection = CLAY_LEFT_TO_RIGHT,
            .childGap        = (uint16_t)(8 * sc),
            .childAlignment  = { .y = CLAY_ALIGN_Y_CENTER },
            .padding         = { (uint16_t)(16*sc), (uint16_t)(16*sc),
                                 (uint16_t)(9*sc),  (uint16_t)(9*sc) },
        },
        .backgroundColor = DARK,
    }) {
        /* logo */
        CLAY_TEXT(cs("Actor Mesh"),
            CLAY_TEXT_CONFIG({ .fontId = 0, .fontSize = (uint16_t)(17*sc),
                               .textColor = WHITE }));

        SPACER()

        /* connection + counts */
        CLAY_TEXT(cs(s_status),
            CLAY_TEXT_CONFIG({ .fontId = 1, .fontSize = (uint16_t)(13*sc),
                               .textColor = g_connected ? GREEN : MUTED }));

        SPACER()

        /* tab buttons */
        for (int t = 0; t < 3; t++) {
            bool active = (g_tab == tabs[t].idx);
            CLAY(CLAY_SIDI(cs("Tab"), (uint32_t)t), {
                .layout = {
                    .childAlignment = { .y = CLAY_ALIGN_Y_CENTER },
                    .padding = { (uint16_t)(11*sc), (uint16_t)(11*sc),
                                 (uint16_t)(5*sc),  (uint16_t)(5*sc) },
                },
                .backgroundColor = active          ? ACCENT
                                  : Clay_Hovered() ? (Clay_Color){0x22,0x22,0x22,255}
                                  :                  TRANS,
                .cornerRadius    = CLAY_CORNER_RADIUS(5 * sc),
            }) {
                CLAY_TEXT(cs(tabs[t].label),
                    CLAY_TEXT_CONFIG({ .fontId = 1, .fontSize = (uint16_t)(13*sc),
                                       .textColor = active ? WHITE : MUTED }));
            }
        }
    }
}
