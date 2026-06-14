#include "../state.h"
#include "actors.h"

static void stat_card(int z, const char *value, const char *label, float sc) {
    CLAY(CLAY_SIDI(cs("StatCard"), (uint32_t)z), {
        .layout = {
            .sizing          = { CLAY_SIZING_GROW(0), CLAY_SIZING_FIT(0) },
            .layoutDirection = CLAY_TOP_TO_BOTTOM,
            .childGap        = (uint16_t)(4 * sc),
            .padding         = { (uint16_t)(16*sc), (uint16_t)(16*sc),
                                 (uint16_t)(14*sc), (uint16_t)(14*sc) },
        },
        .backgroundColor = WHITE,
        .border = { .width = { 0, 1, 0, 0, 0 }, .color = BORDER },
    }) {
        CLAY_TEXT(cs(value),
            CLAY_TEXT_CONFIG({ .fontId = 0, .fontSize = (uint16_t)(30*sc),
                               .textColor = ACCENT }));
        CLAY_TEXT(cs(label),
            CLAY_TEXT_CONFIG({ .fontId = 1, .fontSize = (uint16_t)(11*sc),
                               .textColor = MUTED }));
    }
}

void build_actors(float sc) {
    CLAY(CLAY_ID("ActorScroll"), {
        .layout = {
            .sizing          = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) },
            .layoutDirection = CLAY_TOP_TO_BOTTOM,
            .padding         = { (uint16_t)(20*sc), (uint16_t)(20*sc),
                                 (uint16_t)(16*sc), (uint16_t)(16*sc) },
            .childGap        = (uint16_t)(16 * sc),
        },
        .clip = { .vertical = true, .childOffset = Clay_GetScrollOffset() },
    }) {

        /* ── stat cards ───────────────────────────────────────────── */
        CLAY(CLAY_ID("ActorStats"), {
            .layout = {
                .sizing          = { CLAY_SIZING_GROW(0), CLAY_SIZING_FIT(0) },
                .layoutDirection = CLAY_LEFT_TO_RIGHT,
            },
            .border = { .width = { 1, 1, 1, 1, 0 }, .color = BORDER },
            .cornerRadius = CLAY_CORNER_RADIUS(6 * sc),
        }) {
            stat_card(0, s_alive_count,               "ACTIVE ACTORS", sc);
            stat_card(1, s_event_count,               "EVENTS SEEN",   sc);
            stat_card(2, g_connected ? "live" : "off", "BUS",          sc);
        }

        /* ── section label ────────────────────────────────────────── */
        CLAY_TEXT(cs("ACTORS"),
            CLAY_TEXT_CONFIG({ .fontId = 1, .fontSize = (uint16_t)(11*sc),
                               .textColor = MUTED }));

        /* ── empty state or actor table ──────────────────────────── */
        if (g_nactors == 0) {
            CLAY_AUTO_ID({
                .layout = {
                    .sizing         = { CLAY_SIZING_GROW(0), CLAY_SIZING_FIT(0) },
                    .childAlignment = { .x = CLAY_ALIGN_X_CENTER },
                    .padding        = { 0, 0, (uint16_t)(20*sc), (uint16_t)(20*sc) },
                },
            }) {
                CLAY_TEXT(cs("Waiting for actors to connect\xe2\x80\xa6"),
                    CLAY_TEXT_CONFIG({ .fontId = 1, .fontSize = (uint16_t)(14*sc),
                                       .textColor = MUTED }));
            }
        } else {

        /* ── actor table ──────────────────────────────────────────── */
        CLAY(CLAY_ID("ActorTable"), {
            .layout = {
                .sizing          = { CLAY_SIZING_GROW(0), CLAY_SIZING_FIT(0) },
                .layoutDirection = CLAY_TOP_TO_BOTTOM,
            },
            .backgroundColor = WHITE,
            .border          = { .width = { 1, 1, 1, 1, 0 }, .color = BORDER },
            .cornerRadius    = CLAY_CORNER_RADIUS(6 * sc),
        }) {

            /* header row */
            CLAY(CLAY_ID("ActorTblHdr"), {
                .layout = {
                    .sizing          = { CLAY_SIZING_GROW(0), CLAY_SIZING_FIT(0) },
                    .layoutDirection = CLAY_LEFT_TO_RIGHT,
                    .padding         = { (uint16_t)(12*sc), (uint16_t)(12*sc),
                                         (uint16_t)(6*sc),  (uint16_t)(6*sc) },
                    .childGap        = (uint16_t)(8 * sc),
                },
                .backgroundColor = DARK,
            }) {
                static const char *cols[] = { "Actor", "Inbox", "Outbox", "Last Beat", "Status" };
                static const int   widths[] = { 0, 60, 60, 70, 60 }; /* 0 = grow */
                for (int c = 0; c < 5; c++) {
                    Clay_SizingAxis w = widths[c]
                        ? CLAY_SIZING_FIXED((float)(widths[c] * sc))
                        : CLAY_SIZING_GROW(0);
                    CLAY_AUTO_ID({
                        .layout = { .sizing = { w, CLAY_SIZING_FIT(0) } },
                    }) {
                        CLAY_TEXT(cs(cols[c]),
                            CLAY_TEXT_CONFIG({ .fontId = 1, .fontSize = (uint16_t)(11*sc),
                                               .textColor = { 0x77, 0x77, 0x77, 255 } }));
                    }
                }
            }

            /* data rows */
            for (int i = 0; i < g_nactors; i++) {
                bool hovered = Clay_Hovered();
                Clay_Color row_bg = hovered     ? SURFACE
                                  : i % 2 == 0 ? WHITE
                                  :               BG;

                CLAY(CLAY_SIDI(cs("ActorRow"), (uint32_t)i), {
                    .layout = {
                        .sizing          = { CLAY_SIZING_GROW(0), CLAY_SIZING_FIT(0) },
                        .layoutDirection = CLAY_LEFT_TO_RIGHT,
                        .childGap        = (uint16_t)(8 * sc),
                        .padding         = { (uint16_t)(12*sc), (uint16_t)(12*sc),
                                             (uint16_t)(8*sc),  (uint16_t)(8*sc) },
                    },
                    .backgroundColor = row_bg,
                    .border = { .width = { 0, 0, 1, 0, 0 }, .color = BORDER },
                }) {
                    /* actor name (GROW) */
                    CLAY_AUTO_ID({ .layout = { .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_FIT(0) } } }) {
                        CLAY_TEXT(cs(s_actor_label[i]),
                            CLAY_TEXT_CONFIG({ .fontId = 2, .fontSize = (uint16_t)(13*sc),
                                               .textColor = g_actors[i].alive ? DIM : MUTED }));
                    }
                    /* inbox */
                    CLAY_AUTO_ID({ .layout = { .sizing = { CLAY_SIZING_FIXED(60*sc), CLAY_SIZING_FIT(0) } } }) {
                        CLAY_TEXT(cs(s_actor_inbox[i]),
                            CLAY_TEXT_CONFIG({ .fontId = 2, .fontSize = (uint16_t)(13*sc),
                                               .textColor = MUTED }));
                    }
                    /* outbox */
                    CLAY_AUTO_ID({ .layout = { .sizing = { CLAY_SIZING_FIXED(60*sc), CLAY_SIZING_FIT(0) } } }) {
                        CLAY_TEXT(cs(s_actor_outbox[i]),
                            CLAY_TEXT_CONFIG({ .fontId = 2, .fontSize = (uint16_t)(13*sc),
                                               .textColor = MUTED }));
                    }
                    /* last beat */
                    CLAY_AUTO_ID({ .layout = { .sizing = { CLAY_SIZING_FIXED(70*sc), CLAY_SIZING_FIT(0) } } }) {
                        CLAY_TEXT(cs(s_actor_age[i]),
                            CLAY_TEXT_CONFIG({ .fontId = 2, .fontSize = (uint16_t)(13*sc),
                                               .textColor = MUTED }));
                    }
                    /* status pill */
                    bool alive = g_actors[i].alive;
                    CLAY_AUTO_ID({
                        .layout = {
                            .sizing  = { CLAY_SIZING_FIXED(60*sc), CLAY_SIZING_FIT(0) },
                            .childAlignment = { .x = CLAY_ALIGN_X_LEFT },
                        },
                    }) {
                        CLAY_AUTO_ID({
                            .layout = {
                                .sizing  = { CLAY_SIZING_FIT(0), CLAY_SIZING_FIT(0) },
                                .padding = { (uint16_t)(6*sc), (uint16_t)(6*sc),
                                             (uint16_t)(2*sc), (uint16_t)(2*sc) },
                            },
                            .backgroundColor = alive
                                ? (Clay_Color){0xe8, 0xf8, 0xee, 255}
                                : (Clay_Color){0xf2, 0xf2, 0xf2, 255},
                            .cornerRadius = CLAY_CORNER_RADIUS(4 * sc),
                            .border = {
                                .width = { 1, 1, 1, 1, 0 },
                                .color = alive
                                    ? (Clay_Color){0x86, 0xef, 0xac, 255}
                                    : (Clay_Color){0xdd, 0xdd, 0xdd, 255},
                            },
                        }) {
                            CLAY_TEXT(cs(alive ? "alive" : "offline"),
                                CLAY_TEXT_CONFIG({ .fontId = 1, .fontSize = (uint16_t)(11*sc),
                                                   .textColor = alive
                                                       ? (Clay_Color){0x16, 0xa3, 0x4a, 255}
                                                       : MUTED }));
                        }
                    }
                }
            } /* rows */

        } /* ActorTable */

        } /* else g_nactors > 0 */

    } /* ActorScroll */
}
