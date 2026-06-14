#include "../state.h"
#include "bus.h"

/* Colored topic pill — fully opaque tints to avoid SDL2 alpha-blend issues */
static void topic_pill(const char *topic, float sc) {
    Clay_Color col = topic_color(topic);
    /* light bg: mix color with white at ~12% saturation */
    Clay_Color bg = {
        (uint8_t)(255 - (255 - col.r) / 7),
        (uint8_t)(255 - (255 - col.g) / 7),
        (uint8_t)(255 - (255 - col.b) / 7),
        255
    };
    /* border: halfway between color and white */
    Clay_Color bdr = {
        (uint8_t)(col.r / 2 + 128),
        (uint8_t)(col.g / 2 + 128),
        (uint8_t)(col.b / 2 + 128),
        255
    };
    /* text: darken color slightly for contrast */
    Clay_Color txt = {
        (uint8_t)(col.r * 3 / 4),
        (uint8_t)(col.g * 3 / 4),
        (uint8_t)(col.b * 3 / 4),
        255
    };

    CLAY_AUTO_ID({
        .layout = {
            .sizing  = { CLAY_SIZING_FIT(0), CLAY_SIZING_FIT(0) },
            .padding = { (uint16_t)(6*sc), (uint16_t)(6*sc),
                         (uint16_t)(2*sc), (uint16_t)(2*sc) },
        },
        .backgroundColor = bg,
        .cornerRadius    = CLAY_CORNER_RADIUS(3 * sc),
        .border          = { .width = { 1, 1, 1, 1, 0 }, .color = bdr },
    }) {
        CLAY_TEXT(cs(topic),
            CLAY_TEXT_CONFIG({ .fontId = 2, .fontSize = (uint16_t)(12*sc),
                               .textColor = txt }));
    }
}

/* Detail row: "label  value" in mono */
static void detail_kv(const char *label, const char *value, float sc) {
    CLAY_AUTO_ID({
        .layout = {
            .sizing          = { CLAY_SIZING_GROW(0), CLAY_SIZING_FIT(0) },
            .layoutDirection = CLAY_LEFT_TO_RIGHT,
            .childGap        = (uint16_t)(8 * sc),
        },
    }) {
        CLAY_AUTO_ID({ .layout = { .sizing = { CLAY_SIZING_FIXED(56*sc), CLAY_SIZING_FIT(0) } } }) {
            CLAY_TEXT(cs(label),
                CLAY_TEXT_CONFIG({ .fontId = 2, .fontSize = (uint16_t)(12*sc),
                                   .textColor = MUTED }));
        }
        CLAY_TEXT(cs(value),
            CLAY_TEXT_CONFIG({ .fontId = 2, .fontSize = (uint16_t)(12*sc),
                               .textColor = DIM }));
    }
}

void build_bus(float sc) {
    CLAY(CLAY_ID("BusScroll"), {
        .layout = {
            .sizing          = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) },
            .layoutDirection = CLAY_TOP_TO_BOTTOM,
        },
        .clip = { .vertical = true, .childOffset = Clay_GetScrollOffset() },
    }) {

        /* ── toolbar ──────────────────────────────────────────────── */
        CLAY(CLAY_ID("BusToolbar"), {
            .layout = {
                .sizing          = { CLAY_SIZING_GROW(0), CLAY_SIZING_FIT(0) },
                .layoutDirection = CLAY_LEFT_TO_RIGHT,
                .childAlignment  = { .y = CLAY_ALIGN_Y_CENTER },
                .padding         = { (uint16_t)(16*sc), (uint16_t)(16*sc),
                                     (uint16_t)(10*sc), (uint16_t)(10*sc) },
                .childGap        = (uint16_t)(8 * sc),
            },
            .backgroundColor = WHITE,
            .border = { .width = { 0, 0, 0, 1, 0 }, .color = BORDER },
        }) {
            static char s_count[32];
            snprintf(s_count, sizeof(s_count), "%d events", g_nevents);
            CLAY_TEXT(cs(s_count),
                CLAY_TEXT_CONFIG({ .fontId = 1, .fontSize = (uint16_t)(13*sc),
                                   .textColor = MUTED }));
            SPACER()
            CLAY_TEXT(cs("newest first"),
                CLAY_TEXT_CONFIG({ .fontId = 1, .fontSize = (uint16_t)(11*sc),
                                   .textColor = (Clay_Color){0xbb,0xbb,0xbb,255} }));
        }

        /* ── empty state or event rows ───────────────────────────── */
        if (s_disp_count == 0) {
            CLAY_AUTO_ID({
                .layout = {
                    .sizing         = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) },
                    .childAlignment = { .x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_CENTER },
                    .layoutDirection= CLAY_TOP_TO_BOTTOM,
                    .childGap       = (uint16_t)(8 * sc),
                },
            }) {
                CLAY_TEXT(cs("Bus is quiet"),
                    CLAY_TEXT_CONFIG({ .fontId = 0, .fontSize = (uint16_t)(22*sc),
                                       .textColor = { 0xd0, 0xd0, 0xce, 255 } }));
                CLAY_TEXT(cs("No tuples received yet"),
                    CLAY_TEXT_CONFIG({ .fontId = 1, .fontSize = (uint16_t)(14*sc),
                                       .textColor = MUTED }));
            }
        } else {

        /* ── event rows ───────────────────────────────────────────── */
        for (int di = 0; di < s_disp_count; di++) {
            bool expanded = (g_expanded == di);

            /* outer container: row + optional detail */
            CLAY(CLAY_SIDI(cs("BusEntry"), (uint32_t)di), {
                .layout = {
                    .sizing          = { CLAY_SIZING_GROW(0), CLAY_SIZING_FIT(0) },
                    .layoutDirection = CLAY_TOP_TO_BOTTOM,
                },
                .backgroundColor = expanded ? (Clay_Color){0xf4,0xf4,0xff,255} : WHITE,
                .border = { .width = { 0, 0, 0, 1, 0 }, .color = BORDER },
            }) {

                /* ── main row (click target) ──────────────────────── */
                bool hovered = Clay_Hovered();
                CLAY(CLAY_SIDI(cs("BusRow"), (uint32_t)di), {
                    .layout = {
                        .sizing          = { CLAY_SIZING_GROW(0), CLAY_SIZING_FIT(0) },
                        .layoutDirection = CLAY_LEFT_TO_RIGHT,
                        .childGap        = (uint16_t)(10 * sc),
                        .childAlignment  = { .y = CLAY_ALIGN_Y_CENTER },
                        .padding         = { (uint16_t)(14*sc), (uint16_t)(14*sc),
                                             (uint16_t)(7*sc),  (uint16_t)(7*sc) },
                    },
                    .backgroundColor = hovered && !expanded
                        ? SURFACE : TRANS,
                }) {
                    /* age */
                    CLAY_AUTO_ID({ .layout = { .sizing = { CLAY_SIZING_FIXED(46*sc), CLAY_SIZING_FIT(0) } } }) {
                        CLAY_TEXT(cs(s_evt_age[di]),
                            CLAY_TEXT_CONFIG({ .fontId = 2, .fontSize = (uint16_t)(12*sc),
                                               .textColor = MUTED }));
                    }

                    /* topic pill */
                    topic_pill(s_evt_topic[di], sc);

                    /* origin */
                    CLAY_AUTO_ID({ .layout = { .sizing = { CLAY_SIZING_FIXED(120*sc), CLAY_SIZING_FIT(0) } } }) {
                        CLAY_TEXT(cs(s_evt_origin[di]),
                            CLAY_TEXT_CONFIG({ .fontId = 2, .fontSize = (uint16_t)(12*sc),
                                               .textColor = DIM }));
                    }

                    /* payload preview (truncated, clipped) */
                    CLAY_AUTO_ID({
                        .layout = {
                            .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_FIT(0) },
                        },
                        .clip = { .horizontal = true },
                    }) {
                        CLAY_TEXT(cs(s_evt_preview[di]),
                            CLAY_TEXT_CONFIG({ .fontId = 2, .fontSize = (uint16_t)(12*sc),
                                               .textColor = { 0x99, 0x99, 0x99, 255 },
                                               .wrapMode  = CLAY_TEXT_WRAP_NONE }));
                    }
                } /* BusRow */

                /* ── expanded detail panel ────────────────────────── */
                if (expanded) {
                    CLAY_AUTO_ID({
                        .layout = {
                            .sizing          = { CLAY_SIZING_GROW(0), CLAY_SIZING_FIT(0) },
                            .layoutDirection = CLAY_TOP_TO_BOTTOM,
                            .childGap        = (uint16_t)(5 * sc),
                            .padding         = { (uint16_t)(60*sc),  (uint16_t)(16*sc),
                                                 (uint16_t)(10*sc),  (uint16_t)(12*sc) },
                        },
                        .backgroundColor = (Clay_Color){0xf4, 0xf4, 0xff, 255},
                        .border = { .width = { 3, 0, 0, 0, 0 }, .color = ACCENT },
                    }) {
                        detail_kv("origin", s_exp_origin[0] ? s_exp_origin : "\xe2\x80\x94", sc);
                        detail_kv("id",     s_exp_id,   sc);
                        detail_kv("corr",   s_exp_corr, sc);
                        detail_kv("caus",   s_exp_caus, sc);

                        HRULE(sc)

                        /* full payload */
                        CLAY_AUTO_ID({
                            .layout = {
                                .sizing  = { CLAY_SIZING_GROW(0), CLAY_SIZING_FIT(0) },
                                .padding = { 0, 0, (uint16_t)(4*sc), 0 },
                            },
                        }) {
                            CLAY_TEXT(cs(s_exp_full[0] ? s_exp_full : "(empty)"),
                                CLAY_TEXT_CONFIG({ .fontId = 2, .fontSize = (uint16_t)(12*sc),
                                                   .textColor = DIM,
                                                   .wrapMode  = CLAY_TEXT_WRAP_WORDS }));
                        }
                    }
                }

            } /* BusEntry */
        } /* rows */

        } /* else s_disp_count > 0 */

    } /* BusScroll */
}
