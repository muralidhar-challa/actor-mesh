#include "../state.h"
#include "publish.h"

static void input_field(const char *id, const char *label,
                        const char *display, bool focused,
                        bool multiline, const char *placeholder, float sc) {
    Clay_Color border_col = focused ? ACCENT : BORDER;

    /* single-line: fixed height so it never collapses; multiline: fixed tall box */
    Clay_SizingAxis field_h = multiline
        ? CLAY_SIZING_FIXED(120 * sc)
        : CLAY_SIZING_FIXED(38 * sc);

    CLAY_AUTO_ID({
        .layout = {
            .sizing          = { CLAY_SIZING_GROW(0), CLAY_SIZING_FIT(0) },
            .layoutDirection = CLAY_TOP_TO_BOTTOM,
            .childGap        = (uint16_t)(6 * sc),
        },
    }) {
        CLAY_TEXT(cs(label),
            CLAY_TEXT_CONFIG({ .fontId = 1, .fontSize = (uint16_t)(12*sc),
                               .textColor = focused ? DIM : MUTED }));

        CLAY(CLAY_SID(cs(id)), {
            .layout = {
                .sizing          = { CLAY_SIZING_GROW(0), field_h },
                .childAlignment  = { .y = CLAY_ALIGN_Y_TOP },
                .padding         = { (uint16_t)(12*sc), (uint16_t)(12*sc),
                                     (uint16_t)(9*sc),  (uint16_t)(9*sc) },
            },
            .backgroundColor = (Clay_Color){0xfd, 0xfd, 0xfc, 255},
            .cornerRadius    = CLAY_CORNER_RADIUS(6 * sc),
            .border          = { .width = { 1, 1, 1, 1, 0 }, .color = border_col },
        }) {
            bool has_content = display[0] != '\0';
            const char *txt  = has_content ? display
                             : focused     ? "|"
                             :               placeholder;
            Clay_Color   tcol = has_content ? DIM : MUTED;
            CLAY_TEXT(cs(txt),
                CLAY_TEXT_CONFIG({ .fontId = 2, .fontSize = (uint16_t)(14*sc),
                                   .textColor = tcol,
                                   .wrapMode  = multiline
                                       ? CLAY_TEXT_WRAP_WORDS
                                       : CLAY_TEXT_WRAP_NONE }));
        }
    }
}

void build_publish(float sc) {
    CLAY(CLAY_ID("PublishPane"), {
        .layout = {
            .sizing          = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) },
            .layoutDirection = CLAY_TOP_TO_BOTTOM,
            .childGap        = (uint16_t)(20 * sc),
            .padding         = { (uint16_t)(28*sc), (uint16_t)(28*sc),
                                 (uint16_t)(24*sc), (uint16_t)(24*sc) },
        },
    }) {
        /* heading */
        CLAY_AUTO_ID({
            .layout = {
                .sizing          = { CLAY_SIZING_GROW(0), CLAY_SIZING_FIT(0) },
                .layoutDirection = CLAY_TOP_TO_BOTTOM,
                .childGap        = (uint16_t)(4 * sc),
            },
        }) {
            CLAY_TEXT(cs("Inject tuple"),
                CLAY_TEXT_CONFIG({ .fontId = 0, .fontSize = (uint16_t)(20*sc),
                                   .textColor = DIM }));
            CLAY_TEXT(cs("Publish a raw tuple onto the mesh bus"),
                CLAY_TEXT_CONFIG({ .fontId = 1, .fontSize = (uint16_t)(13*sc),
                                   .textColor = MUTED }));
        }

        HRULE(sc)

        /* topic field */
        input_field("PubTopicField", "TOPIC",
                    s_pub_topic_disp, g_pub_focus == 0,
                    false, "e.g. user_message", sc);

        /* payload field */
        input_field("PubPayloadField", "PAYLOAD",
                    s_pub_payload_disp, g_pub_focus == 1,
                    true, "e.g. {\"query\": \"how many employees?\"}", sc);

        /* send row */
        CLAY_AUTO_ID({
            .layout = {
                .sizing          = { CLAY_SIZING_GROW(0), CLAY_SIZING_FIT(0) },
                .layoutDirection = CLAY_LEFT_TO_RIGHT,
                .childAlignment  = { .y = CLAY_ALIGN_Y_CENTER },
                .childGap        = (uint16_t)(12 * sc),
            },
        }) {
            SPACER()

            bool can_send = g_pub_topic_len > 0 && g_pub_payload_len > 0;
            bool hover    = can_send && Clay_Hovered();

            CLAY(CLAY_ID("PubSendBtn"), {
                .layout = {
                    .sizing         = { CLAY_SIZING_FIT(0), CLAY_SIZING_FIT(0) },
                    .childAlignment = { .x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_CENTER },
                    .padding        = { (uint16_t)(20*sc), (uint16_t)(20*sc),
                                        (uint16_t)(8*sc),  (uint16_t)(8*sc) },
                },
                .backgroundColor = !can_send ? (Clay_Color){0xcc,0xcc,0xcc,255}
                                  : hover     ? ACCENT_H
                                  :             ACCENT,
                .cornerRadius    = CLAY_CORNER_RADIUS(6 * sc),
            }) {
                CLAY_TEXT(cs("Send"),
                    CLAY_TEXT_CONFIG({ .fontId = 1, .fontSize = (uint16_t)(14*sc),
                                       .textColor = !can_send ? (Clay_Color){0x88,0x88,0x88,255}
                                                  : WHITE }));
            }
        }

    } /* PublishPane */
}
