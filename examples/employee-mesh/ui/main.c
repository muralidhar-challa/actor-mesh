#define CLAY_IMPLEMENTATION
#include "../../../vendor/clay.h"
#include "/tmp/clay/renderers/SDL2/clay_renderer_SDL2.c"

#include "state.h"
#include "net.h"
#include "components/header.h"
#include "components/actors.h"
#include "components/bus.h"
#include "components/publish.h"

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <stdlib.h>

/* ── bus autoscroll ──────────────────────────────────────────────── */

static int s_prev_nevents = 0;

static void bus_autoscroll(void) {
    if (g_nevents == s_prev_nevents) return;
    s_prev_nevents = g_nevents;

    /* scroll the bus list back to the top (newest) */
    Clay_ElementId id = Clay_GetElementId(cs("BusScroll"));
    Clay_ScrollContainerData d = Clay_GetScrollContainerData(id);
    if (d.found)
        *d.scrollPosition = (Clay_Vector2){ 0, 0 };
}

/* ── main ────────────────────────────────────────────────────────── */

int main(void) {
    SDL_Init(SDL_INIT_VIDEO);
    TTF_Init();

    SDL_Window *window = SDL_CreateWindow(
        "Actor Mesh",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        1280, 800,
        SDL_WINDOW_RESIZABLE);
    SDL_Renderer *renderer = SDL_CreateRenderer(
        window, -1, SDL_RENDERER_ACCELERATED);

    SDL2_Font fonts[3];
    fonts[0].fontId = 0; fonts[0].font = TTF_OpenFont("fonts/NotoSans.ttf",     64);
    fonts[1].fontId = 1; fonts[1].font = TTF_OpenFont("fonts/NotoSans.ttf",     32);
    fonts[2].fontId = 2; fonts[2].font = TTF_OpenFont("fonts/NotoSansMono.ttf", 28);

    if (!fonts[0].font || !fonts[1].font || !fonts[2].font) {
        fprintf(stderr, "[ui] failed to load fonts — run from ui/ directory\n");
        return 1;
    }

    uint64_t clay_mem = Clay_MinMemorySize();
    Clay_Initialize(
        Clay_CreateArenaWithCapacityAndMemory(clay_mem, calloc(1, clay_mem)),
        (Clay_Dimensions){ 1280, 800 },
        (Clay_ErrorHandler){ 0 }
    );
    Clay_SetMeasureTextFunction(SDL2_MeasureText, fonts);

    net_init();

    uint32_t last_ticks = SDL_GetTicks();

    while (1) {
        uint32_t now_ticks = SDL_GetTicks();
        float    dt        = (now_ticks - last_ticks) / 1000.0f;
        last_ticks = now_ticks;
        g_tick++;

        /* ── events ───────────────────────────────────────────────── */
        bool  mouse_clicked = false;
        float wheel_y       = 0;

        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_QUIT) goto done;

            if (ev.type == SDL_TEXTINPUT && g_tab == 2) {
                char *dst  = (g_pub_focus == 0) ? g_pub_topic   : g_pub_payload;
                int  *dlen = (g_pub_focus == 0) ? &g_pub_topic_len : &g_pub_payload_len;
                int   dmax = (g_pub_focus == 0) ? (int)sizeof(g_pub_topic) - 1
                                                : UI_MAX_INPUT - 1;
                for (char *c = ev.text.text; *c && *dlen < dmax; c++) {
                    dst[(*dlen)++] = *c;
                    dst[*dlen]     = '\0';
                }
            }

            if (ev.type == SDL_KEYDOWN) {
                switch (ev.key.keysym.sym) {
                case SDLK_BACKSPACE:
                    if (g_tab == 2) {
                        if (g_pub_focus == 0 && g_pub_topic_len > 0)
                            g_pub_topic[--g_pub_topic_len] = '\0';
                        else if (g_pub_focus == 1 && g_pub_payload_len > 0)
                            g_pub_payload[--g_pub_payload_len] = '\0';
                    }
                    break;
                case SDLK_RETURN:
                case SDLK_KP_ENTER:
                    if (g_tab == 2
                            && g_pub_topic_len > 0
                            && g_pub_payload_len > 0) {
                        net_send_frame(g_pub_topic,
                                       (uint8_t *)g_pub_payload,
                                       (size_t)g_pub_payload_len);
                        g_pub_payload[0]  = '\0';
                        g_pub_payload_len = 0;
                    }
                    break;
                case SDLK_TAB:
                    if (g_tab == 2)
                        g_pub_focus = !g_pub_focus;
                    else
                        g_tab = (g_tab + 1) % 3;
                    break;
                case SDLK_ESCAPE:
                    if (g_tab == 1) g_expanded = -1;
                    break;
                }
            }

            if (ev.type == SDL_MOUSEBUTTONDOWN && ev.button.button == SDL_BUTTON_LEFT)
                mouse_clicked = true;

            if (ev.type == SDL_MOUSEWHEEL)
                wheel_y = (float)ev.wheel.y * 48.0f;
        }

        /* pointer state */
        int mx, my;
        SDL_GetMouseState(&mx, &my);
        Clay_SetPointerState((Clay_Vector2){ (float)mx, (float)my }, false);
        Clay_UpdateScrollContainers(false, (Clay_Vector2){ 0, wheel_y }, dt);

        net_poll();

        /* ── layout ───────────────────────────────────────────────── */
        int win_w, win_h;
        SDL_GetWindowSize(window, &win_w, &win_h);

        float sc = (float)win_h / 1080.0f;
        if (sc < 1.0f) sc = 1.0f;

        prepare_strings();

        Clay_SetLayoutDimensions((Clay_Dimensions){ (float)win_w, (float)win_h });
        Clay_BeginLayout();

        CLAY(CLAY_ID("Root"), {
            .layout = {
                .sizing          = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) },
                .layoutDirection = CLAY_TOP_TO_BOTTOM,
            },
            .backgroundColor = BG,
        }) {
            build_header(sc);

            if (g_tab == 0) build_actors(sc);
            if (g_tab == 1) build_bus(sc);
            if (g_tab == 2) build_publish(sc);
        }

        Clay_RenderCommandArray cmds = Clay_EndLayout(0);

        /* ── click handling ───────────────────────────────────────── */
        if (mouse_clicked) {
            /* tab switching */
            for (int t = 0; t < 3; t++) {
                if (Clay_PointerOver(Clay_GetElementIdWithIndex(cs("Tab"), (uint32_t)t)))
                    g_tab = t;
            }

            /* bus row expand/collapse */
            if (g_tab == 1) {
                for (int di = 0; di < s_disp_count; di++) {
                    if (Clay_PointerOver(
                            Clay_GetElementIdWithIndex(cs("BusRow"), (uint32_t)di))) {
                        g_expanded = (g_expanded == di) ? -1 : di;
                        break;
                    }
                }
            }

            /* publish field focus */
            if (g_tab == 2) {
                if (Clay_PointerOver(Clay_GetElementId(cs("PubTopicField"))))
                    g_pub_focus = 0;
                if (Clay_PointerOver(Clay_GetElementId(cs("PubPayloadField"))))
                    g_pub_focus = 1;
                if (Clay_PointerOver(Clay_GetElementId(cs("PubSendBtn")))
                        && g_pub_topic_len > 0 && g_pub_payload_len > 0) {
                    net_send_frame(g_pub_topic,
                                   (uint8_t *)g_pub_payload,
                                   (size_t)g_pub_payload_len);
                    g_pub_payload[0]  = '\0';
                    g_pub_payload_len = 0;
                }
            }
        }

        /* bus autoscroll to top on new events */
        if (g_tab == 1) bus_autoscroll();

        /* ── render ───────────────────────────────────────────────── */
        SDL_SetRenderDrawColor(renderer, 0xf9, 0xf9, 0xf8, 255);
        SDL_RenderClear(renderer);
        Clay_SDL2_Render(renderer, cmds, fonts);
        SDL_RenderPresent(renderer);

        SDL_Delay(16);
    }

done:
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    TTF_Quit();
    SDL_Quit();
    return 0;
}
