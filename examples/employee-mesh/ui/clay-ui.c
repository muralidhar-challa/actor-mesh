#define _POSIX_C_SOURCE 200809L
#define CLAY_IMPLEMENTATION
#include "../../../vendor/clay.h"
#include "/tmp/clay/renderers/SDL2/clay_renderer_SDL2.c"
#include <nng/nng.h>
#include <nng/protocol/pubsub0/pub.h>
#include <nng/protocol/pubsub0/sub.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_MSG    512
#define MAX_ACTORS 16
#define MAX_INPUT  2048

typedef struct { char id[32]; int inbox, outbox, alive; } Actor;
static Actor ga[MAX_ACTORS]; static int gn;
typedef struct { char *text; int user; } Msg;
static Msg  gm[MAX_MSG]; static int gmn, gth, gtot;
static char gi[MAX_INPUT]; static int gil;
static int  gtab, gcon;
static nng_socket gp, gs;
static uint8_t gc[16];

static void U7(uint8_t d[16]){struct timespec ts;clock_gettime(CLOCK_REALTIME,&ts);uint64_t ms=(uint64_t)ts.tv_sec*1000+ts.tv_nsec/1000000;for(int i=0;i<6;i++)d[i]=(ms>>(40-i*8))&0xFF;for(int i=6;i<16;i++)d[i]=(uint8_t)(rand()%256);d[6]=(d[6]&0x0f)|0x70;d[8]=(d[8]&0x3f)|0x80;}
static void ni(void){nng_pub0_open(&gp);nng_sub0_open(&gs);const char*bs=getenv("ACTOR_BUS_SUB")?:"tcp://127.0.0.1:5556";const char*bp=getenv("ACTOR_BUS_PUB")?:"tcp://127.0.0.1:5557";nng_dial(gp,bp,NULL,0);nng_dial(gs,bs,NULL,0);nng_socket_set(gs,"sub:subscribe","",0);nng_socket_set_ms(gs,"recv-timeout",10);U7(gc);gcon=1;}
static void np(void){for(;;){nng_msg*m=NULL;if(nng_recvmsg(gs,&m,0))break;void*b=nng_msg_body(m);size_t l=nng_msg_len(m);if(l>=80){char t[33]={0};memcpy(t,b,32);if(strncmp(t,"heartbeat",9)==0){const char*pd=(char*)b+256,*ip=strstr(pd,"\"id\":\"");if(ip){ip+=6;char id[32]={0};int ai=0;while(*ip&&*ip!='"'&&ai<31)id[ai++]=*ip++;int f=0;for(int i=0;i<gn;i++)if(!strcmp(ga[i].id,id)){ga[i].alive=1;char*oo;if((oo=strstr(pd,"\"inbox\":")))sscanf(oo+8,"%d",&ga[i].inbox);if((oo=strstr(pd,"\"outbox\":")))sscanf(oo+9,"%d",&ga[i].outbox);f=1;break;}if(!f&&gn<MAX_ACTORS){strcpy(ga[gn].id,id);ga[gn].alive=1;gn++;}}}if(strncmp(t,"agent_response",14)==0&&l>270&&gth&&gmn<MAX_MSG){const char*pd=(char*)b+256,*ans=strstr(pd,"answer");if(ans){ans+=6;if((unsigned char)*ans>=0xa0&&(unsigned char)*ans<=0xbf){int sl=(unsigned char)*ans-0xa0;char*a=malloc(sl+1);memcpy(a,ans+1,sl);a[sl]=0;gm[gmn].text=a;gm[gmn].user=0;gmn++;gth=0;gtot++;}}}}nng_msg_free(m);}for(int i=0;i<gn;i++)ga[i].alive=0;}
static void ms(const char*topic,const uint8_t*pl,size_t len){uint8_t f[256+4096];memset(f,0,256);int tl=strlen(topic);if(tl>31)tl=31;memcpy(f,topic,tl);U7(f+32);memcpy(f+48,gc,16);memcpy(f+80,"desktop",7);struct timespec ts;clock_gettime(CLOCK_REALTIME,&ts);int64_t ns=ts.tv_sec*1000000000LL+ts.tv_nsec;memcpy(f+112,&ns,8);uint32_t pl2=len;memcpy(f+138,&pl2,4);memcpy(f+256,pl,len);nng_send(gp,f,256+len,0);}
static void cs(const char*q){uint8_t b[4096];int o=0;b[o++]=0x82;b[o++]=0xa4;memcpy(b+o,"type",4);o+=4;b[o++]=0xac;memcpy(b+o,"user_message",12);o+=12;b[o++]=0xa5;memcpy(b+o,"query",5);o+=5;int ql=strlen(q);if(ql<32){b[o++]=0xa0+ql;}else{b[o++]=0xd9;b[o++]=ql;}memcpy(b+o,q,ql);o+=ql;ms("user_message",b,o);}

static Clay_String S(const char*s){return(Clay_String){.length=(int32_t)strlen(s),.chars=s};}
static char*K(int n){static char b[16];if(n<1024)snprintf(b,16,"%d B",n);else snprintf(b,16,"%.1fK",n/1024.0);return b;}

int main(void){
    SDL_Init(SDL_INIT_VIDEO);TTF_Init();
    SDL_Window *w=SDL_CreateWindow("Actor Mesh",SDL_WINDOWPOS_CENTERED,SDL_WINDOWPOS_CENTERED,1280,800,SDL_WINDOW_RESIZABLE);
    SDL_Renderer *r=SDL_CreateRenderer(w,-1,SDL_RENDERER_ACCELERATED);

    SDL2_Font fonts[2];
    fonts[0].fontId=0;fonts[0].font=TTF_OpenFont("fonts/IBMPlexSans.ttf",64);
    fonts[1].fontId=1;fonts[1].font=TTF_OpenFont("fonts/IBMPlexSans.ttf",32);

    Clay_Initialize(Clay_CreateArenaWithCapacityAndMemory(Clay_MinMemorySize(),calloc(1,Clay_MinMemorySize())),(Clay_Dimensions){1280,800},(Clay_ErrorHandler){0});
    Clay_SetMeasureTextFunction(SDL2_MeasureText,fonts);
    ni();

    while(1){
        SDL_Event ev;while(SDL_PollEvent(&ev)){
            if(ev.type==SDL_QUIT)goto done;
            if(ev.type==SDL_TEXTINPUT){for(char*c=ev.text.text;*c&&gil<MAX_INPUT-1;c++){gi[gil++]=*c;gi[gil]=0;}}
            if(ev.type==SDL_KEYDOWN){
                if(ev.key.keysym.sym==SDLK_BACKSPACE&&gil>0){gil--;gi[gil]=0;}
                if((ev.key.keysym.sym==SDLK_RETURN||ev.key.keysym.sym==SDLK_KP_ENTER)&&gil>0&&!gth){
                    if(gmn<MAX_MSG){gm[gmn].text=strdup(gi);gm[gmn].user=1;gmn++;gtot++;}cs(gi);gil=0;gi[0]=0;gth=1;}
                if(ev.key.keysym.sym==SDLK_TAB)gtab=!gtab;
            }
            if(ev.type==SDL_MOUSEBUTTONDOWN){int mx=ev.button.x,my=ev.button.y;int ww;SDL_GetWindowSize(w,&ww,NULL);
                if(my<44&&mx>ww-180&&mx<ww-120)gtab=0;if(my<44&&mx>ww-115&&mx<ww-20)gtab=1;}
        }
        np();
        int ww,wh;SDL_GetWindowSize(w,&ww,&wh);
        float s=wh/1080.0f;if(s<1.0f)s=1.0f;

        Clay_SetLayoutDimensions((Clay_Dimensions){(float)ww,(float)wh});
        Clay_BeginLayout();

        Clay_Color bg={0xf9,0xf9,0xf8,255},drk={0x0a,0x0a,0x0a,255},red={0xc4,0x1e,0x3a,255};
        Clay_Color dim={0x2a,0x2a,0x2a,255},mu={0x88,0x88,0x88,255},wht={255,255,255,255},grn={0x22,0xc5,0x5e,255};

        CLAY(CLAY_ID("R"),{.layout={.sizing={CLAY_SIZING_GROW(0),CLAY_SIZING_GROW(0)},.layoutDirection=CLAY_TOP_TO_BOTTOM},.backgroundColor=bg}){
            CLAY(CLAY_ID("H"),{.layout={.sizing={CLAY_SIZING_GROW(0),CLAY_SIZING_FIT(0)},.layoutDirection=CLAY_LEFT_TO_RIGHT,.childGap=16*s,.padding={10*s,20*s,10*s,20*s}},.backgroundColor=drk}){
                CLAY_TEXT(S("Actor Mesh"),CLAY_TEXT_CONFIG({.fontId=0,.fontSize=22*s,.textColor=wht}));
                CLAY_AUTO_ID({.layout={.sizing={CLAY_SIZING_GROW(0)}}}){}
                int al=0;for(int i=0;i<gn;i++)if(ga[i].alive)al++;
                char st[64];snprintf(st,64,"%s %d actors",gcon?"●":"○",al);
                CLAY_TEXT(S(st),CLAY_TEXT_CONFIG({.fontId=1,.fontSize=15*s,.textColor=gcon?grn:mu}));
                CLAY_AUTO_ID({.layout={.sizing={CLAY_SIZING_GROW(0)}}}){}
                CLAY(CLAY_ID("TC"),{.layout={.padding={6*s,14*s,6*s,14*s}},.backgroundColor=gtab==0?red:(Clay_Color){0,0,0,0},.cornerRadius={4*s,4*s,4*s,4*s}}){
                    CLAY_TEXT(S("Chat"),CLAY_TEXT_CONFIG({.fontId=1,.fontSize=16*s,.textColor=gtab==0?wht:mu}));}
                CLAY(CLAY_ID("TM"),{.layout={.padding={6*s,14*s,6*s,14*s}},.backgroundColor=gtab==1?red:(Clay_Color){0,0,0,0},.cornerRadius={4*s,4*s,4*s,4*s}}){
                    CLAY_TEXT(S("Mesh"),CLAY_TEXT_CONFIG({.fontId=1,.fontSize=16*s,.textColor=gtab==1?wht:mu}));}
            }
            if(gtab==0){
                CLAY(CLAY_ID("Msg"),{.layout={.sizing={CLAY_SIZING_GROW(0),CLAY_SIZING_GROW(0)},.layoutDirection=CLAY_TOP_TO_BOTTOM,.childGap=8*s,.padding={20*s,24*s,20*s,24*s}}}){
                    for(int i=0;i<gmn;i++)if(gm[i].user){CLAY(CLAY_IDI("U",i),{.layout={.layoutDirection=CLAY_LEFT_TO_RIGHT,.sizing={CLAY_SIZING_GROW(0),CLAY_SIZING_FIT(0)},.childAlignment={.x=CLAY_ALIGN_X_RIGHT}}}){CLAY(CLAY_IDI("B",i),{.layout={.sizing={CLAY_SIZING_FIT(0),CLAY_SIZING_FIT(0)},.padding={10*s,18*s,10*s,18*s}},.backgroundColor=drk,.cornerRadius={12*s,12*s,4*s,12*s}}){CLAY_TEXT(S(gm[i].text),CLAY_TEXT_CONFIG({.fontId=1,.fontSize=17*s,.textColor=wht}));}}}
                    else{CLAY(CLAY_IDI("A",i),{.layout={.padding={6*s,12*s,6*s,12*s}}}){CLAY_TEXT(S(gm[i].text),CLAY_TEXT_CONFIG({.fontId=1,.fontSize=17*s,.textColor=dim}));}}
                    if(gth)CLAY_TEXT(S("thinking..."),CLAY_TEXT_CONFIG({.fontId=1,.fontSize=15*s,.textColor=mu}));
                }
                CLAY(CLAY_ID("In"),{.layout={.sizing={CLAY_SIZING_GROW(0),CLAY_SIZING_FIT(0)},.layoutDirection=CLAY_LEFT_TO_RIGHT,.padding={14*s,20*s,14*s,20*s}},.backgroundColor=wh,.border={.width=CLAY_BORDER_OUTSIDE(1*s),.color={0xe2,0xe2,0xe2,255}}}){
                    CLAY(CLAY_ID("F"),{.layout={.sizing={CLAY_SIZING_GROW(0),CLAY_SIZING_FIT(0)},.padding={10*s,18*s,10*s,18*s}},.backgroundColor=drk,.cornerRadius={12*s,12*s,12*s,12*s}}){
                        CLAY_TEXT(S(gi[0]?gi:"Ask about employees..."),CLAY_TEXT_CONFIG({.fontId=1,.fontSize=17*s,.textColor=gi[0]?wht:mu}));}
                }
            }else{
                CLAY(CLAY_ID("T"),{.layout={.sizing={CLAY_SIZING_GROW(0),CLAY_SIZING_GROW(0)},.layoutDirection=CLAY_TOP_TO_BOTTOM,.padding={24*s,24*s,24*s,24*s},.childGap=16*s}}){
                    int al=0;for(int i=0;i<gn;i++)if(ga[i].alive)al++;
                    CLAY(CLAY_ID("St"),{.layout={.sizing={CLAY_SIZING_GROW(0),CLAY_SIZING_FIT(0)},.layoutDirection=CLAY_LEFT_TO_RIGHT,.childGap=1}}){
                        for(int z=0;z<3;z++){char v[16],l[16];if(z==0){snprintf(v,16,"%d",al);snprintf(l,16,"ACTORS");}else if(z==1){snprintf(v,16,"%d",gtot);snprintf(l,16,"MSGS");}else{snprintf(v,16,"%s",gcon?"●":"○");snprintf(l,16,"CONNECTED");}
                            CLAY(CLAY_IDI("S",z),{.layout={.sizing={CLAY_SIZING_FIXED(160*s),CLAY_SIZING_FIT(0)},.layoutDirection=CLAY_TOP_TO_BOTTOM,.padding={20*s,20*s,20*s,20*s}},.backgroundColor=wht}){CLAY_TEXT(S(v),CLAY_TEXT_CONFIG({.fontId=0,.fontSize=34*s,.textColor=red}));CLAY_TEXT(S(l),CLAY_TEXT_CONFIG({.fontId=1,.fontSize=14*s,.textColor=mu}));}}}
                    CLAY(CLAY_ID("H2"),{.layout={.sizing={CLAY_SIZING_GROW(0),CLAY_SIZING_FIT(0)},.layoutDirection=CLAY_LEFT_TO_RIGHT,.padding={8*s,20*s,8*s,20*s}},.backgroundColor=drk}){CLAY_TEXT(S("Actor"),CLAY_TEXT_CONFIG({.fontId=1,.fontSize=15*s,.textColor=wht}));CLAY_AUTO_ID({.layout={.sizing={CLAY_SIZING_GROW(0)}}}){}CLAY_TEXT(S("In / Out"),CLAY_TEXT_CONFIG({.fontId=1,.fontSize=15*s,.textColor=wht}));CLAY_AUTO_ID({.layout={.sizing={CLAY_SIZING_GROW(0)}}}){}CLAY_TEXT(S("Status"),CLAY_TEXT_CONFIG({.fontId=1,.fontSize=15*s,.textColor=wht}));}
                    for(int i=0;i<gn;i++){Clay_Color rb=i%2?bg:wht,tc=ga[i].alive?dim:mu;
                        CLAY(CLAY_IDI("R",i),{.layout={.sizing={CLAY_SIZING_GROW(0),CLAY_SIZING_FIT(0)},.layoutDirection=CLAY_LEFT_TO_RIGHT,.childGap=12*s,.padding={6*s,20*s,6*s,20*s}},.backgroundColor=rb}){
                            char bf[64];snprintf(bf,64,"%s %s",ga[i].alive?"●":"○",ga[i].id);CLAY_TEXT(S(bf),CLAY_TEXT_CONFIG({.fontId=1,.fontSize=15*s,.textColor=tc}));
                            CLAY_AUTO_ID({.layout={.sizing={CLAY_SIZING_GROW(0)}}}){}char io[64];snprintf(io,64,"%s / %s",K(ga[i].inbox),K(ga[i].outbox));CLAY_TEXT(S(io),CLAY_TEXT_CONFIG({.fontId=1,.fontSize=15*s,.textColor=tc}));
                            CLAY_AUTO_ID({.layout={.sizing={CLAY_SIZING_GROW(0)}}}){}CLAY_TEXT(S(ga[i].alive?"alive":"offline"),CLAY_TEXT_CONFIG({.fontId=1,.fontSize=15*s,.textColor=tc}));}}
                }
            }
        }
        Clay_RenderCommandArray cmds=Clay_EndLayout(0);
        SDL_SetRenderDrawColor(r,0xf9,0xf9,0xf8,255);SDL_RenderClear(r);
        Clay_SDL2_Render(r,cmds,fonts);
        SDL_RenderPresent(r);
    }
done:
    SDL_DestroyRenderer(r);SDL_DestroyWindow(w);TTF_Quit();SDL_Quit();return 0;
}
