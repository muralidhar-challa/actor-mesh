/* mpack-get.c — extract values from mpack as string or JSON
 * Usage: mpack-get [-j] <key>        extract string value for key
 *        mpack-get -j                 dump entire input as JSON  
 *        mpack-get -j <key>           dump value at key as JSON
 * Reads mpack from stdin, writes result to stdout. */
#include <stdio.h>
#include <string.h>
#include "../../../../vendor/mpack/mpack.h"

static char g_in[65536];
static int  json_mode = 0;

/* forward declaration to handle nested structures */
static void mpack_to_json(mpack_reader_t *r);

static void mpack_to_json(mpack_reader_t *r) {
    mpack_tag_t tag = mpack_read_tag(r);
    switch (mpack_tag_type(&tag)) {
    case mpack_type_nil: printf("null"); break;
    case mpack_type_bool: printf(tag.v.b ? "true" : "false"); break;
    case mpack_type_int: printf("%lld", (long long)tag.v.i); break;
    case mpack_type_uint: printf("%llu", (unsigned long long)tag.v.u); break;
    case mpack_type_float: printf("%g", tag.v.f); break;
    case mpack_type_double: printf("%g", tag.v.d); break;
    case mpack_type_str: {
        uint32_t len = tag.v.l;
        char buf[65536];
        if (len >= sizeof(buf)) len = sizeof(buf)-1;
        mpack_read_bytes(r, buf, len); buf[len] = 0;
        mpack_done_str(r);
        printf("\""); for (uint32_t i=0;i<len;i++) {char c=buf[i];
            if (c=='"'||c=='\\') printf("\\%c",c);
            else if (c=='\n') printf("\\n"); else if (c=='\r') printf("\\r");
            else if (c>=32&&c<127) putchar(c); else printf("\\u%04x",(unsigned char)c);}
        printf("\""); break; }
    case mpack_type_array: {
        uint32_t count = tag.v.n; printf("[");
        for (uint32_t i=0;i<count;i++) { if(i>0)printf(","); mpack_to_json(r); }
        mpack_done_array(r); printf("]"); break; }
    case mpack_type_map: {
        uint32_t count = tag.v.n; printf("{"); int first=1;
        for (uint32_t i=0;i<count;i++) {
            mpack_to_json(r); printf(":"); mpack_to_json(r);
            if (!first) printf(","); first=0; }
        mpack_done_map(r); printf("}"); break; }
    default: printf("null"); mpack_discard(r); break;
    }
}

static void extract_value(mpack_reader_t *r, const char *key) {
    uint32_t count = mpack_expect_map_max(r, 65536);
    for (uint32_t i=0;i<count && mpack_reader_error(r)==mpack_ok; i++) {
        char kbuf[64];
        mpack_expect_cstr(r, kbuf, sizeof(kbuf));
        if (strcmp(kbuf, key) == 0) {
            if (json_mode) { mpack_to_json(r); }
            else { char vbuf[65536]; mpack_expect_cstr(r, vbuf, sizeof(vbuf)); fputs(vbuf, stdout); }
            return;
        }
        mpack_discard(r);
    }
}

int main(int argc, char **argv) {
    const char *key = NULL;
    for (int i=1;i<argc;i++) {
        if (strcmp(argv[i], "-j")==0) json_mode=1;
        else key=argv[i];
    }

    size_t n = fread(g_in, 1, sizeof(g_in)-1, stdin);
    g_in[n] = 0;

    mpack_reader_t r;
    mpack_reader_init_data(&r, g_in, n);

    if (key) { extract_value(&r, key); }
    else if (json_mode) { mpack_to_json(&r); }
    return 0;
}
