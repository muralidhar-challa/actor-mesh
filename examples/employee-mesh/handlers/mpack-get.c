/* mpack-get — extract a string value from mpack by key */
#include <stdio.h>
#include <string.h>
#include "../../../vendor/mpack/mpack.h"

int main(int argc, char **argv) {
    if (argc < 2) return 1;
    const char *key = argv[1];
    static char g_in[65536], g_val[65536];
    size_t n = fread(g_in, 1, sizeof(g_in), stdin);
    mpack_reader_t r;
    mpack_reader_init_data(&r, g_in, n);
    uint32_t count = mpack_expect_map_max(&r, 16);
    char kbuf[64];
    for (uint32_t i=0; i<count && mpack_reader_error(&r)==mpack_ok; i++) {
        mpack_expect_cstr(&r, kbuf, sizeof(kbuf));
        if (strcmp(kbuf, key) == 0) {
            mpack_expect_cstr(&r, g_val, sizeof(g_val));
            fputs(g_val, stdout); return 0;
        }
        mpack_discard(&r);
    }
    return 1;
}
