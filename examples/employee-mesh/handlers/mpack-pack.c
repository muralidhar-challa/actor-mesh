/* mpack-pack — write a single key=value as mpack to stdout */
#include <stdio.h>
#include <string.h>
#include "../../../vendor/mpack/mpack.h"

int main(int argc, char **argv) {
    if (argc != 3) return 1;
    char out[65536];
    mpack_writer_t w;
    mpack_writer_init(&w, out, sizeof(out));
    mpack_start_map(&w, 1);
    mpack_write_cstr(&w, argv[1]);
    mpack_write_cstr(&w, argv[2]);
    mpack_finish_map(&w);
    fwrite(out, 1, mpack_writer_buffer_used(&w), stdout);
    return 0;
}
