/* echo.c — simplest possible handler: stdin → stdout
 *
 * Used for testing the NNG mesh end-to-end.
 * Reads ACTOR_TUPLE_ID etc from env, writes them back as JSON + payload.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_PAYLOAD 65536  /* 64 KiB */

int main(void) {
#define S(v) ((v) ? (v) : "none")
#define S0(v) ((v) ? (v) : "0")
    const char *tuple_id       = S(getenv("ACTOR_TUPLE_ID"));
    const char *correlation_id = S(getenv("ACTOR_CORRELATION_ID"));
    const char *causation_id   = S(getenv("ACTOR_CAUSATION_ID"));
    const char *origin         = S(getenv("ACTOR_TUPLE_ORIGIN"));
    const char *attempt        = S0(getenv("ACTOR_ATTEMPT"));

    /* Read stdin fully */
    char payload[MAX_PAYLOAD];
    size_t len = fread(payload, 1, sizeof(payload) - 1, stdin);
    payload[len] = '\0';

    /* Write result to stdout */
    fprintf(stdout,
        "{\n"
        "  \"echo\": {\n"
        "    \"tuple_id\": \"%s\",\n"
        "    \"correlation_id\": \"%s\",\n"
        "    \"causation_id\": \"%s\",\n"
        "    \"origin\": \"%s\",\n"
        "    \"attempt\": %s,\n"
        "    \"payload\": \"%.*s\"\n"
        "  }\n"
        "}\n",
        tuple_id, correlation_id, causation_id,
        origin, attempt,
        (int)len, payload
    );

    return 0;
}
