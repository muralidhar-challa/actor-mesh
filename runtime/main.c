/* main.c — actor binary entrypoint
 *
 * Whisper thin. Just calls actor_run().
 * All config from environment.
 */

#include "actor.h"
#include <stdlib.h>

int main(void) {
    return actor_run() == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
