/* actor.h — public API
 *
 * Generic actor runtime. Single function entry point.
 * All config from environment variables — runs anywhere.
 *
 * Required env:
 *   ACTOR_ID            unique actor instance id
 *   ACTOR_TOPIC         subscription topic (comma-separated)
 *   ACTOR_RESULT_TOPIC  topic stamped on result tuples
 *   ACTOR_BUS_SUB       NNG pub endpoint    e.g. tcp://bus:5556
 *   ACTOR_BUS_PUB       NNG sub endpoint    e.g. tcp://bus:5557
 *   ACTOR_HANDLER       handler command     e.g. /handlers/agent.sh
 *   ACTOR_LMDB_PATH     LMDB directory path e.g. /var/actor/lmdb
 *
 * Optional env:
 *   ACTOR_TTL_NS        default tuple TTL nanoseconds (0 = no expiry)
 *   ACTOR_HEARTBEAT_MS  heartbeat interval ms         (0 = disabled)
 *   ACTOR_RETRY_MAX     max handler retries           (default 3)
 */

#ifndef ACTOR_H
#define ACTOR_H

/* actor_run reads env, connects, loops. Blocks until SIGTERM/SIGINT.
 * Returns 0 clean shutdown, -1 fatal error. */
int actor_run(void);

#endif /* ACTOR_H */
