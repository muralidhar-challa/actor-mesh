/* actor_isolation.h — optional process confinement, applied at startup
 *
 * Phase 1: privilege drop and resource limits.
 *
 * Every knob is opt-in through the environment. An actor with none of these
 * set behaves exactly as it did before this file existed -- that is the
 * property the tests in tests/test-isolation.c defend, and it is what lets an
 * existing deployment upgrade without noticing.
 *
 * The rule for everything here is fail closed: a variable that is set and
 * cannot be honoured aborts startup. An actor must never run less confined
 * than it believes it is, because a deployment that configures confinement,
 * sees no error, and gets none is worse off than one that never tried.
 *
 *   ACTOR_UID / ACTOR_GID      drop to this uid/gid, irreversibly
 *   ACTOR_RLIMIT_AS            address space, bytes
 *   ACTOR_RLIMIT_CPU           cpu time, seconds
 *   ACTOR_RLIMIT_NOFILE        open file descriptors
 *   ACTOR_RLIMIT_NPROC         processes/threads for the real uid
 *
 * Returns 0 if everything requested was applied (including the case where
 * nothing was requested), -1 if anything requested could not be.
 */
#ifndef ACTOR_ISOLATION_H
#define ACTOR_ISOLATION_H

/* Apply the actor-lifetime isolation named in the environment.
 *
 * MUST be called before any pthread_create. Two reasons, and the function
 * enforces the first by refusing to run if it can tell threads already exist:
 *
 *   - setresuid/setresgid are per-thread at the syscall level; glibc's
 *     wrappers broadcast to the other threads by signal. Running before any
 *     thread exists avoids depending on that machinery at all.
 *   - the namespace work in later phases has a hard kernel restriction of the
 *     same shape (unshare(CLONE_NEWUSER) fails EINVAL in a multithreaded
 *     process), so the constraint is worth holding from the start rather than
 *     discovering later.
 */
int actor_isolation_apply(void);

#endif /* ACTOR_ISOLATION_H */
