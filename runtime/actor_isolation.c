#define _GNU_SOURCE
/* actor_isolation.c — Phase 1: privilege drop and resource limits.
 *
 * See actor_isolation.h for the contract. The short version: every knob is
 * opt-in, an unset variable is never an error, and a set variable that cannot
 * be applied always is.
 */

#include "actor_isolation.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <limits.h>
#include <unistd.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <grp.h>

/* ── Environment parsing ─────────────────────────────────────────────────── */

/* Parse a non-negative integer. Returns 0 on success, -1 if the value is not a
 * clean number -- a typo in a security-relevant variable must not silently
 * become 0, which is uid 0. */
static int parse_ull(const char* s, unsigned long long* out) {
    if (!s || !*s) return -1;
    errno = 0;
    char* end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno || !end || *end != '\0') return -1;
    *out = v;
    return 0;
}

/* ── Resource limits ─────────────────────────────────────────────────────── */

static int apply_rlimit(const char* env_name, int resource, const char* label) {
    const char* v = getenv(env_name);
    if (!v) return 0;                      /* unset is not a failure */

    unsigned long long n;
    if (parse_ull(v, &n)) {
        fprintf(stderr, "[actor] isolation: %s=\"%s\" is not a number\n", env_name, v);
        return -1;
    }

    struct rlimit rl = { (rlim_t)n, (rlim_t)n };
    if (setrlimit(resource, &rl) != 0) {
        fprintf(stderr, "[actor] isolation: setrlimit(%s, %llu): %s\n",
                label, n, strerror(errno));
        return -1;
    }

    /* Read back. setrlimit can succeed while clamping to a lower hard limit
       the process cannot raise, and a limit quietly weaker than the one asked
       for is exactly the silent degradation this module exists to prevent. */
    struct rlimit got;
    if (getrlimit(resource, &got) != 0) {
        fprintf(stderr, "[actor] isolation: getrlimit(%s): %s\n", label, strerror(errno));
        return -1;
    }
    if (got.rlim_cur != (rlim_t)n) {
        fprintf(stderr, "[actor] isolation: %s applied as %llu, not the requested %llu\n",
                label, (unsigned long long)got.rlim_cur, n);
        return -1;
    }
    return 0;
}

/* ── Privilege drop ──────────────────────────────────────────────────────── */

/* Drop to ACTOR_GID then ACTOR_UID.
 *
 * Order matters: after the uid drop the process is generally no longer
 * privileged enough to change its gid, so the gid must go first.
 *
 * setresgid/setresuid rather than POSIX setgid/setuid deliberately. They set
 * real, effective and saved-set ids explicitly, so an incomplete drop is an
 * error. POSIX setuid drops all three only when the caller is privileged --
 * called without privilege it moves the effective uid alone, leaves the
 * saved-set id intact so privilege remains regainable, and still reports
 * success. That silent partial drop is the failure this module exists to
 * prevent. Both calls are outside POSIX but present on Linux, which is the
 * platform this file is compiled for.
 */
static int drop_privilege(void) {
    const char* uid_s = getenv("ACTOR_UID");
    const char* gid_s = getenv("ACTOR_GID");
    if (!uid_s && !gid_s) return 0;

    unsigned long long uid_v = 0, gid_v = 0;
    if (uid_s && parse_ull(uid_s, &uid_v)) {
        fprintf(stderr, "[actor] isolation: ACTOR_UID=\"%s\" is not a number\n", uid_s);
        return -1;
    }
    if (gid_s && parse_ull(gid_s, &gid_v)) {
        fprintf(stderr, "[actor] isolation: ACTOR_GID=\"%s\" is not a number\n", gid_s);
        return -1;
    }

    if (gid_s) {
        /* Drop supplementary groups first. They survive setresgid, and a
           process still carrying the launching user's groups has not really
           dropped its group privilege. Failure here is only tolerated when
           there is nothing to drop -- an unprivileged actor with no
           supplementary groups beyond its own is already where it needs to
           be. */
        if (setgroups(0, NULL) != 0 && errno != EPERM) {
            fprintf(stderr, "[actor] isolation: setgroups(0): %s\n", strerror(errno));
            return -1;
        }
        if (setresgid((gid_t)gid_v, (gid_t)gid_v, (gid_t)gid_v) != 0) {
            fprintf(stderr, "[actor] isolation: setresgid(%llu): %s\n",
                    gid_v, strerror(errno));
            return -1;
        }
    }

    if (uid_s) {
        if (setresuid((uid_t)uid_v, (uid_t)uid_v, (uid_t)uid_v) != 0) {
            fprintf(stderr, "[actor] isolation: setresuid(%llu): %s\n",
                    uid_v, strerror(errno));
            return -1;
        }
    }

    /* Verify the postcondition rather than trust the return. What matters is
       the state the process is actually in, and a drop that half-happened must
       not reach the poll loop. getresuid/getresgid would read the saved-set id
       directly but are themselves non-standard; comparing real and effective
       ids catches every partial drop these two calls can produce. */
    if (gid_s) {
        gid_t rg = getgid(), eg = getegid();
        if (rg != (gid_t)gid_v || eg != (gid_t)gid_v) {
            fprintf(stderr, "[actor] isolation: gid drop incomplete (real=%u effective=%u "
                            "wanted=%llu)\n", (unsigned)rg, (unsigned)eg, gid_v);
            return -1;
        }
    }
    if (uid_s) {
        uid_t ru = getuid(), eu = geteuid();
        if (ru != (uid_t)uid_v || eu != (uid_t)uid_v) {
            fprintf(stderr, "[actor] isolation: uid drop incomplete (real=%u effective=%u "
                            "wanted=%llu)\n", (unsigned)ru, (unsigned)eu, uid_v);
            return -1;
        }
    }
    return 0;
}

/* ── Entry point ─────────────────────────────────────────────────────────── */

/* True if this process appears to have more than one thread.
 *
 * /proc/self/status carries a labelled "Threads:" line, which avoids parsing
 * /proc/self/stat positionally past a comm field that may itself contain
 * spaces and parentheses. If /proc is not mounted we cannot tell, and this
 * returns 0: the check guards against a future reordering of actor_run, not
 * against an attacker, so being unable to perform it must not block an actor
 * that is in fact single-threaded. */
static int looks_multithreaded(void) {
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return 0;

    char line[256];
    int threads = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "Threads:", 8) == 0) {
            threads = atoi(line + 8);
            break;
        }
    }
    fclose(f);
    return threads > 1;
}

int actor_isolation_apply(void) {
    /* Nothing requested: return before doing anything at all, so an actor that
       does not opt in pays not even the /proc read. */
    int requested =
        getenv("ACTOR_UID")           || getenv("ACTOR_GID")         ||
        getenv("ACTOR_RLIMIT_AS")     || getenv("ACTOR_RLIMIT_CPU")  ||
        getenv("ACTOR_RLIMIT_NOFILE") || getenv("ACTOR_RLIMIT_NPROC");
    if (!requested) return 0;

    if (looks_multithreaded()) {
        fprintf(stderr, "[actor] isolation: refusing to apply after threads have started. "
                        "actor_isolation_apply() must run before any pthread_create.\n");
        return -1;
    }

    /* Limits before the privilege drop: setrlimit can lower a hard limit but
       an unprivileged process cannot raise one back, so anything that needs
       privilege to set must happen while the process still has it.
       RLIMIT_NPROC is the exception that makes the ordering visible -- the
       kernel does not enforce it for real uid 0 or for a process holding
       CAP_SYS_ADMIN/CAP_SYS_RESOURCE, so it only starts to bite once the uid
       drop below has happened. Setting it here and dropping after is what
       makes it mean anything. */
    /* RLIMIT_NPROC deserves a warning the others do not. It counts every
       process AND thread already running under the REAL uid, system wide --
       not this actor's descendants. An actor sharing a uid with anything else
       (a login session, another actor, a busy service account) inherits that
       count, so a limit that looks generous can be below the current usage
       before this process creates a single thread. The failure then lands on
       the next pthread_create as EAGAIN, which surfaces from whatever library
       happened to ask first -- observed as "nng_pub0_open: Out of memory",
       which points nowhere near the cause.
       Give an actor its own uid before setting this, and size it against
       `ps -L -u <uid> | wc -l` rather than against the actor's own thread
       count. */
    if (apply_rlimit("ACTOR_RLIMIT_AS",     RLIMIT_AS,     "RLIMIT_AS")     < 0) return -1;
    if (apply_rlimit("ACTOR_RLIMIT_CPU",    RLIMIT_CPU,    "RLIMIT_CPU")    < 0) return -1;
    if (apply_rlimit("ACTOR_RLIMIT_NOFILE", RLIMIT_NOFILE, "RLIMIT_NOFILE") < 0) return -1;
    if (apply_rlimit("ACTOR_RLIMIT_NPROC",  RLIMIT_NPROC,  "RLIMIT_NPROC")  < 0) return -1;

    if (drop_privilege() < 0) return -1;

    /* One line naming what is actually in force, so an operator can tell
       isolation happened without reading the launcher's configuration. */
    fprintf(stderr, "[actor] isolation: uid=%u gid=%u", (unsigned)getuid(), (unsigned)getgid());
    const char* v;
    if ((v = getenv("ACTOR_RLIMIT_AS")))     fprintf(stderr, " as=%s", v);
    if ((v = getenv("ACTOR_RLIMIT_CPU")))    fprintf(stderr, " cpu=%s", v);
    if ((v = getenv("ACTOR_RLIMIT_NOFILE"))) fprintf(stderr, " nofile=%s", v);
    if ((v = getenv("ACTOR_RLIMIT_NPROC")))  fprintf(stderr, " nproc=%s", v);
    fprintf(stderr, "\n");
    return 0;
}
