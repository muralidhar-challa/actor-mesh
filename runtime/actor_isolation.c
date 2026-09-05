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
#include <stddef.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <grp.h>
#include <fcntl.h>
#include <linux/landlock.h>
#include <sys/syscall.h>
#include <sys/prctl.h>

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



/* ── cgroup v2 ───────────────────────────────────────────────────────────────
 *
 * Join an existing cgroup by writing this pid into its cgroup.procs.
 *
 * The runtime joins and never creates. Creating a hierarchy means deciding
 * where it lives, who owns it, what controllers it enables and who cleans it
 * up -- all of which belong to whatever launches actors, which already knows
 * those answers. An actor that had to create its own would also need write
 * access to the parent cgroup directory, which is a larger privilege than
 * anything else this module asks for.
 *
 * Note this joins the ACTOR, and every handler it forks inherits the
 * membership. Per-tuple accounting is not possible this way and is not
 * attempted.
 */
static int join_cgroup(void) {
    const char* path = getenv("ACTOR_CGROUP_PATH");
    if (!path) return 0;

    char procs[PATH_MAX];
    int n = snprintf(procs, sizeof(procs), "%s/cgroup.procs", path);
    if (n < 0 || (size_t)n >= sizeof(procs)) {
        fprintf(stderr, "[actor] isolation: ACTOR_CGROUP_PATH too long\n");
        return -1;
    }

    FILE* f = fopen(procs, "w");
    if (!f) {
        fprintf(stderr, "[actor] isolation: cgroup \"%s\": %s "
                        "(the cgroup must already exist and be writable; "
                        "the runtime never creates one)\n", procs, strerror(errno));
        return -1;
    }
    if (fprintf(f, "%d\n", (int)getpid()) < 0 || fclose(f) != 0) {
        fprintf(stderr, "[actor] isolation: writing pid to \"%s\": %s\n",
                procs, strerror(errno));
        return -1;
    }

    /* Verify membership rather than trust the write. A cgroup.procs write can
       be accepted and still not place the process where it was asked -- for
       instance when the target has domain-controller constraints -- and an
       actor that believes it is accounted for when it is not is exactly the
       silent degradation this module refuses. */
    FILE* sf = fopen("/proc/self/cgroup", "r");
    if (!sf) {
        fprintf(stderr, "[actor] isolation: cannot read /proc/self/cgroup to verify\n");
        return -1;
    }
    char line[PATH_MAX + 64];
    int joined = 0;
    /* Unified hierarchy lines look like "0::/path/relative/to/mount". */
    while (fgets(line, sizeof(line), sf)) {
        char* rel = strstr(line, "0::");
        if (!rel) continue;
        rel += 3;
        line[strcspn(line, "\n")] = '\0';
        size_t rl = strlen(rel);
        if (rl && rl <= strlen(path) && strcmp(path + strlen(path) - rl, rel) == 0) joined = 1;
        break;
    }
    fclose(sf);
    if (!joined) {
        fprintf(stderr, "[actor] isolation: cgroup write to \"%s\" was accepted but "
                        "the process is not a member\n", path);
        return -1;
    }

    fprintf(stderr, "[actor] isolation: cgroup=%s\n", path);
    return 0;
}

/* ── Landlock ────────────────────────────────────────────────────────────────
 *
 * Path-based confinement: after landlock_restrict_self the process may only
 * touch what was explicitly granted. Everything else fails as though it were
 * not there.
 *
 * Two things about the read set are easy to get wrong and expensive to debug,
 * so they are stated here rather than left to the operator:
 *
 *   - ACTOR_LMDB_PATH must be inside ACTOR_LANDLOCK_RW. The actor opens its
 *     LMDB after this runs, so a path outside the ruleset turns into an
 *     LMDB error at startup rather than an obvious configuration one. This
 *     module checks for it explicitly and says so.
 *
 *   - The exec path needs more than the handler binary. The runtime runs
 *     handlers via execl("/bin/sh", "sh", "-c", ...), so /bin/sh, the handler
 *     itself, the dynamic loader and every shared library they pull in must be
 *     readable and executable. On a distro build of this runtime that is
 *     /lib64 and /usr/lib64 (libnng, liblmdb, libc, libtinfo, the mbedtls
 *     trio, ld-linux). A statically linked handler needs far less.
 *
 * There is no ambient "allow everything not mentioned": a ruleset with no
 * rules denies everything. So an ACTOR_LANDLOCK_RO that forgets the loader
 * produces a handler that cannot start, which is why phase 2 does not ship
 * without the enumeration above having been done for a real handler.
 */

#ifndef LANDLOCK_ACCESS_FS_REFER
#  define LANDLOCK_ACCESS_FS_REFER (1ULL << 13)
#endif
#ifndef LANDLOCK_ACCESS_FS_TRUNCATE
#  define LANDLOCK_ACCESS_FS_TRUNCATE (1ULL << 14)
#endif
#ifndef LANDLOCK_ACCESS_FS_IOCTL_DEV
#  define LANDLOCK_ACCESS_FS_IOCTL_DEV (1ULL << 15)
#endif

static long ll_create_ruleset(const struct landlock_ruleset_attr* attr,
                              size_t size, __u32 flags) {
    return syscall(__NR_landlock_create_ruleset, attr, size, flags);
}
static long ll_add_rule(int fd, enum landlock_rule_type t,
                        const void* attr, __u32 flags) {
    return syscall(__NR_landlock_add_rule, fd, t, attr, flags);
}
static long ll_restrict_self(int fd, __u32 flags) {
    return syscall(__NR_landlock_restrict_self, fd, flags);
}

/* Read-only and read-write access sets. Both include EXECUTE: the handler and
   its libraries are read through the RO set, and a handler staged into a
   writable working directory still has to be runnable. */
#define LL_RO (LANDLOCK_ACCESS_FS_READ_FILE | \
               LANDLOCK_ACCESS_FS_READ_DIR  | \
               LANDLOCK_ACCESS_FS_EXECUTE)

#define LL_RW (LL_RO                             | \
               LANDLOCK_ACCESS_FS_WRITE_FILE     | \
               LANDLOCK_ACCESS_FS_REMOVE_FILE    | \
               LANDLOCK_ACCESS_FS_REMOVE_DIR     | \
               LANDLOCK_ACCESS_FS_MAKE_REG       | \
               LANDLOCK_ACCESS_FS_MAKE_DIR       | \
               LANDLOCK_ACCESS_FS_MAKE_SOCK      | \
               LANDLOCK_ACCESS_FS_MAKE_FIFO      | \
               LANDLOCK_ACCESS_FS_MAKE_SYM       | \
               LANDLOCK_ACCESS_FS_TRUNCATE)

/* Add one path to the ruleset. A path that does not exist is a configuration
   error: silently skipping it would grant less than asked for, which is the
   silent-degradation failure this module exists to prevent. */
static int ll_add_path(int rs, const char* path, uint64_t allowed) {
    int fd = open(path, O_PATH | O_CLOEXEC);
    if (fd < 0) {
        fprintf(stderr, "[actor] isolation: landlock path \"%s\": %s\n",
                path, strerror(errno));
        return -1;
    }
    struct landlock_path_beneath_attr pb = { .allowed_access = allowed,
                                             .parent_fd = fd };
    int rc = (int)ll_add_rule(rs, LANDLOCK_RULE_PATH_BENEATH, &pb, 0);
    if (rc) fprintf(stderr, "[actor] isolation: landlock_add_rule(\"%s\"): %s\n",
                    path, strerror(errno));
    close(fd);
    return rc;
}

/* Walk a colon-separated list, applying fn to each non-empty element. */
static int for_each_path(const char* list, int rs, uint64_t allowed,
                         int (*fn)(int, const char*, uint64_t)) {
    if (!list || !*list) return 0;
    char buf[4096];
    if (strlen(list) >= sizeof(buf)) {
        fprintf(stderr, "[actor] isolation: landlock path list too long\n");
        return -1;
    }
    strcpy(buf, list);
    char* save = NULL;
    for (char* t = strtok_r(buf, ":", &save); t; t = strtok_r(NULL, ":", &save)) {
        while (*t == ' ') t++;
        if (!*t) continue;
        if (fn(rs, t, allowed) != 0) return -1;
    }
    return 0;
}

/* Is `path` at or beneath any element of the colon-separated `list`? Used to
   check ACTOR_LMDB_PATH against the rw set before the ruleset is enforced, so
   the operator gets a clear message instead of an LMDB failure later. */
static int path_covered(const char* path, const char* list) {
    if (!list || !*list || !path) return 0;
    char rp[PATH_MAX];
    /* The LMDB directory may not exist yet; compare textually against the
       configured value in that case rather than failing the check. */
    const char* target = realpath(path, rp) ? rp : path;
    size_t tlen = strlen(target);

    char buf[4096];
    if (strlen(list) >= sizeof(buf)) return 0;
    strcpy(buf, list);
    char* save = NULL;
    for (char* t = strtok_r(buf, ":", &save); t; t = strtok_r(NULL, ":", &save)) {
        while (*t == ' ') t++;
        if (!*t) continue;
        char er[PATH_MAX];
        const char* entry = realpath(t, er) ? er : t;
        size_t elen = strlen(entry);
        if (elen == 1 && entry[0] == '/') return 1;              /* "/" covers all */
        if (tlen >= elen && strncmp(target, entry, elen) == 0 &&
            (target[elen] == '\0' || target[elen] == '/')) return 1;
    }
    return 0;
}

static int apply_landlock(void) {
    const char* ro  = getenv("ACTOR_LANDLOCK_RO");
    const char* rw  = getenv("ACTOR_LANDLOCK_RW");
    const char* net = getenv("ACTOR_LANDLOCK_NET_CONNECT");
    if (!ro && !rw && !net) return 0;

    /* Negotiate the ABI. The kernel reports the highest version it supports;
       anything we ask for beyond that is silently absent, which is the one
       outcome not acceptable here -- so the unsupported case is an error, not
       a downgrade. */
    long abi = ll_create_ruleset(NULL, 0, LANDLOCK_CREATE_RULESET_VERSION);
    if (abi < 0) {
        fprintf(stderr, "[actor] isolation: landlock unavailable (%s). "
                        "Needs kernel 5.13+ with landlock enabled and in the "
                        "active LSM list.\n", strerror(errno));
        return -1;
    }
    if (net && abi < 4) {
        fprintf(stderr, "[actor] isolation: ACTOR_LANDLOCK_NET_CONNECT needs "
                        "landlock ABI 4 (kernel 6.7+); this kernel reports ABI %ld\n", abi);
        return -1;
    }

    /* Handled access must not name rights the running kernel does not know,
       or landlock_create_ruleset rejects the whole attr. Build it up by ABI. */
    uint64_t fs_handled = LL_RW;
    if (abi < 3) fs_handled &= ~(uint64_t)LANDLOCK_ACCESS_FS_TRUNCATE;
    /* REFER (ABI 2+) is deliberately not handled: it is always denied without
       an explicit rule, and handling it would break rename/link inside a
       single rw subtree, which handlers legitimately do. */

    struct landlock_ruleset_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.handled_access_fs = fs_handled;
    size_t attr_size = sizeof(attr);
    if (abi >= 4 && net) {
        attr.handled_access_net = LANDLOCK_ACCESS_NET_CONNECT_TCP;
    } else if (abi < 4) {
        /* Older kernels have no net field; submit the smaller struct. */
        attr_size = offsetof(struct landlock_ruleset_attr, handled_access_fs)
                  + sizeof(attr.handled_access_fs);
    }

    int rs = (int)ll_create_ruleset(&attr, attr_size, 0);
    if (rs < 0) {
        fprintf(stderr, "[actor] isolation: landlock_create_ruleset: %s\n", strerror(errno));
        return -1;
    }

    /* The actor opens its own LMDB after this ruleset is enforced. A path
       outside the rw set turns into an opaque LMDB failure at startup, so
       catch it here where the message can name the actual problem. */
    const char* lmdb = getenv("ACTOR_LMDB_PATH");
    if (lmdb && !path_covered(lmdb, rw)) {
        fprintf(stderr, "[actor] isolation: ACTOR_LMDB_PATH=\"%s\" is not inside "
                        "ACTOR_LANDLOCK_RW=\"%s\" -- the actor could not open its "
                        "own database\n", lmdb, rw ? rw : "");
        close(rs);
        return -1;
    }

    if (for_each_path(ro, rs, LL_RO, ll_add_path) < 0) { close(rs); return -1; }
    if (for_each_path(rw, rs, fs_handled, ll_add_path) < 0) { close(rs); return -1; }

    /* Outbound TCP ports, when asked for. */
    if (net && abi >= 4) {
        char buf[1024];
        if (strlen(net) >= sizeof(buf)) {
            fprintf(stderr, "[actor] isolation: ACTOR_LANDLOCK_NET_CONNECT too long\n");
            close(rs); return -1;
        }
        strcpy(buf, net);
        char* save = NULL;
        for (char* t = strtok_r(buf, ":", &save); t; t = strtok_r(NULL, ":", &save)) {
            while (*t == ' ') t++;
            if (!*t) continue;
            unsigned long long port;
            if (parse_ull(t, &port) || port > 65535) {
                fprintf(stderr, "[actor] isolation: bad port \"%s\" in "
                                "ACTOR_LANDLOCK_NET_CONNECT\n", t);
                close(rs); return -1;
            }
            struct landlock_net_port_attr np = {
                .allowed_access = LANDLOCK_ACCESS_NET_CONNECT_TCP,
                .port = port
            };
            if (ll_add_rule(rs, LANDLOCK_RULE_NET_PORT, &np, 0)) {
                fprintf(stderr, "[actor] isolation: landlock_add_rule(port %llu): %s\n",
                        port, strerror(errno));
                close(rs); return -1;
            }
        }
    }

    /* no_new_privs is required before restrict_self, and is wanted anyway:
       it stops a setuid binary anywhere below this process regaining what was
       just dropped. */
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)) {
        fprintf(stderr, "[actor] isolation: prctl(NO_NEW_PRIVS): %s\n", strerror(errno));
        close(rs); return -1;
    }
    if (ll_restrict_self(rs, 0)) {
        fprintf(stderr, "[actor] isolation: landlock_restrict_self: %s\n", strerror(errno));
        close(rs); return -1;
    }
    close(rs);
    fprintf(stderr, "[actor] isolation: landlock abi=%ld ro=%s rw=%s net=%s\n",
            abi, ro ? ro : "-", rw ? rw : "-", net ? net : "-");
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
        getenv("ACTOR_RLIMIT_NOFILE") || getenv("ACTOR_RLIMIT_NPROC")  ||
        getenv("ACTOR_LANDLOCK_RO")   || getenv("ACTOR_LANDLOCK_RW")   ||
        getenv("ACTOR_LANDLOCK_NET_CONNECT") || getenv("ACTOR_CGROUP_PATH");
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

    /* cgroup before Landlock: joining needs to open a path under
       /sys/fs/cgroup, which an operator has no reason to put in the ruleset. */
    if (join_cgroup() < 0) return -1;

    /* Landlock last: it is irreversible, and everything above still needs an
       unrestricted view of the filesystem to do its job. */
    if (apply_landlock() < 0) return -1;

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
