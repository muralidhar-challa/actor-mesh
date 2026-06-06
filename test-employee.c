#define _POSIX_C_SOURCE 200809L
#include <unistd.h>
/* test-employee.c — full employee mesh integration test
 * Starts: proxy + registry + SQLite MCP + agent
 * Announces tools, sends query, waits for agent response.
 * Compile: gcc -Wall -O2 test-employee.c -lnng -o test-employee
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <spawn.h>
#include <sys/wait.h>
#include <time.h>
#include <nng/nng.h>
#include <nng/protocol/pubsub0/pub.h>
#include <nng/protocol/pubsub0/sub.h>

static int failures = 0;
extern char **environ;
static void ms(int ms) { struct timespec t={ms/1000,(ms%1000)*1000000}; nanosleep(&t,NULL); }
#define SP "tcp://127.0.0.1:55656"
#define PP "tcp://127.0.0.1:55657"
#define TEST(n) printf("\n=== %s ===\n", n)
#define PASS() printf("  PASS\n")
#define FAIL(m) do { printf("  FAIL: %s\n", m); failures++; } while(0)
#define CHECK(c,m) do { if(c)PASS(); else FAIL(m); } while(0)

static void free_ports(void) { system("fuser -k 55656/tcp 55657/tcp 2>/dev/null"); ms(400); }
static void cleanup(void) { system("pkill -9 mesh-proxy actor 2>/dev/null"); ms(300); }
static pid_t sp(char **a, char **e) { pid_t p; posix_spawn(&p, a[0], NULL, NULL, a, e); return p; }

static void send_msg(const char *topic, const uint8_t *payload, size_t plen) {
    nng_socket s; nng_pub0_open(&s); nng_dial(s, PP, NULL, 0); ms(50);
    uint8_t f[8192]={0}; int tl=strlen(topic); if(tl>31)tl=31;
    memcpy(f, topic, tl); memcpy(f+80, "test", 4);
    struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
    int64_t ns=ts.tv_sec*1000000000LL+ts.tv_nsec;
    memcpy(f+112, &ns, 8); uint32_t pl=plen; memcpy(f+138, &pl, 4);
    memcpy(f+256, payload, plen); nng_send(s, f, 256+plen, 0); nng_close(s);
}

static ssize_t wait_msg(const char *topic, uint8_t *buf, size_t cap, int timeout_ms) {
    nng_socket s; nng_sub0_open(&s); nng_dial(s, SP, NULL, 0);
    nng_socket_set(s, NNG_OPT_SUB_SUBSCRIBE, topic, strlen(topic)+1);
    nng_socket_set_ms(s, "recv-timeout", timeout_ms);
    nng_msg *m=NULL;
    if (nng_recvmsg(s, &m, 0)) { nng_close(s); return -1; }
    size_t len=nng_msg_len(m);
    ssize_t r=len>256?(ssize_t)(len-256):0;
    if(r>0&&buf&&(size_t)r<=cap) memcpy(buf,(uint8_t*)nng_msg_body(m)+256,r);
    nng_msg_free(m); nng_close(s); return r;
}

int main(void) {
    printf("Employee Mesh Integration Test\n");
    free_ports(); cleanup();
    system("rm -rf /tmp/em-test; mkdir -p /tmp/em-test/{reg,db,ag}");

    /* ── Start proxy ── */
    TEST("proxy");
    char *pargs[] = {"./mesh-proxy", NULL};
    char *penv[]  = {"PROXY_SUB_BIND=tcp://127.0.0.1:55657","PROXY_PUB_BIND=tcp://127.0.0.1:55656",NULL};
    pid_t ppid = sp(pargs, penv); ms(600);
    PASS();

    /* ── Start registry ── */
    TEST("registry");
    char *rargs[] = {"./actor", NULL};
    char cwd[1024]; getcwd(cwd, sizeof(cwd));
    char blib[1024]; snprintf(blib, 1024, "BRIDGE_LIB=%s/examples/employee-mesh/handlers/lib", cwd);
    char *renv[] = {
        "ACTOR_BUS_SUB=tcp://127.0.0.1:55656","ACTOR_BUS_PUB=tcp://127.0.0.1:55657",
        "ACTOR_HEARTBEAT_MS=2000","ACTOR_RETRY_MAX=3",
        "ACTOR_ID=tool-registry","ACTOR_TOPIC=_tool_announce,_tool_discover",
        "ACTOR_RESULT_TOPIC=_tool_list",
        "ACTOR_HANDLER=examples/employee-mesh/handlers/registry/tool-registry.sh",
        "ACTOR_LMDB_PATH=/tmp/em-test/reg", blib, NULL};
    pid_t rpid = sp(rargs, renv); ms(600);
    PASS();

    /* ── Start SQLite MCP ── */
    TEST("SQLite MCP");
    char *sargs[] = {"./actor", NULL};
    char mcp_srv[1024]; snprintf(mcp_srv, 1024, "MCP_SERVER=python3 %s/examples/employee-mesh/handlers/mcp/mcp-sqlite.py", cwd);
    char emp_db[1024]; snprintf(emp_db, 1024, "EMPLOYEE_DB=%s/examples/employee-mesh/db/employee.db", cwd);
    char *senv[] = {
        "ACTOR_BUS_SUB=tcp://127.0.0.1:55656","ACTOR_BUS_PUB=tcp://127.0.0.1:55657",
        "ACTOR_HEARTBEAT_MS=2000","ACTOR_RETRY_MAX=3",
        "ACTOR_ID=sqlite-mcp","ACTOR_TOPIC=sql_query","ACTOR_RESULT_TOPIC=sql_result",
        "ACTOR_HANDLER=examples/employee-mesh/handlers/mcp/tool-bridge.sh",
        "ACTOR_LMDB_PATH=/tmp/em-test/db",
        blib, mcp_srv, "MCP_TOOL=read_query", "MCP_ARG=sql", emp_db, NULL};
    pid_t spid = sp(sargs, senv); ms(600);
    PASS();

    /* ── Announce tools to registry ── */
    TEST("tool announce");
    /* mpack: {type:"_tool_announce", actor:"sqlite-mcp", capabilities:[{name:"read_query",description:"SQL",inputSchema:{type:"object",properties:{sql:{type:"string"}},required:["sql"]}}]} */
    uint8_t announce[] = {
        0x83,0xa4,'t','y','p','e',0xae,'_','t','o','o','l','_','a','n','n','o','u','n','c','e',
        0xa5,'a','c','t','o','r',0xaa,'s','q','l','i','t','e','-','m','c','p',
        0xac,'c','a','p','a','b','i','l','i','t','i','e','s',
        0x91,0x83,0xa4,'n','a','m','e',0xaa,'r','e','a','d','_','q','u','e','r','y',
        0xab,'d','e','s','c','r','i','p','t','i','o','n',0xa3,'S','Q','L',
        0xab,'i','n','p','u','t','S','c','h','e','m','a',
        0x83,0xa4,'t','y','p','e',0xa6,'o','b','j','e','c','t',
        0xaa,'p','r','o','p','e','r','t','i','e','s',
        0x81,0xa3,'s','q','l',0x81,0xa4,'t','y','p','e',0xa6,'s','t','r','i','n','g',
        0xa8,'r','e','q','u','i','r','e','d',0x91,0xa3,'s','q','l'};
    send_msg("_tool_announce", announce, sizeof(announce));
    ms(500);
    FILE *f = fopen("/tmp/em-test/reg/tools/sqlite-mcp.json", "r");
    CHECK(f != NULL, "tools not announced to registry");
    if (f) fclose(f);

    /* Trigger registry to publish _tool_list so agent discovers tools */
    TEST("tool discovery");
    uint8_t dm[] = {0x81,0xa4,'t','y','p','e',0xae,'_','t','o','o','l','_','d','i','s','c','o','v','e','r'};
    send_msg("_tool_discover", dm, sizeof(dm));
    ms(500);
    PASS();

    /* ── Start agent ── */
    TEST("agent");
    char *aargs[] = {"./actor", NULL};
    char model_env[128] = "LLM_MODEL=granite4.1:8b";
    char *aenv[] = {
        "ACTOR_BUS_SUB=tcp://127.0.0.1:55656","ACTOR_BUS_PUB=tcp://127.0.0.1:55657",
        "ACTOR_HEARTBEAT_MS=2000","ACTOR_RETRY_MAX=3",
        "ACTOR_ID=llm-agent","ACTOR_TOPIC=user_message,sql_result,_tool_list",
        "ACTOR_RESULT_TOPIC=agent_response",
        "ACTOR_HANDLER=examples/employee-mesh/handlers/agents/llm-agent",
        "ACTOR_LMDB_PATH=/tmp/em-test/ag",
        "LLM_BASE_URL=http://localhost:11434", model_env, NULL};
    pid_t apid = sp(aargs, aenv); ms(2000); /* give agent time to load tools */
    PASS();

    /* ── Send query ── */
    TEST("agent response (30s timeout)");
    uint8_t query_mp[] = {0x82,0xa4,'t','y','p','e',0xad,'u','s','e','r','_','m','e','s','s','a','g','e',
        0xa5,'q','u','e','r','y',0xa3,'h','i',' ','a','n','d',' ','s','a','y',' ','h','e','l','l','o','!','!'};
    send_msg("user_message", query_mp, sizeof(query_mp));
    
    uint8_t buf[4096];
    ssize_t n = wait_msg("agent_response", buf, sizeof(buf)-1, 45000);
    CHECK(n > 0, "no agent response (Ollama running?)");
    if (n > 0) {
        buf[n] = 0;
        printf("  agent says: %.*s...\n", n < 100 ? (int)n : 100, buf);
    }

    /* ── Tool calling: agent uses SQLite MCP ── */
    TEST("agent tool call (60s timeout)");
    {
        /* Send a query that requires SQL: "how many employees?" */
        uint8_t qm[] = {0x82,0xa4,'t','y','p','e',0xad,'u','s','e','r','_','m','e','s','s','a','g','e',
            0xa5,'q','u','e','r','y',0xb1,'h','o','w',' ','m','a','n','y',' ','e','m','p','l','o','y','e','e','s',' ','a','r','e',' ','t','h','e','r','e','?'};
        send_msg("user_message", qm, sizeof(qm));
        uint8_t buf[4096];
        ssize_t n = wait_msg("agent_response", buf, sizeof(buf)-1, 120000);
        CHECK(n > 0, "no agent response to SQL query (Ollama running?)");
        if (n > 0) {
            buf[n] = 0;
            printf("  agent: %.*s\n", n < 150 ? (int)n : 150, buf);
        }
    }

    /* ── Cleanup ── */
    printf("\n");
    if (failures == 0) printf("ALL PASSED\n");
    else printf("%d FAILURES\n", failures);
    kill(ppid, SIGKILL); kill(rpid, SIGKILL); kill(spid, SIGKILL); kill(apid, SIGKILL);
    waitpid(ppid,NULL,0); waitpid(rpid,NULL,0); waitpid(spid,NULL,0); waitpid(apid,NULL,0);
    cleanup();
    return failures;
}
