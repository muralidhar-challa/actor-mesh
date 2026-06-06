#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <time.h>
#include <nng/nng.h>
#include <nng/protocol/pubsub0/pub.h>
#include <nng/protocol/pubsub0/sub.h>

static void msleep(int ms) { struct timespec ts={ms/1000,(ms%1000)*1000000}; nanosleep(&ts,NULL); }
static int failures = 0;
#define TEST(name) printf("=== %s ===\n", name)
#define PASS() printf("  PASS\n")
#define FAIL(msg) do { printf("  FAIL: %s\n", msg); failures++; } while(0)

int main(void) {
    system("pkill -9 mesh-proxy actor 2>/dev/null");
    msleep(500);
    /* Free ports */
    system("fuser -k 15556/tcp 15557/tcp 25556/tcp 25557/tcp 35556/tcp 35557/tcp 2>/dev/null");
    msleep(300);
    
    /* ── Test 1: Proxy ── */
    TEST("proxy forwarding");
    system("PROXY_SUB_BIND=tcp://127.0.0.1:15557 PROXY_PUB_BIND=tcp://127.0.0.1:15556 ./mesh-proxy &");
    msleep(500);
    
    nng_socket sub, pub;
    nng_sub0_open(&sub); nng_dial(sub, "tcp://127.0.0.1:15556", NULL, 0);
    nng_socket_set(sub, NNG_OPT_SUB_SUBSCRIBE, "t1", 2);
    nng_socket_set_ms(sub, "recv-timeout", 2000);
    nng_pub0_open(&pub); nng_dial(pub, "tcp://127.0.0.1:15557", NULL, 0);
    msleep(100);
    
    char f[260]={0}; memcpy(f, "t1", 2);
    nng_send(pub, f, sizeof(f), 0);
    nng_msg *m = NULL;
    if (nng_recvmsg(sub, &m, 0) == 0) PASS(); else FAIL("no forward");
    if (m) nng_msg_free(m);
    nng_close(sub); nng_close(pub);
    system("pkill -9 mesh-proxy 2>/dev/null"); msleep(300);

    /* ── Test 2: Actor ── */
    TEST("actor receives");
    system("mkdir -p /tmp/test-mesh-actor");
    system("PROXY_SUB_BIND=tcp://127.0.0.1:25557 PROXY_PUB_BIND=tcp://127.0.0.1:25556 ./mesh-proxy &");
    msleep(500);
    system("ACTOR_BUS_SUB=tcp://127.0.0.1:25556 ACTOR_BUS_PUB=tcp://127.0.0.1:25557 "
           "ACTOR_ID=test ACTOR_TOPIC=hello ACTOR_RESULT_TOPIC=world ACTOR_HEARTBEAT_MS=0 "
           "ACTOR_HANDLER='sh -c \"echo world; echo ok\"' "
           "ACTOR_LMDB_PATH=/tmp/test-mesh-actor ./actor &");
    msleep(500);
    
    nng_sub0_open(&sub); nng_dial(sub, "tcp://127.0.0.1:25556", NULL, 0);
    nng_socket_set(sub, NNG_OPT_SUB_SUBSCRIBE, "world", 5);
    nng_socket_set_ms(sub, "recv-timeout", 3000);
    nng_pub0_open(&pub); nng_dial(pub, "tcp://127.0.0.1:25557", NULL, 0);
    msleep(100);
    memset(f,0,260); memcpy(f,"hello",5);
    nng_send(pub, f, sizeof(f), 0);
    m = NULL;
    if (nng_recvmsg(sub, &m, 0) == 0) { PASS(); if(m)nng_msg_free(m); }
    else FAIL("actor no response");
    nng_close(sub); nng_close(pub);
    system("pkill -9 mesh-proxy actor 2>/dev/null"); msleep(300);
    
    /* ── Test 3: Registry ── */
    TEST("registry stores tools");
    system("mkdir -p /tmp/test-mesh-reg");
    system("PROXY_SUB_BIND=tcp://127.0.0.1:35557 PROXY_PUB_BIND=tcp://127.0.0.1:35556 ./mesh-proxy &");
    msleep(500);
    system("rm -rf /tmp/test-mesh-reg/tools; mkdir -p /tmp/test-mesh-reg; ACTOR_BUS_SUB=tcp://127.0.0.1:35556 ACTOR_BUS_PUB=tcp://127.0.0.1:35557 "
           "ACTOR_ID=reg ACTOR_TOPIC=_tool_announce,_tool_discover ACTOR_RESULT_TOPIC=_tool_list ACTOR_HEARTBEAT_MS=0 "
           "ACTOR_HANDLER=examples/employee-mesh/handlers/registry/tool-registry.sh "
           "ACTOR_LMDB_PATH=/tmp/test-mesh-reg BRIDGE_LIB=examples/employee-mesh/handlers/lib ./actor &");
    msleep(500);
    
    nng_pub0_open(&pub); nng_dial(pub, "tcp://127.0.0.1:35557", NULL, 0);
    msleep(100);
    /* mpack: {type:_tool_announce, actor:"t", capabilities:[{name:"x"}]} */
    unsigned char p[]={0x83,0xa4,'t','y','p','e',0xaf,'_','t','o','o','l','_','a','n','n','o','u','n','c','e',
        0xa5,'a','c','t','o','r',0xa1,'t',0xac,'c','a','p','a','b','i','l','i','t','i','e','s',
        0x91,0x81,0xa4,'n','a','m','e',0xa1,'x'};
    char h[256]={0}; memcpy(h,"_tool_announce",15); h[80]='d';
    unsigned char f2[sizeof(h)+sizeof(p)];
    memcpy(f2,h,256); memcpy(f2+256,p,sizeof(p));
    nng_send(pub, f2, sizeof(f2), 0);
    nng_close(pub);
    msleep(500);
    
    FILE *fp = fopen("/tmp/test-mesh-reg/tools/t.json","r");
    if (fp) { PASS(); char b[256]; fgets(b,sizeof(b),fp); printf("  stored: %.60s\n",b); fclose(fp); }
    else FAIL("registry did not store");
    system("pkill -9 mesh-proxy actor 2>/dev/null"); msleep(300);
    
    printf("\n%s: %d failures\n", failures?"FAIL":"PASS", failures);
    return failures;
}
