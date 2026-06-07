# Multi-Agent Orchestration

The actor mesh doesn't need an orchestrator. Topics ARE the orchestration.

## The pattern

```
Agent A ───pub: findings ──→ Agent B
Agent B ───pub: questions ──→ Agent A
Agent C ───pub: context ────→ Agent A, Agent B
```

No central process. No workflow engine. Each agent subscribes to the topics it cares about. The topology is defined by subscription, not configuration.

## Three-Agent Project Team

### Real-world scenario

| Agent | External I/O | Mesh Topics | Role |
|---|---|---|---|
| **PM Agent** | Teams (read/write) | `pm_insight`, `dev_status`, `support_alert` | Project manager perspective |
| **Dev Agent** | Teams (read/write) | `dev_status`, `pm_question`, `support_context` | Developer perspective |
| **Support Agent** | Jira (read-only) | `support_alert`, `support_context` | Ticket monitoring |

### Communication rules

- **PM ↔ Dev**: talk on Teams directly (human-facing). Mesh topics for agent-to-agent context sharing
- **Support → PM/Dev**: reads Jira, publishes alerts/context on mesh topics. PM/Dev receive and respond on Teams
- **Support Agent**: NEVER writes to Jira — read-only on tickets, writes only via mesh topics

### Setup

```sh
# ── Mesh bus ──
./bin/mesh-proxy &

# ── PM Agent ──
# Subscribes: dev_status, support_alert
# Publishes:  pm_insight
ACTOR_ID=pm-agent \
ACTOR_TOPIC=dev_status,support_alert \
ACTOR_RESULT_TOPIC=pm_insight \
ACTOR_HANDLER=./handlers/agents/llm-agent \
ACTOR_LMDB_PATH=/var/actor/pm \
LLM_API_KEY=sk-... ./bin/actor &

# ── Dev Agent ──
# Subscribes: pm_question, support_context
# Publishes:  dev_status
ACTOR_ID=dev-agent \
ACTOR_TOPIC=pm_question,support_context \
ACTOR_RESULT_TOPIC=dev_status \
ACTOR_HANDLER=./handlers/agents/llm-agent \
ACTOR_LMDB_PATH=/var/actor/dev \
LLM_API_KEY=sk-... ./bin/actor &

# ── Support Agent ──
# Subscribes: nothing from mesh (reads Jira on schedule)
# Publishes:  support_alert, support_context
ACTOR_ID=support-agent \
ACTOR_TOPIC=none \
ACTOR_RESULT_TOPIC=support_alert \
ACTOR_HANDLER=./handlers/agents/support-watcher.sh \
ACTOR_LMDB_PATH=/var/actor/support ./bin/actor &
```

### Inter-agent communication flow

```
Support Agent reads Jira → finds critical ticket
  → pub support_alert {severity: critical, ticket: "PAY-420", summary: "..."}

PM Agent receives support_alert
  → analyzes impact on timeline
  → pub pm_insight {risk: "schedule slip", affected: "sprint 4"}
  → writes Teams: "@channel Heads up: PAY-420 is critical, may impact sprint 4"

Dev Agent receives support_context (auto-follow on support_alert)
  → checks known issues, codebase
  → pub dev_status {ticket: "PAY-420", assessment: "known race condition in payment module"}
  → writes Teams: "Looking at PAY-420 — looks like the payment race condition. Working on fix."
```

### System prompt (example for Support Agent)

```
You monitor Jira tickets. You CANNOT write to Jira.
When you find a critical/severe ticket:
  1. Publish support_alert with severity, ticket ID, summary
  2. Publish support_context with full ticket details
  3. DO NOT take action — let PM/Dev agents decide

Jira access: read-only via MCP server
```

## Adding Teams + Jira via MCP

```sh
# Teams MCP server (reads/writes channels)
MCP_SERVER=teams-mcp-server MCP_TOOL=send_message \
ACTOR_HANDLER=./handlers/mcp/tool-bridge.sh ./bin/actor &

# Jira MCP server (read-only)
MCP_SERVER=jira-mcp-server MCP_TOOL=get_issue \
ACTOR_HANDLER=./handlers/mcp/tool-bridge.sh ./bin/actor &
```

## Why this works without an orchestrator

1. **Decoupled**: PM agent doesn't know Dev agent exists. It just subscribes to `dev_status` and publishes `pm_insight`
2. **Composable**: Add a "QA Agent" by subscribing to `dev_status` and publishing `qa_report`. Zero changes to existing agents
3. **Durable**: Every inter-agent message is a tuple with correlation ID, stored in LMDB. Crash recovery is automatic
4. **Observable**: Subscribe to `*` and see every inter-agent message. Full traceability
5. **Tool agnostic**: Teams, Jira, Slack, email — all are MCP servers. Agents don't know the difference
