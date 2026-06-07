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

---


> Reference for setting up agents on Claude Platform and mapping to Actor Mesh patterns.

## 1. Claude Platform Agent Creation (Console UI)

### Via Console

1. Go to [console.anthropic.com](https://console.anthropic.com)
2. **Workbench** → Create new agent
3. Configure:
   - **System Prompt**: Agent's role, personality, rules
   - **Model**: Claude Sonnet 4 (cost/performance), Haiku 3.5 (fast/cheap)
   - **Tools**: MCP servers or custom tool definitions
   - **Temperature**: 0.0 for deterministic, 0.3 for creative
4. **Deploy** → API endpoint or embedded chat

### Tool Definition (JSON Schema)

```json
{
  "name": "query_database",
  "description": "Execute a read-only SQL query",
  "input_schema": {
    "type": "object",
    "properties": {
      "sql": { "type": "string", "description": "SQL SELECT query" }
    },
    "required": ["sql"]
  }
}
```

### MCP Server Connection

In Claude Platform Console → Agent → Tools → Add MCP Server:

```json
{
  "mcpServers": {
    "sqlite": {
      "command": "python3",
      "args": ["mcp-server-sqlite"],
      "env": { "SQLITE_DB": "/path/to/db.sqlite" }
    },
    "jira": {
      "command": "npx",
      "args": ["@anthropic/mcp-server-jira"],
      "env": { "JIRA_URL": "https://company.atlassian.net", "JIRA_TOKEN": "..." }
    }
  }
}
```

## 2. Multi-Agent Pattern: Supervisor-Worker

From Anthropic's "Building Effective Agents" (Dec 2024).

```
┌─────────────┐
│ Supervisor   │ ← decides which worker to call
└──────┬──────┘
       │
  ┌────┼────┬────┐
  ↓    ↓    ↓    ↓
Worker Worker Worker Worker
  A     B     C     D
```

**Supervisor system prompt:**
```
You are a project coordinator. You have access to specialized workers.
Analyze each request. Route to the right worker. Aggregate results.
Never perform work a worker can do.
```

**Worker system prompt:**
```
You are a specialized agent. Execute only your assigned task.
Report results clearly. Do not make decisions outside your scope.
```

### Claude Platform Setup for 3-Agent Team

#### Agent 1: PM Agent
```
System: "You represent project management. Monitor timelines, risks, and priorities.
         Communicate status on Teams. Escalate critical issues."
Tools:  teams-search, teams-send-message
Model:  Claude Sonnet 4
```

#### Agent 2: Dev Agent  
```
System: "You represent the development team. Track code quality, deployments,
         and technical debt. Report blockers immediately."
Tools:  teams-search, teams-send-message, github-search
Model:  Claude Sonnet 4
```

#### Agent 3: Support Agent
```
System: "You monitor Jira tickets. You CANNOT create or update Jira issues.
         When you detect critical tickets, alert PM and Dev agents.
         Provide full ticket context. Let others decide actions."
Tools:  jira-search (read-only)
Model:  Claude Sonnet 4
```

### Inter-Agent Communication

On Claude Platform, agents communicate via **shared context** or **API chaining**:

```python
# Agent A calls Agent B
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    system="You are PM Agent. Analyze this dev update: " + dev_status,
    messages=[{"role": "user", "content": "What is the project impact?"}]
)
```

> **Actor Mesh equivalent**: Agents communicate via topics. Agent A publishes `pm_insight`. Agent B subscribes to `pm_insight`. No API call needed — the mesh handles routing.

## 3. Claude API: Tool Use Loop

The agentic loop in Claude's API:

```python
import anthropic

client = anthropic.Anthropic()

messages = [{"role": "user", "content": "How many employees?"}]

while True:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        system="You have SQL tools.",
        messages=messages,
        tools=[{
            "name": "query_db",
            "description": "Execute SQL",
            "input_schema": {
                "type": "object",
                "properties": {"sql": {"type": "string"}},
                "required": ["sql"]
            }
        }]
    )
    
    if response.stop_reason == "tool_use":
        # Execute tool
        tool = response.content[-1]
        result = execute_sql(tool.input["sql"])
        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool.id, "content": result}]
        })
    else:
        # Final answer
        print(response.content[-1].text)
        break
```

> **Actor Mesh equivalent**: Same loop. Tool execution happens via mesh topics (publish `sql_query`, subscribe to `sql_result`). The actor runtime handles the fork/exec, the mesh handles routing.

## 4. Auto Agent Creation (Claude Platform API)

Create agents programmatically via the Platform API:

```python
import requests

# Create agent
resp = requests.post(
    "https://api.anthropic.com/v1/agents",
    headers={"x-api-key": "...", "anthropic-version": "2025-01-01"},
    json={
        "name": "support-agent",
        "system": "You monitor Jira tickets. Read-only access.",
        "model": "claude-sonnet-4-20250514",
        "tools": [{"type": "mcp", "server": "jira-readonly"}]
    }
)
agent_id = resp.json()["id"]
```

> **Actor Mesh equivalent**: Start an actor process with env vars. Same result, zero API calls.
> ```sh
> ACTOR_ID=support-agent ACTOR_TOPIC=... ./bin/actor &
> ```

## 5. Comparison: Claude Platform vs Actor Mesh

| | Claude Platform | Actor Mesh |
|---|---|---|
| **Agent creation** | Console UI or API | env vars + `./bin/actor &` |
| **Tool execution** | MCP servers (stdio) | MCP servers via tool-bridge (same) |
| **Inter-agent comms** | API chaining (custom code) | Mesh topics (pub/sub, automatic) |
| **State management** | Conversation history in API | LMDB per actor, correlation IDs |
| **Durability** | API retry | LMDB inbox/outbox, crash recovery |
| **Observability** | Platform dashboard | Subscribe to `*` topic |
| **Scaling** | API rate limits | Add actors, no API calls between them |
| **Cost** | Per-token API cost | Local models supported, API optional |
| **Deployment** | Cloud only | Anywhere: edge, cloud, local |

## 6. Actor Mesh → Claude Platform Mappings

| Claude Concept | Actor Mesh Implementation |
|---|---|
| Agent | `ACTOR_ID=pm-agent ACTOR_HANDLER=llm-agent ./bin/actor &` |
| System prompt | LLM system prompt built from tool discovery |
| Tools | Registry → `_tool_announce` → `_tool_list` → agent |
| MCP server | `tool-bridge.sh` + any MCP server binary |
| Tool use loop | ReAct loop in llm-agent handler |
| Multi-agent | Topics as communication channels |
| Console UI | `make run-ui` (Clay desktop) or Quarto docs |
