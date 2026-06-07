# Claude Platform: Agent Setup & Multi-Agent Orchestration

> Reference for setting up agents on Claude Platform and mapping to Actor Mesh patterns.

## 1. Claude Platform Agent Creation (Console UI)

### Via Console

1. Go to [console.anthropic.com](https://console.anthropic.com)
2. **Workbench** → Create new agent
3. Configure:
   - **System Prompt**: Agent's role, personality, rules
   - **Model**: Claude Sonnet 4 (recommended for agents) or Opus 4
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
