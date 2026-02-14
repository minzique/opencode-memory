# OpenCode Memory — Architecture

Persistent, cross-instance memory for AI agents running in OpenCode.

## Problem

AI agents in OpenCode lose all context between sessions. When context compacts or a session ends, decisions, constraints, user preferences, and learned patterns vanish. Multi-machine setups (laptop + Mac Mini) make this worse — each instance is fully isolated.

## Design Principles

1. **Mac Mini is the hub** — always-on, hosts the authoritative memory store
2. **Local-first** — each machine has fast local access, no cloud required
3. **Quality over quantity** — save decisions and constraints, not transcripts
4. **Semantic recall** — "what did we decide about auth?" should work
5. **Zero heavy deps** — no Docker, no PyTorch, no GPU required

## Hardware

| Machine | Role | Specs | Network |
|---------|------|-------|---------|
| Mac Mini | **Base of operations** — hosts memory service, OpenCode serve | M4 10-core, 16GB RAM, 32GB free | `minzis-mac-mini.local` |
| MacBook Pro | Primary development machine, connects to Mac Mini | 8GB RAM, 35GB free | LAN |
| Future | Any machine with HTTP access to Mac Mini | — | LAN/VPN |

## Architecture

```
                         Mac Mini (always-on)
                    ┌─────────────────────────────┐
                    │                             │
                    │   Memory Service :4097      │ ◄── Python FastAPI
                    │   ┌───────────────────┐     │     + sqlite-vec
                    │   │ SQLite + vectors  │     │     + OpenAI embeddings
                    │   │ quality gates     │     │
                    │   │ memory types      │     │
                    │   └───────────────────┘     │
                    │                             │
                    │   OpenCode Serve :4096      │ ◄── Agent execution
                    │                             │
                    └──────────┬──────────────────┘
                               │
                          HTTP │ (LAN)
                               │
          ┌────────────────────┼─────────────────────┐
          │                    │                     │
    MacBook Pro           Future Machine         Mobile/Web
    ┌──────────┐         ┌──────────┐
    │ OpenCode │         │ OpenCode │
    │          │         │          │
    │ MCP:     │         │ MCP:     │
    │ memory   │◄─HTTP──►│ memory   │
    │ (client) │         │ (client) │
    │          │         │          │
    │ MCP:     │         │ MCP:     │
    │ bridge   │◄─HTTP──►│ bridge   │
    └──────────┘         └──────────┘
```

## Memory Model

Three tiers, inspired by cognitive science:

### Tier 1: Working Memory (in-session)
- Handled by OpenCode's context window
- Todos, current task state, recent messages
- Lost on compaction/session end → captured by Tier 2

### Tier 2: Episodic Memory (session snapshots)
- **What**: Structured snapshots of session state
- **When**: Captured before compaction, on session end, on explicit save
- **TTL**: 7-30 days (configurable)
- **Content**: Todos, decisions, constraints, failed approaches, explored files
- **Schema**: `opencode-continuity` Snapshot type

### Tier 3: Semantic Memory (long-term patterns)
- **What**: Learned patterns, user preferences, project conventions
- **When**: Extracted from episodes, explicitly taught
- **TTL**: Permanent (with decay scoring)
- **Content**: Typed patterns with embeddings for semantic search
- **Types**: preference, convention, constraint, decision, architecture, error-solution

### Memory Types

```
┌─────────────────────────────────────────────────────┐
│ Episode                                              │
│  session_id, timestamp                               │
│  todos: [{content, status, priority}]                │
│  decisions: [{content, context, confidence}]         │
│  constraints: [{content, type, source}]              │
│  failed_approaches: [{approach, error}]              │
│  explored_files: [string]                            │
│  summary: string                                     │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ Pattern                                              │
│  content: string (the knowledge)                     │
│  embedding: float[1536] (for semantic search)        │
│  type: preference | convention | constraint | ...    │
│  scope: user | project                               │
│  project_id?: string                                 │
│  score: 0-1 (quality/relevance)                      │
│  retrieval_count: number                             │
│  tags: string[]                                      │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ Entity (knowledge graph node)                        │
│  name: string                                        │
│  type: person | project | tool | concept             │
│  observations: string[] (facts about this entity)    │
│  relations: [{target, type, metadata}]               │
└─────────────────────────────────────────────────────┘
```

## Quality Gates

Every memory candidate is scored before storage:

| Gate | Check | Action |
|------|-------|--------|
| Deduplication | Cosine similarity > 0.92 with existing | Merge or skip |
| Staleness | Not retrieved in N days | Decay score |
| Confidence | User-confirmed vs inferred | Weight accordingly |
| Size Budget | Total context < token limit | Prune lowest scores |
| Specificity | Generic advice vs project-specific | Prefer specific |

## API Design

### REST Endpoints (Memory Service on Mac Mini)

```
POST   /api/remember     — Store a memory (auto-classifies type)
GET    /api/recall        — Semantic search across memories
GET    /api/recall/:id    — Get specific memory by ID
DELETE /api/forget/:id    — Delete a memory
POST   /api/episode       — Save a session episode/snapshot
GET    /api/episodes      — List recent episodes
GET    /api/patterns      — List patterns (with filters)
POST   /api/entity        — Create/update knowledge graph entity
GET    /api/entity/:name  — Get entity with observations
GET    /api/graph          — Query knowledge graph
GET    /api/status         — Service health + storage stats
POST   /api/prune          — Trigger cleanup of stale memories
```

### MCP Tools (Client on each machine)

| Tool | Description |
|------|-------------|
| `remember` | Store a memory — text + optional type/tags/scope |
| `recall` | Semantic search — returns ranked results |
| `forget` | Delete a memory by ID |
| `save_episode` | Capture current session state as episode |
| `list_episodes` | Browse recent session episodes |
| `teach` | Explicitly teach a pattern (high confidence) |
| `memory_status` | Show storage stats and health |

## Implementation Phases

### Phase 1: Immediate Memory (MCP Knowledge Graph)

Zero custom code. Install official `@modelcontextprotocol/server-memory` on both machines.

- Storage: JSONL file (entities + relations + observations)
- Search: Exact name match (no semantic)
- Sync: Git-synced JSONL file OR independent stores
- Value: Persistent knowledge graph, works today

### Phase 2: Python Memory Service (Mac Mini)

Custom service with semantic search and quality gates.

- Storage: SQLite + sqlite-vec (Python, not Bun — Bun's binding is broken)
- Embeddings: OpenAI text-embedding-3-small via API
- Server: FastAPI on Mac Mini port 4097
- Client: Thin MCP server (TypeScript) wrapping HTTP calls
- Quality: Dedup, decay, scoring from opencode-continuity design

### Phase 3: Future Enhancements

- Local embeddings via Ollama (when installed)
- Knowledge graph layer (Neo4j or in-SQLite)
- Cloud backup (S3/Turso) when storage fills
- Compaction hooks (auto-capture before context loss)
- Cross-project pattern sharing

## Reuse from Existing Projects

| Source | What to Reuse |
|--------|---------------|
| `opencode-continuity` types.ts | Snapshot, Pattern, Constraint, Decision, FailedApproach types |
| `opencode-continuity` store/ | SnapshotStore, PatternStore (SQLite schema + queries) |
| `opencode-continuity` snapshot/ | Extractor (todo/decision/constraint parsing), Builder (priority budget) |
| `opencode-continuity` config.ts | Config schema + loader |
| `aria-core` memory.py | Mem0 API pattern (add/search/get_all/delete) |
| `opencode-bridge` | HTTP client patterns, MCP tool registration |

## File Structure

```
opencode-memory/
├── ARCHITECTURE.md           # This document
├── README.md                 # Setup + usage guide
│
├── phase1/                   # Phase 1: Official memory server config
│   ├── setup.sh              # Install + configure on both machines
│   └── test.sh               # Verify memory works
│
├── service/                  # Phase 2: Python memory service (runs on Mac Mini)
│   ├── pyproject.toml
│   ├── memory_service/
│   │   ├── __init__.py
│   │   ├── main.py           # FastAPI app
│   │   ├── config.py         # Settings
│   │   ├── models.py         # Pydantic models (from opencode-continuity types)
│   │   ├── store.py          # SQLite + sqlite-vec storage
│   │   ├── embeddings.py     # OpenAI embedding client
│   │   ├── quality.py        # Quality gates (dedup, decay, scoring)
│   │   └── routes/
│   │       ├── remember.py
│   │       ├── recall.py
│   │       ├── episodes.py
│   │       ├── entities.py
│   │       └── admin.py
│   └── tests/
│       ├── test_store.py
│       ├── test_quality.py
│       ├── test_api.py
│       └── conftest.py
│
├── mcp-client/               # Phase 2: MCP wrapper (runs on each machine)
│   ├── package.json
│   ├── tsconfig.json
│   ├── src/
│   │   ├── index.ts          # MCP server entry
│   │   ├── client.ts         # HTTP client for memory service
│   │   └── tools/
│   │       ├── remember.ts
│   │       ├── recall.ts
│   │       └── episodes.ts
│   └── test/
│       └── mcp-test.ts
│
└── scripts/
    ├── deploy-mac-mini.sh    # Deploy service on Mac Mini
    └── configure-client.sh   # Configure MCP client on any machine
```
