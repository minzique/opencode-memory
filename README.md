# opencode-memory

Persistent memory service for AI agents — vector similarity search over memories with LLM-powered extraction.

## Overview

**opencode-memory** is a FastAPI-based HTTP service that provides semantic memory storage and retrieval for AI agents. It combines SQLite with sqlite-vec for vector similarity search, OpenAI embeddings for semantic understanding, and GPT-4o-mini for intelligent memory extraction from unstructured text.

The service runs on a Mac Mini at port 4097 and provides a REST API for storing, searching, and managing memories across multiple agent sessions and projects.

## Architecture

- **FastAPI** — async HTTP server with automatic OpenAPI docs
- **SQLite + sqlite-vec** — persistent storage with vector similarity search (cosine distance)
- **OpenAI embeddings** — text-embedding-3-small (1536 dimensions) for semantic search
- **GPT-4o-mini** — LLM-powered memory extraction from raw text
- **Quality gates** — near-duplicate detection (similarity ≥ 0.92 rejects), memory consolidation (0.85-0.92 merges)

## Features

### Memory Management
- **Semantic search** — natural language queries with vector similarity
- **Memory types** — preference, convention, constraint, decision, architecture, error-solution, fact, episode, failure, pattern, working_context
- **Scopes** — global, project, session
- **Deduplication** — automatic detection and rejection of near-duplicates (similarity ≥ 0.92)
- **Consolidation** — merges similar memories (0.85 ≤ similarity < 0.92) to reduce redundancy
- **Scoring** — combined ranking using semantic similarity (60%), recency (15%), access frequency (10%), and confidence (15%)

### LLM Extraction
- **Structured extraction** — GPT-4o-mini extracts typed memories from unstructured text
- **Auto-classification** — assigns memory types, confidence scores, and tags
- **Batch processing** — efficient embedding generation for multiple memories

### Working State
- **Project state** — save and restore working context per project
- **State fields** — objective, approach, progress, files_touched, tried_and_failed, next_steps, blockers, open_questions

### Episodes
- **Session snapshots** — structured capture of session state
- **Episode data** — todos, decisions, constraints, failed approaches, explored files
- **Auto-extraction** — decisions and constraints automatically converted to searchable memories

### Bootstrap
- **Full context loading** — get working state, relevant memories, recent episodes, constraints, and failed approaches in one call
- **Project-scoped** — automatically filters by project_id or infers from cwd

## API Reference

### Memory Operations

#### `POST /remember`
Store a new memory with automatic embedding generation.

**Request:**
```json
{
  "content": "User prefers TypeScript over JavaScript for new projects",
  "type": "preference",
  "scope": "global",
  "project_id": null,
  "tags": ["typescript", "language"],
  "source": "session_abc123",
  "confidence": 1.0,
  "metadata": {}
}
```

**Response:**
```json
{
  "status": "created",
  "id": "mem_a1b2c3d4e5f6"
}
```

**Status codes:**
- `201` — memory created
- `200` — duplicate detected (returns `existing_id`) or consolidated (returns `existing_id`)

#### `POST /recall`
Search memories using semantic similarity.

**Request:**
```json
{
  "query": "What did we decide about authentication?",
  "limit": 10,
  "threshold": 0.3,
  "types": ["decision", "constraint"],
  "scope": "project",
  "project_id": "my-app",
  "tags": ["auth"]
}
```

**Response:**
```json
{
  "results": [
    {
      "memory": {
        "id": "mem_xyz789",
        "content": "Use JWT tokens with 15-minute expiry for API authentication",
        "type": "decision",
        "scope": "project",
        "project_id": "my-app",
        "tags": ["auth", "jwt"],
        "confidence": 0.9,
        "score": 1.0,
        "retrieval_count": 5,
        "created_at": 1707945600.0,
        "updated_at": 1707945600.0
      },
      "similarity": 0.87
    }
  ],
  "query": "What did we decide about authentication?",
  "total": 1
}
```

#### `GET /recall/{memory_id}`
Retrieve a specific memory by ID.

**Response:**
```json
{
  "id": "mem_xyz789",
  "content": "Use JWT tokens with 15-minute expiry",
  "type": "decision",
  "scope": "project",
  "project_id": "my-app",
  "tags": ["auth"],
  "confidence": 0.9,
  "score": 1.0,
  "retrieval_count": 5,
  "created_at": 1707945600.0,
  "updated_at": 1707945600.0
}
```

#### `DELETE /forget/{memory_id}`
Delete a memory and its vector embedding.

**Response:** `204 No Content`

### LLM Extraction

#### `POST /extract`
Extract structured memories from raw text using GPT-4o-mini.

**Request:**
```json
{
  "text": "We decided to use PostgreSQL for the database because it has better JSON support than MySQL. The team prefers to write tests in pytest rather than unittest.",
  "context": "Project setup discussion",
  "source": "session_abc123"
}
```

**Response:**
```json
{
  "extracted": 2,
  "memory_ids": ["mem_a1b2c3", "mem_d4e5f6"]
}
```

The service automatically:
1. Calls GPT-4o-mini to extract structured memories
2. Generates embeddings for each extracted memory
3. Stores memories with tags `["extracted", "source:session_abc123"]`
4. Returns the IDs of created memories

### Episodes

#### `POST /episode`
Save a session snapshot with structured data.

**Request:**
```json
{
  "session_id": "ses_abc123",
  "project_id": "my-app",
  "summary": "Implemented user authentication with JWT",
  "todos": [
    {"content": "Add refresh token endpoint", "status": "pending", "priority": "high"}
  ],
  "decisions": [
    {"content": "Use JWT with 15-minute expiry", "context": "Security review", "confidence": 0.9}
  ],
  "constraints": [
    {"content": "Must support OAuth2", "type": "must", "source": "requirements"}
  ],
  "failed_approaches": [
    {"approach": "Session cookies", "error": "CORS issues", "context": "Initial attempt"}
  ],
  "explored_files": ["src/auth/jwt.ts", "src/middleware/auth.ts"],
  "metadata": {}
}
```

**Response:**
```json
{
  "status": "created",
  "id": "ep_xyz789",
  "extracted_memories": 2
}
```

The service automatically extracts decisions and constraints as searchable memories.

#### `GET /episodes`
List recent episodes with optional project filtering.

**Query params:**
- `project_id` (optional) — filter by project
- `limit` (default: 20) — max episodes to return

**Response:**
```json
[
  {
    "id": "ep_xyz789",
    "session_id": "ses_abc123",
    "project_id": "my-app",
    "summary": "Implemented user authentication",
    "todos": [...],
    "decisions": [...],
    "constraints": [...],
    "failed_approaches": [...],
    "explored_files": [...],
    "created_at": 1707945600.0,
    "metadata": {}
  }
]
```

### Working State

#### `PUT /state/{project_id}`
Save working state for a project.

**Request:**
```json
{
  "objective": "Implement user authentication",
  "approach": "JWT-based auth with refresh tokens",
  "progress": "Completed login endpoint, working on refresh",
  "files_touched": ["src/auth/jwt.ts", "src/routes/auth.ts"],
  "tried_and_failed": ["Session-based auth (CORS issues)"],
  "next_steps": ["Add refresh token endpoint", "Write integration tests"],
  "blockers": ["Need to decide on token storage strategy"],
  "open_questions": ["Should we use httpOnly cookies or localStorage?"],
  "metadata": {}
}
```

**Response:**
```json
{
  "status": "saved",
  "project_id": "my-app"
}
```

#### `GET /state/{project_id}`
Retrieve working state for a project.

**Response:**
```json
{
  "data": {
    "objective": "Implement user authentication",
    "approach": "JWT-based auth with refresh tokens",
    ...
  },
  "updated_at": 1707945600.0
}
```

#### `DELETE /state/{project_id}`
Clear working state for a project.

**Response:** `204 No Content`

### Bootstrap

#### `POST /bootstrap`
Load full session context in one call.

**Request:**
```json
{
  "project_id": "my-app",
  "cwd": "/Users/dev/my-app",
  "include_state": true,
  "include_memories": true,
  "include_episodes": true,
  "memory_limit": 15,
  "episode_limit": 3
}
```

**Response:**
```json
{
  "project_id": "my-app",
  "state": {
    "objective": "Implement user authentication",
    "approach": "JWT-based auth",
    ...
  },
  "memories": [
    {
      "id": "mem_abc123",
      "content": "User prefers TypeScript",
      "type": "preference",
      ...
    }
  ],
  "recent_episodes": [
    {
      "id": "ep_xyz789",
      "summary": "Implemented authentication",
      ...
    }
  ],
  "constraints": [
    {
      "id": "mem_def456",
      "content": "Must support OAuth2",
      "type": "constraint",
      ...
    }
  ],
  "failed_approaches": [
    {
      "id": "mem_ghi789",
      "content": "Session cookies failed due to CORS",
      "type": "error-solution",
      ...
    }
  ]
}
```

The bootstrap endpoint:
- Auto-detects `project_id` from `cwd` if not provided
- Loads working state
- Fetches relevant memories (preference, convention, constraint, architecture types)
- Includes recent episodes
- Separates constraints and failed approaches for easy access

### Service Status

#### `GET /status`
Service health and statistics.

**Response:**
```json
{
  "healthy": true,
  "version": "0.3.0",
  "memory_count": 1247,
  "episode_count": 89,
  "state_count": 12,
  "db_size_bytes": 5242880,
  "embedding_provider": "openai",
  "uptime_seconds": 86400.5
}
```

#### `GET /health`
Simple health check.

**Response:**
```json
{
  "status": "ok"
}
```

#### `GET /`
Service info and endpoint list.

**Response:**
```json
{
  "service": "opencode-memory",
  "version": "0.3.0",
  "endpoints": [...]
}
```

## Setup

### Requirements
- Python 3.12+
- OpenAI API key

### Installation

```bash
cd service
pip install -e .
```

### Configuration

Set environment variables (or create a `.env` file in the `service/` directory):

```bash
# Required
MEMORY_OPENAI_API_KEY=sk-...

# Optional (defaults shown)
MEMORY_HOST=0.0.0.0
MEMORY_PORT=4097
MEMORY_DB_PATH=~/.opencode/memory.db
MEMORY_EMBEDDING_MODEL=text-embedding-3-small
MEMORY_EMBEDDING_DIMENSIONS=1536
MEMORY_EXTRACTION_MODEL=gpt-4o-mini
MEMORY_CONSOLIDATION_THRESHOLD=0.85
MEMORY_DEDUPE_THRESHOLD=0.92
```

### Running the Service

```bash
# Development mode (auto-reload)
memory-service

# Or with uvicorn directly
uvicorn memory_service.main:app --host 0.0.0.0 --port 4097 --reload
```

The service will:
- Create the SQLite database at `~/.opencode/memory.db`
- Load the sqlite-vec extension
- Start the FastAPI server on port 4097
- Generate OpenAPI docs at `http://localhost:4097/docs`

## Deployment

### macOS (launchd)

The service runs on a Mac Mini using launchd for automatic startup.

Create `~/Library/LaunchAgents/com.opencode.memory.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.opencode.memory</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/python</string>
        <string>-m</string>
        <string>uvicorn</string>
        <string>memory_service.main:app</string>
        <string>--host</string>
        <string>0.0.0.0</string>
        <string>--port</string>
        <string>4097</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/Users/minzi/Developer/opencode-memory/service</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>MEMORY_OPENAI_API_KEY</key>
        <string>sk-...</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/opencode-memory.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/opencode-memory.error.log</string>
</dict>
</plist>
```

Load the service:

```bash
launchctl load ~/Library/LaunchAgents/com.opencode.memory.plist
launchctl start com.opencode.memory
```

## Configuration Details

### Memory Types

| Type | Description | Use Case |
|------|-------------|----------|
| `preference` | User preferences | "Prefer TypeScript over JavaScript" |
| `convention` | Code conventions | "Use kebab-case for file names" |
| `constraint` | Hard requirements | "Must support OAuth2" |
| `decision` | Architectural decisions | "Use PostgreSQL for database" |
| `architecture` | System design | "Microservices with API gateway" |
| `error-solution` | Failed approaches | "Session cookies failed due to CORS" |
| `fact` | General knowledge | "API rate limit is 100 req/min" |
| `episode` | Session snapshots | Auto-generated from episodes |
| `failure` | Things that didn't work | "Approach X failed because Y" |
| `pattern` | Recurring patterns | "Always validate input at API boundary" |
| `working_context` | Current task context | "Working on auth implementation" |

### Scopes

| Scope | Description | Visibility |
|-------|-------------|------------|
| `global` | User-level preferences | All projects |
| `project` | Project-specific | Single project (requires `project_id`) |
| `session` | Session-specific | Single session (ephemeral) |

### Quality Thresholds

| Threshold | Value | Behavior |
|-----------|-------|----------|
| `dedupe_threshold` | 0.92 | Reject near-duplicates (similarity ≥ 0.92) |
| `consolidation_threshold` | 0.85 | Merge similar memories (0.85 ≤ similarity < 0.92) |
| `recall_threshold` | 0.3 (default) | Minimum similarity for search results |

### Scoring Formula

Combined score = 0.60 × similarity + 0.15 × recency + 0.10 × access_frequency + 0.15 × confidence

- **Similarity** — cosine similarity to query embedding
- **Recency** — exponential decay based on age (30-day half-life)
- **Access frequency** — logarithmic scaling of retrieval count
- **Confidence** — user-provided confidence score (0.0-1.0)

## Development

### Project Structure

```
service/
├── pyproject.toml              # Dependencies and metadata
├── memory_service/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app and routes
│   ├── config.py               # Settings (Pydantic BaseSettings)
│   ├── models.py               # Request/response models
│   ├── store.py                # SQLite + sqlite-vec storage
│   ├── embeddings.py           # OpenAI embedding client
│   └── extractor.py            # LLM-powered memory extraction
└── tests/                      # (future)
```

### Dependencies

- `fastapi>=0.128.0` — async web framework
- `uvicorn>=0.40.0` — ASGI server
- `sqlite-vec>=0.1.6` — vector similarity search extension
- `openai>=1.50.0` — OpenAI API client
- `pydantic>=2.7.0` — data validation
- `pydantic-settings>=2.0.0` — settings management

### Database Schema

**memories table:**
- `id` (TEXT PRIMARY KEY) — `mem_` prefix + 12-char hex
- `vec_id` (INTEGER) — rowid in vec_memories table
- `content` (TEXT) — the memory content
- `type` (TEXT) — memory type enum
- `scope` (TEXT) — global/project/session
- `project_id` (TEXT) — project identifier (nullable)
- `tags` (TEXT) — JSON array of tags
- `source` (TEXT) — where this came from (nullable)
- `confidence` (REAL) — 0.0-1.0
- `score` (REAL) — combined quality score
- `retrieval_count` (INTEGER) — access frequency
- `last_accessed_at` (REAL) — Unix timestamp
- `created_at` (REAL) — Unix timestamp
- `updated_at` (REAL) — Unix timestamp
- `metadata` (TEXT) — JSON object

**vec_memories table (virtual):**
- `rowid` (INTEGER PRIMARY KEY) — auto-generated
- `embedding` (float[1536]) — vector embedding

**episodes table:**
- `id` (TEXT PRIMARY KEY) — `ep_` prefix + 12-char hex
- `session_id` (TEXT) — session identifier
- `project_id` (TEXT) — project identifier
- `summary` (TEXT) — episode summary
- `data` (TEXT) — JSON object with todos, decisions, etc.
- `created_at` (REAL) — Unix timestamp
- `metadata` (TEXT) — JSON object

**working_state table:**
- `project_id` (TEXT PRIMARY KEY) — project identifier
- `data` (TEXT) — JSON object with state fields
- `updated_at` (REAL) — Unix timestamp

