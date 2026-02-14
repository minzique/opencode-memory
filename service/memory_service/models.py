"""Request/response models for the memory service API.

All models use Pydantic v2 for validation and auto-generated OpenAPI docs.
Designed to be agent-friendly: clear field names, good defaults, helpful descriptions.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    preference = "preference"
    convention = "convention"
    constraint = "constraint"
    decision = "decision"
    architecture = "architecture"
    error_solution = "error-solution"
    fact = "fact"
    episode = "episode"
    failure = "failure"
    pattern = "pattern"
    working_context = "working_context"


class MemoryScope(str, Enum):
    global_ = "global"
    project = "project"
    session = "session"

class RememberRequest(BaseModel):
    """Store a new memory. The service auto-generates embeddings for semantic search."""
    content: str = Field(..., description="The knowledge to remember. Be specific.")
    type: MemoryType = Field(MemoryType.fact, description="Category of this memory")
    scope: MemoryScope = Field(MemoryScope.global_, description="Visibility scope")
    project_id: str | None = Field(None, description="Project identifier (for project-scoped)")
    tags: list[str] = Field(default_factory=list, description="Tags for filtering")
    source: str | None = Field(None, description="Where this memory came from (session ID, agent, etc.)")
    confidence: float = Field(0.8, ge=0.0, le=1.0, description="How confident we are (1.0 = certain)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary extra data")


class RecallQuery(BaseModel):
    """Search for memories. Supports semantic search, filtering, and pagination."""
    query: str = Field(..., description="What you want to recall — natural language")
    limit: int = Field(10, ge=1, le=100, description="Max results to return")
    threshold: float = Field(0.3, ge=0.0, le=1.0, description="Min similarity score (0-1)")
    types: list[MemoryType] | None = Field(None, description="Filter by memory type")
    scope: MemoryScope | None = Field(None, description="Filter by scope")
    project_id: str | None = Field(None, description="Filter by project")
    tags: list[str] | None = Field(None, description="Filter by tags (any match)")


class EpisodeRequest(BaseModel):
    """Save a session snapshot — structured capture of what happened."""
    session_id: str = Field(..., description="The session this episode is from")
    project_id: str = Field("global", description="Project identifier")
    summary: str = Field(..., description="Brief description of what happened")
    todos: list[TodoItem] = Field(default_factory=list)
    decisions: list[DecisionItem] = Field(default_factory=list)
    constraints: list[ConstraintItem] = Field(default_factory=list)
    failed_approaches: list[FailedApproachItem] = Field(default_factory=list)
    explored_files: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TodoItem(BaseModel):
    content: str
    status: str = "pending"
    priority: str = "medium"


class DecisionItem(BaseModel):
    content: str
    context: str = ""
    confidence: float = 0.8


class ConstraintItem(BaseModel):
    content: str
    type: str = "must"
    source: str = "inferred"


class FailedApproachItem(BaseModel):
    approach: str
    error: str = ""
    context: str = ""


class MemoryRecord(BaseModel):
    """A stored memory as returned by the API."""
    id: str
    content: str
    type: MemoryType
    scope: MemoryScope
    project_id: str | None = None
    tags: list[str] = []
    source: str | None = None
    confidence: float = 1.0
    score: float = 1.0
    retrieval_count: int = 0
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = {}


class RecallResult(BaseModel):
    """A search result with similarity score."""
    memory: MemoryRecord
    similarity: float = Field(description="Cosine similarity to query (0-1)")


class RecallResponse(BaseModel):
    """Response from /recall endpoint."""
    results: list[RecallResult]
    query: str
    total: int


class EpisodeRecord(BaseModel):
    """A stored session episode."""
    id: str
    session_id: str
    project_id: str
    summary: str
    todos: list[TodoItem] = []
    decisions: list[DecisionItem] = []
    constraints: list[ConstraintItem] = []
    failed_approaches: list[FailedApproachItem] = []
    explored_files: list[str] = []
    created_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = {}


class WorkingState(BaseModel):
    objective: str = Field("", description="Current high-level goal")
    approach: str = Field("", description="Current approach being taken")
    progress: str = Field("", description="What's done so far")
    files_touched: list[str] = Field(default_factory=list, description="Files modified in current task")
    tried_and_failed: list[str] = Field(default_factory=list, description="Approaches that didn't work")
    next_steps: list[str] = Field(default_factory=list, description="What to do next")
    blockers: list[str] = Field(default_factory=list, description="Current blockers")
    open_questions: list[str] = Field(default_factory=list, description="Unresolved questions")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary extra data")


class BootstrapRequest(BaseModel):
    project_id: str | None = Field(None, description="Project to load context for")
    cwd: str | None = Field(None, description="Current working directory (auto-detects project)")
    include_state: bool = Field(True, description="Include working state")
    include_memories: bool = Field(True, description="Include relevant memories")
    include_episodes: bool = Field(True, description="Include recent episodes")
    include_cross_project: bool = Field(True, description="Include global memories from other projects")
    memory_limit: int = Field(15, description="Max memories to include")
    episode_limit: int = Field(3, description="Max recent episodes to include")


class CrossProjectMemory(BaseModel):
    memory: MemoryRecord
    origin_project: str | None = None


class BootstrapResponse(BaseModel):
    project_id: str | None = None
    state: WorkingState | None = None
    memories: list[MemoryRecord] = Field(default_factory=list)
    cross_project: list[CrossProjectMemory] = Field(default_factory=list)
    recent_episodes: list[EpisodeRecord] = Field(default_factory=list)
    constraints: list[MemoryRecord] = Field(default_factory=list)
    failed_approaches: list[MemoryRecord] = Field(default_factory=list)


class ServiceStatus(BaseModel):
    healthy: bool = True
    version: str = "0.1.0"
    memory_count: int = 0
    episode_count: int = 0
    state_count: int = 0
    db_size_bytes: int = 0
    embedding_provider: str = "openai"
    uptime_seconds: float = 0.0


# ------------------------------------------------------------------
# Extraction models
# ------------------------------------------------------------------


class ExtractRequest(BaseModel):
    """Submit raw text for LLM-powered structured memory extraction."""

    text: str = Field(..., description="Raw text to extract memories from")
    context: str | None = Field(None, description="Optional context (project, task, etc.)")
    source: str | None = Field(None, description="Where this text came from")
    project_id: str | None = Field(None, description="Project identifier for scoping extracted memories")
    scope: str | None = Field(None, description="Scope override (global/project/session)")


class ExtractedMemory(BaseModel):
    """A single memory extracted by the LLM."""

    content: str = Field(..., description="The extracted knowledge")
    type: MemoryType = Field(..., description="Category of this memory")
    confidence: float = Field(0.8, ge=0.0, le=1.0, description="Extraction confidence")
    tags: list[str] = Field(default_factory=list, description="Relevant tags")


class ExtractResponse(BaseModel):
    extracted: int = Field(0, description="Number of memories extracted and stored")
    memory_ids: list[str] = Field(default_factory=list, description="IDs of created memories")


class PublishRequest(BaseModel):
    title: str = Field(..., description="Post title")
    description: str = Field(..., description="One-line summary")
    body: str = Field(..., description="Markdown body content (no frontmatter)")
    author: str = Field("Lume", description="Post author")
    slug: str | None = Field(None, description="URL slug (auto-generated from title if omitted)")
    push: bool = Field(True, description="Git push after commit to trigger deploy")
