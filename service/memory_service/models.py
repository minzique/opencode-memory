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
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="How confident we are (1.0 = certain)")
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


class ServiceStatus(BaseModel):
    """Service health and statistics."""
    healthy: bool = True
    version: str = "0.1.0"
    memory_count: int = 0
    episode_count: int = 0
    db_size_bytes: int = 0
    embedding_provider: str = "openai"
    uptime_seconds: float = 0.0
