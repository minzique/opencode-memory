"""SQLite + sqlite-vec storage backend.

Uses sync sqlite3 with check_same_thread=False (sqlite-vec requires sync driver).
Vector search via vec0 virtual table with cosine distance metric.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import time
from uuid import uuid4

from pathlib import Path, PurePosixPath

import sqlite_vec
from sqlite_vec import serialize_float32

logger = logging.getLogger(__name__)

# Stop walking at these paths (never include them in the chain).
# Home dir is added dynamically so sub-project chains don't inherit
# every memory from every project the user has ever worked on.
_STOP_PATHS = {"", "/", "/Users", "/home", str(Path.home())}

# Maximum ancestor depth.  Even if we haven't hit a stop path, cap
# the chain to avoid extremely deep (and noisy) lookups.
_MAX_ANCESTOR_DEPTH = 4


def ancestor_chain(project_id: str | None, *, max_depth: int = _MAX_ANCESTOR_DEPTH) -> list[str]:
    """Return [exact, parent, grandparent, ...] up to *max_depth* entries.

    Stops early at home directory, /Users, /home, or filesystem root.
    """
    if not project_id or not project_id.startswith("/"):
        return [project_id] if project_id else []
    parts: list[str] = []
    p = PurePosixPath(project_id)
    while str(p) not in _STOP_PATHS and len(parts) < max_depth:
        parts.append(str(p))
        parent = str(p.parent)
        if parent == str(p):
            break
        p = PurePosixPath(parent)
    return parts

_SCHEMA_TABLES = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    type TEXT NOT NULL DEFAULT 'fact',
    scope TEXT NOT NULL DEFAULT 'global',
    project_id TEXT,
    tags TEXT DEFAULT '[]',
    source TEXT,
    confidence REAL DEFAULT 1.0,
    score REAL DEFAULT 1.0,
    retrieval_count INTEGER DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    metadata TEXT DEFAULT '{}'
);

CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
    embedding float[1536] distance_metric=cosine
);

CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    project_id TEXT NOT NULL DEFAULT 'global',
    summary TEXT NOT NULL,
    data TEXT NOT NULL,
    created_at REAL NOT NULL,
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS working_state (
    project_id TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS todos (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    project_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    priority TEXT NOT NULL DEFAULT 'medium',
    tags TEXT DEFAULT '[]',
    parent_id TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    completed_at REAL,
    metadata TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope);
CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project_id);
CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);
CREATE INDEX IF NOT EXISTS idx_episodes_project ON episodes(project_id);
CREATE INDEX IF NOT EXISTS idx_todos_project ON todos(project_id);
CREATE INDEX IF NOT EXISTS idx_todos_status ON todos(status);
"""

_SCHEMA_POST_MIGRATE = """
CREATE INDEX IF NOT EXISTS idx_memories_vec_id ON memories(vec_id);
"""


class MemoryStore:

    def __init__(self, db_path: str) -> None:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self.db_path = db_path
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row

        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)

        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("PRAGMA cache_size=-64000")
        self.db.execute("PRAGMA temp_store=MEMORY")

        self.db.executescript(_SCHEMA_TABLES)
        self.db.commit()

        self._migrate_schema()

        self.db.executescript(_SCHEMA_POST_MIGRATE)
        self.db.commit()
        logger.info("MemoryStore initialized: %s", db_path)

    # ------------------------------------------------------------------
    # Schema migration
    # ------------------------------------------------------------------

    def _migrate_schema(self) -> None:
        """Migrate existing data: move vec_id from metadata JSON to column."""
        cols = [row[1] for row in self.db.execute("PRAGMA table_info(memories)").fetchall()]
        if "vec_id" not in cols:
            self.db.execute("ALTER TABLE memories ADD COLUMN vec_id INTEGER")
            self.db.execute("ALTER TABLE memories ADD COLUMN last_accessed_at REAL")
            # Migrate existing data: extract vec_id from metadata JSON
            rows = self.db.execute("SELECT id, metadata FROM memories").fetchall()
            for row in rows:
                meta = json.loads(row[1])
                vid = meta.pop("vec_id", None)
                if vid is not None:
                    self.db.execute(
                        "UPDATE memories SET vec_id = ?, metadata = ? WHERE id = ?",
                        (vid, json.dumps(meta), row[0]),
                    )
            self.db.commit()
            logger.info("Migrated vec_id from metadata to column")
        elif "last_accessed_at" not in cols:
            self.db.execute("ALTER TABLE memories ADD COLUMN last_accessed_at REAL")
            self.db.commit()
            logger.info("Added last_accessed_at column")

    def add_memory(
        self,
        memory_id: str,
        content: str,
        embedding: list[float],
        type: str,
        scope: str,
        project_id: str | None,
        tags: list[str],
        source: str | None,
        confidence: float,
        metadata: dict,
    ) -> str:
        now = time.time()

        try:
            self.db.execute("BEGIN IMMEDIATE")

            cursor = self.db.execute(
                "INSERT INTO vec_memories(embedding) VALUES (?)",
                [serialize_float32(embedding)],
            )
            vec_id = cursor.lastrowid

            self.db.execute(
                """
                INSERT INTO memories
                    (id, vec_id, content, type, scope, project_id, tags, source,
                     confidence, score, retrieval_count, last_accessed_at,
                     created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1.0, 0, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    vec_id,
                    content,
                    type,
                    scope,
                    project_id,
                    json.dumps(tags),
                    source,
                    confidence,
                    now,
                    now,
                    now,
                    json.dumps(metadata),
                ),
            )

            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise

        logger.debug("Stored memory %s (vec_id=%d)", memory_id, vec_id)
        return memory_id

    def search_memories(
        self,
        query_embedding: list[float],
        limit: int = 10,
        threshold: float = 0.3,
        types: list[str] | None = None,
        scope: str | None = None,
        project_id: str | None = None,
        tags: list[str] | None = None,
    ) -> list[tuple[dict, float]]:
        fetch_limit = limit * 5

        rows = self.db.execute(
            "SELECT rowid, distance FROM vec_memories "
            "WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            [serialize_float32(query_embedding), fetch_limit],
        ).fetchall()

        if not rows:
            return []

        vec_distances: dict[int, float] = {row[0]: row[1] for row in rows}
        vec_ids = list(vec_distances.keys())

        placeholders = ",".join("?" for _ in vec_ids)
        all_memories = self.db.execute(
            f"SELECT * FROM memories WHERE vec_id IN ({placeholders})",
            vec_ids,
        ).fetchall()

        scored: list[tuple[dict, float, float]] = []
        for mem in all_memories:
            mem_dict = dict(mem)
            vec_id = mem_dict.get("vec_id")
            if vec_id is None:
                continue

            distance = vec_distances.get(vec_id)
            if distance is None:
                continue

            similarity = 1.0 - (distance / 2.0)

            if similarity < threshold:
                continue

            if types and mem_dict["type"] not in types:
                continue
            if scope and mem_dict["scope"] != scope:
                continue
            if project_id and mem_dict.get("project_id") != project_id:
                continue
            if tags:
                mem_tags = json.loads(mem_dict["tags"])
                if not any(t in mem_tags for t in tags):
                    continue

            mem_dict["tags"] = json.loads(mem_dict["tags"])
            mem_dict["metadata"] = json.loads(mem_dict["metadata"])

            combined = self._compute_combined_score(similarity, mem_dict)
            scored.append((mem_dict, similarity, combined))

        scored.sort(key=lambda x: x[2], reverse=True)
        return [(s[0], s[1]) for s in scored[:limit]]

    def _compute_combined_score(self, similarity: float, mem_dict: dict) -> float:
        now = time.time()

        created_days = (now - mem_dict["created_at"]) / 86400
        creation_recency = math.exp(-created_days / 30)

        last_access = mem_dict.get("last_accessed_at") or mem_dict["created_at"]
        access_days = (now - last_access) / 86400
        access_recency = math.exp(-access_days / 14)

        access_count = mem_dict.get("retrieval_count", 0)
        access_freq = min(1.0, math.log(access_count + 1) / math.log(50))

        confidence = mem_dict.get("confidence", 1.0)

        return (
            0.50 * similarity
            + 0.10 * creation_recency
            + 0.15 * access_recency
            + 0.10 * access_freq
            + 0.15 * confidence
        )

    def get_memory(self, memory_id: str) -> dict | None:
        row = self.db.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            return None
        mem = dict(row)
        mem["tags"] = json.loads(mem["tags"])
        mem["metadata"] = json.loads(mem["metadata"])
        return mem

    def delete_memory(self, memory_id: str) -> bool:
        mem = self.get_memory(memory_id)
        if mem is None:
            return False

        vec_id = mem.get("vec_id")
        self.db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        if vec_id is not None:
            try:
                self.db.execute("DELETE FROM vec_memories WHERE rowid = ?", (vec_id,))
            except Exception:
                logger.warning("Failed to delete vec row %s for memory %s", vec_id, memory_id)
        self.db.commit()
        return True

    def update_retrieval_count(self, memory_id: str) -> None:
        now = time.time()
        self.db.execute(
            "UPDATE memories SET retrieval_count = retrieval_count + 1, "
            "last_accessed_at = ?, updated_at = ? WHERE id = ?",
            (now, now, memory_id),
        )
        self.db.commit()

    def check_duplicate(self, embedding: list[float], threshold: float) -> str | None:
        rows = self.db.execute(
            "SELECT rowid, distance FROM vec_memories "
            "WHERE embedding MATCH ? ORDER BY distance LIMIT 1",
            [serialize_float32(embedding)],
        ).fetchall()

        if not rows:
            return None

        vec_id, distance = rows[0]
        # cosine distance ∈ [0, 2] → similarity = 1 - distance/2
        similarity = 1.0 - (distance / 2.0)

        if similarity >= threshold:
            row = self.db.execute(
                "SELECT id FROM memories WHERE vec_id = ?",
                (vec_id,),
            ).fetchone()
            if row:
                return row[0]

        return None

    def check_near_duplicate(
        self, embedding: list[float], consolidation_threshold: float, dedupe_threshold: float
    ) -> tuple[str | None, float]:
        """Return (memory_id, similarity) of the closest match, or (None, 0.0)."""
        rows = self.db.execute(
            "SELECT rowid, distance FROM vec_memories "
            "WHERE embedding MATCH ? ORDER BY distance LIMIT 1",
            [serialize_float32(embedding)],
        ).fetchall()

        if not rows:
            return None, 0.0

        vec_id, distance = rows[0]
        similarity = 1.0 - (distance / 2.0)

        if similarity >= consolidation_threshold:
            row = self.db.execute(
                "SELECT id FROM memories WHERE vec_id = ?", (vec_id,)
            ).fetchone()
            if row:
                return row[0], similarity

        return None, similarity

    def consolidate_memory(
        self,
        existing_id: str,
        new_content: str,
        new_tags: list[str],
        new_confidence: float,
        new_metadata: dict | None = None,
    ) -> None:
        """Merge new memory data into an existing near-duplicate."""
        mem = self.get_memory(existing_id)
        if mem is None:
            return

        merged_content = f"{mem['content']}\n---\n{new_content}"
        merged_confidence = max(mem.get("confidence", 1.0), new_confidence)

        existing_tags: list[str] = mem.get("tags", [])
        merged_tags = list(dict.fromkeys(existing_tags + new_tags))

        existing_meta: dict = mem.get("metadata", {})
        if new_metadata:
            existing_meta.update(new_metadata)

        now = time.time()
        self.db.execute(
            "UPDATE memories SET content = ?, confidence = ?, tags = ?, "
            "metadata = ?, updated_at = ? WHERE id = ?",
            (
                merged_content,
                merged_confidence,
                json.dumps(merged_tags),
                json.dumps(existing_meta),
                now,
                existing_id,
            ),
        )
        self.db.commit()
        logger.debug("Consolidated memory %s", existing_id)

    def get_memories_by_type(
        self,
        types: list[str],
        project_id: str | None = None,
        project_ids: list[str] | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Fetch memories by type.

        If *project_ids* is given (e.g. from ancestor_chain), match any of
        those project paths.  Otherwise fall back to exact project_id match.
        """
        type_ph = ",".join("?" for _ in types)
        query = f"SELECT * FROM memories WHERE type IN ({type_ph}) "
        params: list = list(types)

        if project_ids:
            pid_ph = ",".join("?" for _ in project_ids)
            query += f"AND (project_id IN ({pid_ph}) OR project_id IS NULL) "
            params.extend(project_ids)
        elif project_id:
            query += "AND (project_id = ? OR project_id IS NULL) "
            params.append(project_id)

        query += "ORDER BY score DESC, retrieval_count DESC LIMIT ?"
        params.append(limit)
        rows = self.db.execute(query, params).fetchall()
        results = []
        for row in rows:
            mem = dict(row)
            mem["tags"] = json.loads(mem["tags"])
            mem["metadata"] = json.loads(mem["metadata"])
            results.append(mem)
        return results

    def list_memories(self, limit: int = 50, offset: int = 0) -> list[dict]:
        rows = self.db.execute(
            "SELECT * FROM memories ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        results = []
        for row in rows:
            mem = dict(row)
            mem["tags"] = json.loads(mem["tags"])
            mem["metadata"] = json.loads(mem["metadata"])
            results.append(mem)
        return results

    # ------------------------------------------------------------------
    # Working state
    # ------------------------------------------------------------------

    def set_state(self, project_id: str, data: dict) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO working_state (project_id, data, updated_at) VALUES (?, ?, ?)",
            (project_id, json.dumps(data), time.time()),
        )
        self.db.commit()

    def get_state(self, project_id: str) -> dict | None:
        row = self.db.execute(
            "SELECT data, updated_at FROM working_state WHERE project_id = ?", (project_id,)
        ).fetchone()
        if row is None:
            return None
        return {"data": json.loads(row[0]), "updated_at": row[1]}

    def get_state_hierarchical(self, project_ids: list[str]) -> dict | None:
        """Walk ancestor chain and return the first (most specific) working state found."""
        for pid in project_ids:
            state = self.get_state(pid)
            if state is not None:
                return state
        return None

    def list_states(self) -> list[dict]:
        rows = self.db.execute(
            "SELECT project_id, data, updated_at FROM working_state ORDER BY updated_at DESC"
        ).fetchall()
        results = []
        for row in rows:
            results.append({
                "project_id": row[0],
                "data": json.loads(row[1]),
                "updated_at": row[2]
            })
        return results

    def delete_state(self, project_id: str) -> bool:
        cursor = self.db.execute("DELETE FROM working_state WHERE project_id = ?", (project_id,))
        self.db.commit()
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Episodes
    # ------------------------------------------------------------------

    def add_episode(
        self,
        episode_id: str,
        session_id: str,
        project_id: str,
        summary: str,
        data: dict,
        metadata: dict,
    ) -> str:
        now = time.time()
        self.db.execute(
            """
            INSERT INTO episodes (id, session_id, project_id, summary, data, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                episode_id, session_id, project_id, summary,
                json.dumps(data), now, json.dumps(metadata),
            ),
        )
        self.db.commit()
        logger.debug("Stored episode %s for session %s", episode_id, session_id)
        return episode_id

    def list_episodes(
        self,
        project_id: str | None = None,
        project_ids: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """List episodes. If project_ids given, match any in the chain."""
        if project_ids:
            pid_ph = ",".join("?" for _ in project_ids)
            rows = self.db.execute(
                f"SELECT * FROM episodes WHERE project_id IN ({pid_ph}) "
                "ORDER BY created_at DESC LIMIT ?",
                [*project_ids, limit],
            ).fetchall()
        elif project_id:
            rows = self.db.execute(
                "SELECT * FROM episodes WHERE project_id = ? ORDER BY created_at DESC LIMIT ?",
                (project_id, limit),
            ).fetchall()
        else:
            rows = self.db.execute(
                "SELECT * FROM episodes ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

        results = []
        for row in rows:
            ep = dict(row)
            ep["data"] = json.loads(ep["data"])
            ep["metadata"] = json.loads(ep["metadata"])
            results.append(ep)
        return results

    def get_stats(self) -> dict:
        memory_count = self.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        episode_count = self.db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        state_count = self.db.execute("SELECT COUNT(*) FROM working_state").fetchone()[0]
        try:
            db_size = os.path.getsize(self.db_path)
        except OSError:
            db_size = 0
        return {
            "memory_count": memory_count,
            "episode_count": episode_count,
            "state_count": state_count,
            "db_size_bytes": db_size,
        }

    def get_global_memories(
        self,
        allowed_types: list[str],
        exclude_project_id: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        placeholders = ",".join("?" for _ in allowed_types)
        query = f"SELECT * FROM memories WHERE scope = 'global' AND type IN ({placeholders}) "
        params: list = list(allowed_types)
        if exclude_project_id:
            query += "AND (project_id != ? OR project_id IS NULL) "
            params.append(exclude_project_id)
        query += "ORDER BY score DESC, retrieval_count DESC LIMIT ?"
        params.append(limit)
        rows = self.db.execute(query, params).fetchall()
        results = []
        for row in rows:
            mem = dict(row)
            mem["tags"] = json.loads(mem["tags"])
            mem["metadata"] = json.loads(mem["metadata"])
            results.append(mem)
        return results

    def cleanup_memories(
        self,
        sources: list[str] | None = None,
        tags: list[str] | None = None,
        types: list[str] | None = None,
        max_confidence: float | None = None,
        content_prefixes: list[str] | None = None,
        dry_run: bool = True,
    ) -> dict:
        """Delete memories matching configurable purge rules.

        Args:
            sources: Delete memories with source IN this list (e.g. ["tool:bash", "tool:read"])
            tags: Delete memories containing ANY of these tags
            types: Delete memories with type IN this list
            max_confidence: Delete memories with confidence <= this value
            content_prefixes: Delete memories whose content starts with any of these prefixes
            dry_run: If True, only count — don't actually delete
        """
        # Build conditions — all are ANDed if multiple are provided,
        # but each individual filter matches broadly (OR within filter)
        conditions: list[str] = []
        params: list = []

        if sources:
            placeholders = ",".join("?" for _ in sources)
            conditions.append(f"source IN ({placeholders})")
            params.extend(sources)

        if tags:
            # tags column is JSON array — check if any tag is present
            tag_conditions = []
            for tag in tags:
                tag_conditions.append("tags LIKE ?")
                params.append(f'%"{tag}"%')
            conditions.append(f"({' OR '.join(tag_conditions)})")

        if types:
            placeholders = ",".join("?" for _ in types)
            conditions.append(f"type IN ({placeholders})")
            params.extend(types)

        if max_confidence is not None:
            conditions.append("confidence <= ?")
            params.append(max_confidence)

        if content_prefixes:
            prefix_conditions = []
            for prefix in content_prefixes:
                prefix_conditions.append("content LIKE ?")
                params.append(f"{prefix}%")
            conditions.append(f"({' OR '.join(prefix_conditions)})")

        if not conditions:
            return {"deleted_count": 0, "matched_count": 0, "dry_run": dry_run, "error": "No filters specified"}

        where_clause = " AND ".join(conditions)

        # Count first
        count = self.db.execute(
            f"SELECT COUNT(*) FROM memories WHERE {where_clause}", params
        ).fetchone()[0]

        if dry_run:
            # Return sample of what would be deleted
            sample_rows = self.db.execute(
                f"SELECT id, content, type, source, confidence FROM memories WHERE {where_clause} LIMIT 10",
                params,
            ).fetchall()
            samples = [
                {"id": row[0], "content": row[1][:100], "type": row[2], "source": row[3], "confidence": row[4]}
                for row in sample_rows
            ]
            return {"matched_count": count, "deleted_count": 0, "dry_run": True, "samples": samples}

        # Actually delete
        to_delete = self.db.execute(
            f"SELECT id FROM memories WHERE {where_clause}", params
        ).fetchall()
        deleted_ids = [row[0] for row in to_delete]

        for mid in deleted_ids:
            self.delete_memory(mid)

        logger.info("Cleanup: deleted %d memories (filters: %s)", len(deleted_ids), where_clause)
        return {"deleted_count": len(deleted_ids), "matched_count": count, "dry_run": False}

    def decay_scores(self, factor: float, min_score: float) -> dict:
        now = time.time()
        self.db.execute(
            "UPDATE memories SET score = score * ?, updated_at = ?",
            (factor, now),
        )
        to_delete = self.db.execute(
            "SELECT id FROM memories WHERE score < ?", (min_score,)
        ).fetchall()
        deleted_ids = [row[0] for row in to_delete]
        for mid in deleted_ids:
            self.delete_memory(mid)
        self.db.commit()
        logger.info("Decayed scores by %.2f, deleted %d low-score memories", factor, len(deleted_ids))
        return {"decayed_at": now, "deleted_count": len(deleted_ids), "deleted_ids": deleted_ids}

    def close(self) -> None:
        self.db.close()
        logger.info("MemoryStore closed")

    @staticmethod
    def generate_memory_id() -> str:
        return f"mem_{uuid4().hex[:12]}"

    @staticmethod
    def generate_episode_id() -> str:
        return f"ep_{uuid4().hex[:12]}"

    @staticmethod
    def generate_todo_id() -> str:
        return f"todo_{uuid4().hex[:12]}"

    # ------------------------------------------------------------------
    # Persistent Todos
    # ------------------------------------------------------------------

    def _parse_todo_row(self, row: sqlite3.Row) -> dict:
        """Convert a raw DB row into a dict with parsed JSON fields."""
        todo = dict(row)
        todo["tags"] = json.loads(todo.get("tags") or "[]")
        todo["metadata"] = json.loads(todo.get("metadata") or "{}")
        return todo

    def add_todo(
        self,
        todo_id: str,
        content: str,
        project_id: str | None = None,
        status: str = "pending",
        priority: str = "medium",
        tags: list[str] | None = None,
        parent_id: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        now = time.time()
        self.db.execute(
            """
            INSERT INTO todos (id, content, project_id, status, priority, tags,
                              parent_id, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                todo_id, content, project_id, status, priority,
                json.dumps(tags or []), parent_id, now, now,
                json.dumps(metadata or {}),
            ),
        )
        self.db.commit()
        logger.debug("Created todo %s", todo_id)
        return todo_id

    def get_todo(self, todo_id: str) -> dict | None:
        row = self.db.execute("SELECT * FROM todos WHERE id = ?", (todo_id,)).fetchone()
        if row is None:
            return None
        return self._parse_todo_row(row)

    def list_todos(
        self,
        project_id: str | None = None,
        status: str | None = None,
        priority: str | None = None,
        include_completed: bool = False,
        limit: int = 50,
    ) -> list[dict]:
        query = "SELECT * FROM todos WHERE 1=1 "
        params: list = []

        if project_id:
            query += "AND project_id = ? "
            params.append(project_id)
        if status:
            query += "AND status = ? "
            params.append(status)
        if priority:
            query += "AND priority = ? "
            params.append(priority)
        if not include_completed:
            query += "AND status NOT IN ('completed', 'cancelled') "

        # Priority ordering: high > medium > low, then by creation
        query += (
            "ORDER BY "
            "CASE priority WHEN 'high' THEN 0 WHEN 'medium' THEN 1 WHEN 'low' THEN 2 END, "
            "created_at ASC "
            "LIMIT ?"
        )
        params.append(limit)

        rows = self.db.execute(query, params).fetchall()
        return [self._parse_todo_row(row) for row in rows]

    def update_todo(self, todo_id: str, updates: dict) -> dict | None:
        """Update specific fields of a todo. Returns the updated todo or None."""
        existing = self.get_todo(todo_id)
        if existing is None:
            return None

        now = time.time()
        set_parts: list[str] = ["updated_at = ?"]
        params: list = [now]

        if "content" in updates and updates["content"] is not None:
            set_parts.append("content = ?")
            params.append(updates["content"])
        if "status" in updates and updates["status"] is not None:
            set_parts.append("status = ?")
            params.append(updates["status"])
            # Auto-set completed_at
            if updates["status"] in ("completed", "cancelled"):
                set_parts.append("completed_at = ?")
                params.append(now)
            elif existing.get("completed_at"):
                # Re-opening: clear completed_at
                set_parts.append("completed_at = NULL")
        if "priority" in updates and updates["priority"] is not None:
            set_parts.append("priority = ?")
            params.append(updates["priority"])
        if "tags" in updates and updates["tags"] is not None:
            set_parts.append("tags = ?")
            params.append(json.dumps(updates["tags"]))
        if "metadata" in updates and updates["metadata"] is not None:
            set_parts.append("metadata = ?")
            params.append(json.dumps(updates["metadata"]))

        params.append(todo_id)
        self.db.execute(
            f"UPDATE todos SET {', '.join(set_parts)} WHERE id = ?",
            params,
        )
        self.db.commit()
        return self.get_todo(todo_id)

    def delete_todo(self, todo_id: str) -> bool:
        cursor = self.db.execute("DELETE FROM todos WHERE id = ?", (todo_id,))
        self.db.commit()
        return cursor.rowcount > 0
