"""SQLite + sqlite-vec storage backend.

Uses sync sqlite3 with check_same_thread=False (sqlite-vec requires sync driver).
Vector search via vec0 virtual table with cosine distance metric.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from uuid import uuid4

import sqlite_vec
from sqlite_vec import serialize_float32

logger = logging.getLogger(__name__)

_SCHEMA = """
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

CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope);
CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project_id);
CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);
CREATE INDEX IF NOT EXISTS idx_episodes_project ON episodes(project_id);
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

        self.db.executescript(_SCHEMA)
        self.db.commit()

        self._next_vec_id = self._get_max_vec_id() + 1
        logger.info("MemoryStore initialized: %s", db_path)

    def _get_max_vec_id(self) -> int:
        try:
            row = self.db.execute("SELECT MAX(rowid) FROM vec_memories").fetchone()
            return row[0] if row[0] is not None else 0
        except Exception:
            return 0

    def _alloc_vec_id(self) -> int:
        vid = self._next_vec_id
        self._next_vec_id += 1
        return vid

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
        vec_id = self._alloc_vec_id()

        self.db.execute(
            """
            INSERT INTO memories
                (id, content, type, scope, project_id, tags, source,
                 confidence, score, retrieval_count,
                 created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1.0, 0, ?, ?, ?)
            """,
            (
                memory_id,
                content,
                type,
                scope,
                project_id,
                json.dumps(tags),
                source,
                confidence,
                now,
                now,
                json.dumps(metadata | {"vec_id": vec_id}),
            ),
        )

        self.db.execute(
            "INSERT INTO vec_memories(rowid, embedding) VALUES (?, ?)",
            [vec_id, serialize_float32(embedding)],
        )

        self.db.commit()
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
            f"SELECT * FROM memories WHERE json_extract(metadata, '$.vec_id') IN ({placeholders})",
            vec_ids,
        ).fetchall()

        results: list[tuple[dict, float]] = []
        for mem in all_memories:
            mem_dict = dict(mem)
            mem_meta = json.loads(mem_dict["metadata"])
            vec_id = mem_meta.get("vec_id")
            if vec_id is None:
                continue

            distance = vec_distances.get(vec_id)
            if distance is None:
                continue

            # cosine distance ∈ [0, 2] → similarity = 1 - distance/2
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
            results.append((mem_dict, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

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

        vec_id = mem["metadata"].get("vec_id")
        self.db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        if vec_id is not None:
            try:
                self.db.execute("DELETE FROM vec_memories WHERE rowid = ?", (vec_id,))
            except Exception:
                logger.warning("Failed to delete vec row %s for memory %s", vec_id, memory_id)
        self.db.commit()
        return True

    def update_retrieval_count(self, memory_id: str) -> None:
        self.db.execute(
            "UPDATE memories SET retrieval_count = retrieval_count + 1, "
            "updated_at = ? WHERE id = ?",
            (time.time(), memory_id),
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
                "SELECT id FROM memories WHERE json_extract(metadata, '$.vec_id') = ?",
                (vec_id,),
            ).fetchone()
            if row:
                return row[0]

        return None

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

    def list_episodes(self, project_id: str | None = None, limit: int = 20) -> list[dict]:
        if project_id:
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
        try:
            db_size = os.path.getsize(self.db_path)
        except OSError:
            db_size = 0
        return {
            "memory_count": memory_count,
            "episode_count": episode_count,
            "db_size_bytes": db_size,
        }

    def decay_scores(self, factor: float, min_score: float) -> None:
        self.db.execute(
            "UPDATE memories SET score = score * ?, updated_at = ?",
            (factor, time.time()),
        )
        to_delete = self.db.execute(
            "SELECT id FROM memories WHERE score < ?", (min_score,)
        ).fetchall()
        for row in to_delete:
            self.delete_memory(row[0])
        self.db.commit()
        logger.info("Decayed scores by %.2f, deleted %d low-score memories", factor, len(to_delete))

    def close(self) -> None:
        self.db.close()
        logger.info("MemoryStore closed")

    @staticmethod
    def generate_memory_id() -> str:
        return f"mem_{uuid4().hex[:12]}"

    @staticmethod
    def generate_episode_id() -> str:
        return f"ep_{uuid4().hex[:12]}"
