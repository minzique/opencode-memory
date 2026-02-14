from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Response

from memory_service.config import settings
from memory_service.embeddings import EmbeddingClient
from memory_service.extractor import extract_memories
from memory_service.models import (
    BootstrapRequest,
    BootstrapResponse,
    CrossProjectMemory,
    EpisodeRecord,
    EpisodeRequest,
    ExtractRequest,
    ExtractResponse,
    MemoryRecord,
    PublishRequest,
    RecallQuery,
    RecallResponse,
    RecallResult,
    RememberRequest,
    ServiceStatus,
    WorkingState,
)
from memory_service.store import MemoryStore
from memory_service.dashboard import router as dashboard_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_decay_task: asyncio.Task | None = None


async def _decay_loop(store: MemoryStore) -> None:
    interval = settings.decay_interval_hours * 3600
    while True:
        await asyncio.sleep(interval)
        try:
            result = store.decay_scores(settings.decay_factor, settings.min_score)
            logger.info("Scheduled decay: deleted %d memories", result["deleted_count"])
        except Exception:
            logger.exception("Scheduled decay failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _decay_task
    app.state.store = MemoryStore(settings.db_path)
    app.state.embeddings = EmbeddingClient(
        settings.openai_api_key, settings.embedding_model, settings.embedding_dimensions
    )
    app.state.start_time = time.time()
    app.state.last_decay_at = None

    if settings.decay_interval_hours > 0:
        _decay_task = asyncio.create_task(_decay_loop(app.state.store))
        logger.info("Decay scheduler started (every %.1fh)", settings.decay_interval_hours)

    logger.info("Memory service started on %s:%d", settings.host, settings.port)
    yield

    if _decay_task:
        _decay_task.cancel()
        try:
            await _decay_task
        except asyncio.CancelledError:
            pass

    app.state.store.close()
    logger.info("Memory service stopped")


app = FastAPI(
    title="OpenCode Memory Service",
    description="Persistent semantic memory for AI agents",
    version="0.4.0",
    lifespan=lifespan,
)

app.include_router(dashboard_router)


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "service": "opencode-memory",
        "version": "0.4.0",
        "endpoints": [
            {"method": "POST", "path": "/remember", "description": "Store a memory"},
            {"method": "POST", "path": "/extract", "description": "Extract memories from text via LLM"},
            {"method": "POST", "path": "/recall", "description": "Search memories"},
            {"method": "GET", "path": "/recall/{memory_id}", "description": "Get specific memory"},
            {"method": "DELETE", "path": "/forget/{memory_id}", "description": "Delete a memory"},
            {"method": "POST", "path": "/episode", "description": "Save session episode"},
            {"method": "GET", "path": "/episodes", "description": "List episodes"},
            {"method": "PUT", "path": "/state/{project_id}", "description": "Save working state"},
            {"method": "GET", "path": "/state/{project_id}", "description": "Get working state"},
            {"method": "DELETE", "path": "/state/{project_id}", "description": "Clear working state"},
            {"method": "POST", "path": "/bootstrap", "description": "Get full session context"},
            {"method": "GET", "path": "/status", "description": "Service status"},
            {"method": "POST", "path": "/decay", "description": "Trigger memory score decay"},
            {"method": "POST", "path": "/publish", "description": "Publish a blog post"},
            {"method": "GET", "path": "/health", "description": "Health check"},
        ],
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/remember")
async def remember(request: RememberRequest, response: Response) -> dict[str, str]:
    store: MemoryStore = app.state.store
    embeddings: EmbeddingClient = app.state.embeddings

    try:
        embedding = await embeddings.embed(request.content)

        existing_id, similarity = store.check_near_duplicate(
            embedding, settings.consolidation_threshold, settings.dedupe_threshold
        )

        if existing_id and similarity >= settings.dedupe_threshold:
            response.status_code = 200
            return {"status": "duplicate", "existing_id": existing_id}

        if existing_id and similarity >= settings.consolidation_threshold:
            store.consolidate_memory(
                existing_id=existing_id,
                new_content=request.content,
                new_tags=request.tags,
                new_confidence=request.confidence,
                new_metadata=request.metadata,
            )
            response.status_code = 200
            return {"status": "consolidated", "existing_id": existing_id}

        memory_id = MemoryStore.generate_memory_id()
        store.add_memory(
            memory_id=memory_id,
            content=request.content,
            embedding=embedding,
            type=request.type.value,
            scope=request.scope.value,
            project_id=request.project_id,
            tags=request.tags,
            source=request.source,
            confidence=request.confidence,
            metadata=request.metadata,
        )
        response.status_code = 201
        return {"status": "created", "id": memory_id}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to store memory")
        raise HTTPException(status_code=500, detail=f"Failed to store memory: {exc}") from exc


@app.post("/extract")
async def extract(request: ExtractRequest) -> ExtractResponse:
    store: MemoryStore = app.state.store
    embeddings: EmbeddingClient = app.state.embeddings

    try:
        client = embeddings._ensure_client()
        memories = await extract_memories(
            client=client,
            model=settings.extraction_model,
            text=request.text,
            context=request.context,
        )

        if not memories:
            return ExtractResponse(extracted=0, memory_ids=[])

        texts = [m.content for m in memories]
        vectors = await embeddings.embed_batch(texts)

        # Types that are inherently global (not project-specific)
        _GLOBAL_TYPES = {"preference", "convention"}

        memory_ids: list[str] = []
        for mem, embedding in zip(memories, vectors):
            memory_id = MemoryStore.generate_memory_id()
            # Infer scope: explicit override > type-based inference > project default
            if request.scope:
                mem_scope = request.scope
            elif mem.type.value in _GLOBAL_TYPES:
                mem_scope = "global"
            else:
                mem_scope = "project"
            store.add_memory(
                memory_id=memory_id,
                content=mem.content,
                embedding=embedding,
                type=mem.type.value,
                scope=mem_scope,
                project_id=request.project_id,
                tags=mem.tags + (["extracted", f"source:{request.source}"] if request.source else ["extracted"]),
                source=request.source,
                confidence=mem.confidence,
                metadata={},
            )
            memory_ids.append(memory_id)

        return ExtractResponse(extracted=len(memory_ids), memory_ids=memory_ids)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception:
        logger.exception("Extraction failed")
        return ExtractResponse(extracted=0, memory_ids=[])


@app.post("/recall")
async def recall(query: RecallQuery) -> RecallResponse:
    store: MemoryStore = app.state.store
    embeddings: EmbeddingClient = app.state.embeddings

    try:
        query_embedding = await embeddings.embed(query.query)

        type_values = [t.value for t in query.types] if query.types else None
        scope_value = query.scope.value if query.scope else None
        tag_values = query.tags if query.tags else None

        results = store.search_memories(
            query_embedding=query_embedding,
            limit=query.limit,
            threshold=query.threshold,
            types=type_values,
            scope=scope_value,
            project_id=query.project_id,
            tags=tag_values,
        )

        recall_results = []
        for mem_dict, similarity in results:
            store.update_retrieval_count(mem_dict["id"])
            recall_results.append(
                RecallResult(
                    memory=MemoryRecord(**{
                        k: v for k, v in mem_dict.items() if k in MemoryRecord.model_fields
                    }),
                    similarity=similarity,
                )
            )

        return RecallResponse(results=recall_results, query=query.query, total=len(recall_results))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to recall memories")
        raise HTTPException(status_code=500, detail=f"Failed to recall: {exc}") from exc


@app.get("/memories")
async def list_memories(limit: int = 50, offset: int = 0) -> list[MemoryRecord]:
    store: MemoryStore = app.state.store
    memories = store.list_memories(limit=limit, offset=offset)
    return [
        MemoryRecord(**{k: v for k, v in mem.items() if k in MemoryRecord.model_fields})
        for mem in memories
    ]


@app.get("/recall/{memory_id}")
async def get_memory(memory_id: str) -> MemoryRecord:
    store: MemoryStore = app.state.store
    mem = store.get_memory(memory_id)
    if mem is None:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
    return MemoryRecord(**{k: v for k, v in mem.items() if k in MemoryRecord.model_fields})


@app.delete("/forget/{memory_id}", status_code=204)
async def forget(memory_id: str) -> Response:
    store: MemoryStore = app.state.store
    deleted = store.delete_memory(memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
    return Response(status_code=204)


@app.post("/episode", status_code=201)
async def save_episode(request: EpisodeRequest) -> dict[str, Any]:
    store: MemoryStore = app.state.store
    embeddings: EmbeddingClient = app.state.embeddings

    try:
        episode_id = MemoryStore.generate_episode_id()
        data = {
            "todos": [t.model_dump() for t in request.todos],
            "decisions": [d.model_dump() for d in request.decisions],
            "constraints": [c.model_dump() for c in request.constraints],
            "failed_approaches": [f.model_dump() for f in request.failed_approaches],
            "explored_files": request.explored_files,
        }
        store.add_episode(
            episode_id=episode_id,
            session_id=request.session_id,
            project_id=request.project_id,
            summary=request.summary,
            data=data,
            metadata=request.metadata,
        )

        extracted = 0
        items_to_embed: list[tuple[str, str]] = []

        for decision in request.decisions:
            items_to_embed.append((decision.content, "decision"))
        for constraint in request.constraints:
            items_to_embed.append((constraint.content, "constraint"))

        if items_to_embed:
            try:
                texts = [item[0] for item in items_to_embed]
                vectors = await embeddings.embed_batch(texts)
                for (content, mem_type), vector in zip(items_to_embed, vectors):
                    mem_id = MemoryStore.generate_memory_id()
                    store.add_memory(
                        memory_id=mem_id,
                        content=content,
                        embedding=vector,
                        type=mem_type,
                        scope="project",
                        project_id=request.project_id,
                        tags=["auto-extracted", f"session:{request.session_id}"],
                        source=f"episode:{episode_id}",
                        confidence=0.8,
                        metadata={"episode_id": episode_id},
                    )
                    extracted += 1
            except Exception:
                logger.warning("Failed to extract memories from episode %s", episode_id)

        return {"status": "created", "id": episode_id, "extracted_memories": extracted}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to save episode")
        raise HTTPException(status_code=500, detail=f"Failed to save episode: {exc}") from exc


@app.get("/episodes")
async def list_episodes(
    project_id: str | None = None,
    limit: int = 20,
) -> list[EpisodeRecord]:
    store: MemoryStore = app.state.store
    episodes = store.list_episodes(project_id=project_id, limit=limit)

    results = []
    for ep in episodes:
        ep_data = ep.get("data", {})
        results.append(
            EpisodeRecord(
                id=ep["id"],
                session_id=ep["session_id"],
                project_id=ep["project_id"],
                summary=ep["summary"],
                todos=ep_data.get("todos", []),
                decisions=ep_data.get("decisions", []),
                constraints=ep_data.get("constraints", []),
                failed_approaches=ep_data.get("failed_approaches", []),
                explored_files=ep_data.get("explored_files", []),
                created_at=ep["created_at"],
                metadata=ep.get("metadata", {}),
            )
        )
    return results


@app.put("/state/{project_id:path}")
async def save_state(project_id: str, request: WorkingState) -> dict[str, str]:
    store: MemoryStore = app.state.store
    store.set_state(project_id, request.model_dump())
    return {"status": "saved", "project_id": project_id}


@app.get("/state/{project_id:path}")
async def get_state(project_id: str) -> dict[str, Any]:
    store: MemoryStore = app.state.store
    state = store.get_state(project_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"No state for project {project_id}")
    return state


@app.get("/states")
async def list_states() -> list[dict[str, Any]]:
    store: MemoryStore = app.state.store
    return store.list_states()


@app.delete("/state/{project_id:path}", status_code=204)
async def delete_state(project_id: str) -> Response:
    store: MemoryStore = app.state.store
    deleted = store.delete_state(project_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"No state for project {project_id}")
    return Response(status_code=204)


@app.post("/decay")
async def trigger_decay() -> dict[str, Any]:
    store: MemoryStore = app.state.store
    result = store.decay_scores(settings.decay_factor, settings.min_score)
    app.state.last_decay_at = result["decayed_at"]
    return result


@app.post("/bootstrap")
async def bootstrap(request: BootstrapRequest) -> BootstrapResponse:
    store: MemoryStore = app.state.store

    project_id = request.project_id
    if not project_id and request.cwd:
        project_id = request.cwd.rstrip("/").rsplit("/", 1)[-1]

    response = BootstrapResponse(project_id=project_id)

    if request.include_state and project_id:
        state_data = store.get_state(project_id)
        if state_data:
            response.state = WorkingState(**state_data["data"])

    if request.include_memories:
        limit = request.memory_limit
        identity_slots = max(3, limit // 5)
        constraint_slots = limit - identity_slots

        identity_mems = store.get_memories_by_type(
            ["architecture", "preference", "convention"],
            project_id=project_id,
            limit=identity_slots,
        )
        constraint_mems_raw = store.get_memories_by_type(
            ["constraint"],
            project_id=project_id,
            limit=constraint_slots,
        )

        seen_ids: set[str] = set()
        for mem in identity_mems + constraint_mems_raw:
            if mem["id"] in seen_ids:
                continue
            seen_ids.add(mem["id"])
            response.memories.append(
                MemoryRecord(**{k: v for k, v in mem.items() if k in MemoryRecord.model_fields})
            )

        for mem in constraint_mems_raw:
            response.constraints.append(
                MemoryRecord(**{k: v for k, v in mem.items() if k in MemoryRecord.model_fields})
            )

        if project_id:
            error_mems = store.get_memories_by_type(
                ["error-solution", "failure"], project_id=project_id, limit=10
            )
            for mem in error_mems:
                response.failed_approaches.append(
                    MemoryRecord(**{k: v for k, v in mem.items() if k in MemoryRecord.model_fields})
                )

    if (
        request.include_cross_project
        and settings.cross_project_enabled
        and project_id
    ):
        cross_slots = max(1, int(request.memory_limit * settings.cross_project_budget_pct))
        global_mems = store.get_global_memories(
            allowed_types=settings.cross_project_types,
            exclude_project_id=project_id,
            limit=cross_slots,
        )
        seen_ids_for_cross = {m.id for m in response.memories}
        for mem in global_mems:
            if mem["id"] in seen_ids_for_cross:
                continue
            record = MemoryRecord(
                **{k: v for k, v in mem.items() if k in MemoryRecord.model_fields}
            )
            response.cross_project.append(
                CrossProjectMemory(
                    memory=record,
                    origin_project=mem.get("project_id"),
                )
            )

    if request.include_episodes and project_id:
        episodes = store.list_episodes(project_id=project_id, limit=request.episode_limit)
        for ep in episodes:
            ep_data = ep.get("data", {})
            response.recent_episodes.append(
                EpisodeRecord(
                    id=ep["id"],
                    session_id=ep["session_id"],
                    project_id=ep["project_id"],
                    summary=ep["summary"],
                    todos=ep_data.get("todos", []),
                    decisions=ep_data.get("decisions", []),
                    constraints=ep_data.get("constraints", []),
                    failed_approaches=ep_data.get("failed_approaches", []),
                    explored_files=ep_data.get("explored_files", []),
                    created_at=ep["created_at"],
                    metadata=ep.get("metadata", {}),
                )
            )

    return response


@app.post("/publish", status_code=201)
async def publish(request: PublishRequest) -> dict[str, Any]:
    import re
    import subprocess
    from datetime import date
    from pathlib import Path

    repo = Path(settings.blog_repo_path)
    content_dir = repo / settings.blog_content_dir
    if not content_dir.is_dir():
        raise HTTPException(status_code=500, detail=f"Blog content dir not found: {content_dir}")

    slug = request.slug
    if not slug:
        slug = re.sub(r"[^a-z0-9]+", "-", request.title.lower()).strip("-")

    filename = f"{slug}.md"
    filepath = content_dir / filename

    if filepath.exists():
        raise HTTPException(status_code=409, detail=f"Post already exists: {filename}")

    today = date.today().isoformat()
    frontmatter = (
        f"---\n"
        f"title: '{request.title}'\n"
        f"description: '{request.description}'\n"
        f"pubDate: '{today}'\n"
        f"author: '{request.author}'\n"
        f"---\n\n"
    )
    filepath.write_text(frontmatter + request.body, encoding="utf-8")

    try:
        subprocess.run(
            ["git", "add", str(filepath)],
            cwd=str(repo), check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", f"post: {request.title}"],
            cwd=str(repo), check=True, capture_output=True,
        )

        pushed = False
        if request.push:
            subprocess.run(
                ["git", "push", "origin", "main"],
                cwd=str(repo), check=True, capture_output=True,
            )
            pushed = True

        return {
            "status": "published",
            "slug": slug,
            "file": str(filepath),
            "pushed": pushed,
            "url": f"https://minzique.github.io/log/blog/{slug}/",
        }
    except subprocess.CalledProcessError as exc:
        logger.exception("Git operation failed during publish")
        raise HTTPException(
            status_code=500,
            detail=f"Git failed: {exc.stderr.decode() if exc.stderr else str(exc)}",
        ) from exc


@app.get("/status")
async def status() -> ServiceStatus:
    store: MemoryStore = app.state.store
    stats = store.get_stats()
    return ServiceStatus(
        healthy=True,
        version="0.4.0",
        memory_count=stats["memory_count"],
        episode_count=stats["episode_count"],
        state_count=stats["state_count"],
        db_size_bytes=stats["db_size_bytes"],
        embedding_provider="openai",
        uptime_seconds=time.time() - app.state.start_time,
    )


def run():
    import uvicorn

    uvicorn.run("memory_service.main:app", host=settings.host, port=settings.port, reload=True)
