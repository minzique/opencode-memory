from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Response

from memory_service.config import settings
from memory_service.embeddings import EmbeddingClient
from memory_service.models import (
    EpisodeRecord,
    EpisodeRequest,
    MemoryRecord,
    RecallQuery,
    RecallResponse,
    RecallResult,
    RememberRequest,
    ServiceStatus,
)
from memory_service.store import MemoryStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.store = MemoryStore(settings.db_path)
    app.state.embeddings = EmbeddingClient(
        settings.openai_api_key, settings.embedding_model, settings.embedding_dimensions
    )
    app.state.start_time = time.time()
    logger.info("Memory service started on %s:%d", settings.host, settings.port)
    yield
    app.state.store.close()
    logger.info("Memory service stopped")


app = FastAPI(
    title="OpenCode Memory Service",
    description="Persistent semantic memory for AI agents",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "service": "opencode-memory",
        "version": "0.1.0",
        "endpoints": [
            {"method": "POST", "path": "/remember", "description": "Store a memory"},
            {"method": "POST", "path": "/recall", "description": "Search memories"},
            {"method": "GET", "path": "/recall/{memory_id}", "description": "Get specific memory"},
            {"method": "DELETE", "path": "/forget/{memory_id}", "description": "Delete a memory"},
            {"method": "POST", "path": "/episode", "description": "Save session episode"},
            {"method": "GET", "path": "/episodes", "description": "List episodes"},
            {"method": "GET", "path": "/status", "description": "Service status"},
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

        existing_id = store.check_duplicate(embedding, settings.dedupe_threshold)
        if existing_id:
            response.status_code = 200
            return {"status": "duplicate", "existing_id": existing_id}

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


@app.get("/status")
async def status() -> ServiceStatus:
    store: MemoryStore = app.state.store
    stats = store.get_stats()
    return ServiceStatus(
        healthy=True,
        version="0.1.0",
        memory_count=stats["memory_count"],
        episode_count=stats["episode_count"],
        db_size_bytes=stats["db_size_bytes"],
        embedding_provider="openai",
        uptime_seconds=time.time() - app.state.start_time,
    )


def run():
    import uvicorn

    uvicorn.run("memory_service.main:app", host=settings.host, port=settings.port, reload=True)
