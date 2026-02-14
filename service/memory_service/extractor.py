from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI

from memory_service.models import ExtractedMemory, MemoryType

logger = logging.getLogger(__name__)

_VALID_TYPES = {t.value for t in MemoryType}

_SYSTEM_PROMPT = (
    "You extract structured memories from text. "
    "Return a JSON array of objects with these fields:\n"
    '- "content": concise statement of the knowledge\n'
    '- "type": one of: decision, constraint, fact, error-solution, preference, convention\n'
    '- "confidence": 0.0-1.0 how certain this is\n'
    '- "tags": list of short keyword tags\n\n'
    "Rules:\n"
    "- Only extract clear, actionable knowledge\n"
    "- Skip vague or trivial statements\n"
    "- Each memory should stand alone without context\n"
    "- Return [] if nothing worth extracting\n"
    "- Return ONLY the JSON array, no markdown fences"
)


async def extract_memories(
    client: AsyncOpenAI,
    model: str,
    text: str,
    context: str | None = None,
) -> list[ExtractedMemory]:
    user_content = text
    if context:
        user_content = f"Context: {context}\n\n{text}"

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            max_tokens=2048,
        )

        raw = response.choices[0].message.content or "[]"
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        items = json.loads(raw)
        if not isinstance(items, list):
            logger.warning("Extraction returned non-list: %s", type(items))
            return []

        memories: list[ExtractedMemory] = []
        for item in items:
            if not isinstance(item, dict) or "content" not in item:
                continue
            mem_type = item.get("type", "fact")
            if mem_type not in _VALID_TYPES:
                mem_type = "fact"
            memories.append(
                ExtractedMemory(
                    content=item["content"],
                    type=MemoryType(mem_type),
                    confidence=min(1.0, max(0.0, float(item.get("confidence", 0.8)))),
                    tags=item.get("tags", []),
                )
            )
        return memories

    except json.JSONDecodeError:
        logger.warning("Failed to parse extraction JSON")
        return []
    except Exception:
        logger.exception("Memory extraction failed")
        return []
