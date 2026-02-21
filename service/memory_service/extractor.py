from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI

from memory_service.models import ExtractedMemory, MemoryType

logger = logging.getLogger(__name__)

_VALID_TYPES = {t.value for t in MemoryType}

_SYSTEM_PROMPT = (
    "Extract ONLY high-signal, reusable knowledge from user messages. "
    "Return a JSON array of objects. Return [] if nothing is worth remembering.\n\n"
    "Fields per object:\n"
    '- "content": ONE concise sentence (max 150 chars). Must stand alone without context.\n'
    '- "type": decision | constraint | architecture | pattern | convention | preference | error-solution | failure | fact\n'
    '- "confidence": 0.0-1.0\n'
    '- "tags": [short keywords]\n\n'
    "EXTRACT:\n"
    "- Explicit choices: 'use X over Y', 'switch to Z'\n"
    "- Rules the user states: 'always/never/must/avoid'\n"
    "- Preferences: 'I prefer X', 'let's use Y'\n"
    "- Error patterns: 'X broke because Y, fix: Z'\n\n"
    "REJECT (return []):\n"
    "- Task instructions ('I need you to...', 'please implement...')\n"
    "- Questions without answers\n"
    "- Casual chat, greetings, acknowledgments\n"
    "- Context that's only relevant to the current session\n"
    "- Anything vague or not reusable across sessions\n\n"
    "Return ONLY the JSON array, no markdown fences."
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
