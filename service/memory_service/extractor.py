from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI

from memory_service.models import ExtractedMemory, MemoryType

logger = logging.getLogger(__name__)

_VALID_TYPES = {t.value for t in MemoryType}

_SYSTEM_PROMPT = (
    "You are a memory extraction engine. Extract ONLY durable, reusable knowledge "
    "that will be valuable ACROSS FUTURE SESSIONS. Return a JSON array. Return [] if nothing qualifies.\n\n"
    "Fields per object:\n"
    '- "content": ONE concise, self-contained sentence (max 150 chars). Must make sense without any session context.\n'
    '- "type": decision | constraint | architecture | pattern | convention | preference | error-solution | failure | fact\n'
    '- "confidence": 0.0-1.0\n'
    '- "tags": [short keywords]\n\n'
    "EXTRACT (high bar — would you want this injected at the start of a brand-new session?):\n"
    "- Permanent rules: 'always X', 'never Y', 'must use Z'\n"
    "- Explicit tech choices: 'use Gemini over GPT-5 for extraction', 'switch DB to Postgres'\n"
    "- Preferences with lasting value: 'I prefer dark themes', 'monorepo layout'\n"
    "- Diagnosed error patterns: 'X fails because Y, fix: Z'\n"
    "- Architectural decisions: 'service A talks to B via gRPC'\n\n"
    "REJECT — return [] for ALL of these:\n"
    "- One-shot task instructions: 'make sure X', 'fix Y', 'deploy to prod', 'merge the PR'\n"
    "- Imperative commands: 'go through these issues', 'check if...', 'figure out...'\n"
    "- Status checks: 'is it running?', 'make sure it is stable'\n"
    "- Session-specific context: 'staging has changes prod does not', 'wait for CI'\n"
    "- Questions, greetings, acknowledgments, progress updates\n"
    "- Anything you would forget after the current task is done\n\n"
    "KEY TEST: If the message is the user telling the AI what to DO right now, it is a task — REJECT it.\n"
    "Only extract what the user BELIEVES or DECIDED permanently.\n\n"
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
            max_completion_tokens=2048,
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
