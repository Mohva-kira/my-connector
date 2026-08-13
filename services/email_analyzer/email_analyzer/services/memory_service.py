"""Memory Engine foundation (Personal Executive AI Assistant pivot, Phase 5).

Deterministic pipeline: conversation turns -> fact candidates (LLM,
structured output) -> validation (category/confidence, dedup) -> Memory
row. No vector search, no automatic wiring into every chat call yet —
extraction is an explicit action (see api/routers/memory.py), triggered by
the caller, so nothing about existing chat behavior or cost changes
silently.

"Extractor"/"Store"/"Retriever"/"Scoring" (named in the pivot plan as
separate modules) are intentionally kept as methods on this one service
instead of five near-empty classes — each currently has one responsibility
and a couple of lines; splitting them now would be premature abstraction
(context/ai-workflow-rules.md, "avoid unnecessary abstractions"). Split
out if/when a method grows real, independent complexity.
"""

from __future__ import annotations

import logging
import uuid
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from email_analyzer.config import DEFAULT_GEMINI_MODEL
from email_analyzer.db.models import Memory, MemoryCategory
from email_analyzer.llm import (
    extract_memory_candidates_gemini,
    extract_memory_candidates_openai,
    get_gemini_api_key,
    get_openai_api_key,
    select_assistant_provider,
)
from email_analyzer.repositories.memory_repository import MemoryRepository

logger = logging.getLogger(__name__)

_VALID_CATEGORIES = {c.value for c in MemoryCategory}
_MIN_CONFIDENCE = 0.5


class MemoryService:
    def __init__(self, db: Session) -> None:
        self._repo = MemoryRepository(db)

    def list_memories(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, limit: int = 200
    ) -> List[Memory]:
        return self._repo.list_for_user(tenant_id, user_id, limit)

    def delete_memory(self, tenant_id: uuid.UUID, user_id: uuid.UUID, memory_id: uuid.UUID) -> bool:
        return self._repo.delete(tenant_id, user_id, memory_id)

    def extract_and_store(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        turns: List[Dict[str, str]],
        preferred_provider: Optional[str] = None,
        openai_model: str = "gpt-4o-mini",
        gemini_model: str = DEFAULT_GEMINI_MODEL,
    ) -> List[Memory]:
        """Extrait les faits durables d'un échange et les persiste
        (dédupliqués). Retourne uniquement les *nouveaux* Memory créés."""
        logger.info("memory.extract_and_store start tenant=%s user=%s", tenant_id, user_id)

        provider = select_assistant_provider(preferred_provider)
        if provider is None:
            logger.warning("memory.extract_and_store no_provider tenant=%s", tenant_id)
            return []

        if provider == "gemini":
            result = extract_memory_candidates_gemini(turns, gemini_model, get_gemini_api_key())
        else:
            result = extract_memory_candidates_openai(turns, openai_model, get_openai_api_key())

        if result.get("erreur") or result.get("data") is None:
            logger.warning(
                "memory.extract_and_store llm_error tenant=%s error=%s",
                tenant_id,
                result.get("erreur"),
            )
            return []

        created: List[Memory] = []
        for candidate in result["data"].facts:
            content = candidate.content.strip()
            category = candidate.category if candidate.category in _VALID_CATEGORIES else MemoryCategory.fact.value
            if not content or candidate.confidence < _MIN_CONFIDENCE:
                continue
            if self._repo.find_by_content(tenant_id, user_id, content) is not None:
                continue
            memory = self._repo.create(
                tenant_id=tenant_id,
                user_id=user_id,
                content=content,
                category=category,
                source="conversation",
                confidence=candidate.confidence,
            )
            created.append(memory)

        logger.info(
            "memory.extract_and_store done tenant=%s user=%s created=%d",
            tenant_id,
            user_id,
            len(created),
        )
        return created
