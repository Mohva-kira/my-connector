"""Data access for the `memories` table (Memory Engine foundation).

Same pattern as ConversationRepository: single place issuing SQL for this
concept, no query logic in the service or the router.
"""

from __future__ import annotations

import uuid
from typing import List, Optional

from sqlalchemy.orm import Session

from email_analyzer.db.models import Memory


class MemoryRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def list_for_user(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, limit: int = 200
    ) -> List[Memory]:
        return (
            self._db.query(Memory)
            .filter(Memory.tenant_id == tenant_id, Memory.user_id == user_id)
            .order_by(Memory.confidence.desc(), Memory.created_at.desc())
            .limit(limit)
            .all()
        )

    def find_by_content(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, content: str
    ) -> Optional[Memory]:
        """Déduplication exacte (insensible à la casse/aux espaces) — pas de
        similarité vectorielle en v1, voir Memory.__doc__."""
        normalized = content.strip().lower()
        for m in self.list_for_user(tenant_id, user_id, limit=1000):
            if m.content.strip().lower() == normalized:
                return m
        return None

    def create(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        content: str,
        category: str,
        source: str,
        confidence: float,
    ) -> Memory:
        memory = Memory(
            tenant_id=tenant_id,
            user_id=user_id,
            content=content,
            category=category,
            source=source,
            confidence=confidence,
        )
        self._db.add(memory)
        self._db.commit()
        self._db.refresh(memory)
        return memory

    def delete(self, tenant_id: uuid.UUID, user_id: uuid.UUID, memory_id: uuid.UUID) -> bool:
        memory = (
            self._db.query(Memory)
            .filter(
                Memory.id == memory_id, Memory.tenant_id == tenant_id, Memory.user_id == user_id
            )
            .first()
        )
        if memory is None:
            return False
        self._db.delete(memory)
        self._db.commit()
        return True
