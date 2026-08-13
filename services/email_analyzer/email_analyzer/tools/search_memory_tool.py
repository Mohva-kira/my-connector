"""SearchMemoryTool — thin wrapper over MemoryService.list_memories, with
an optional keyword filter. Pure delegation: no new query logic, this
tool exists to give Memory a "tool-shaped" entry point for Phase 9's
Agent (see tools/base.py)."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from email_analyzer.services.memory_service import MemoryService


class SearchMemoryTool:
    name = "search_memory"
    description = (
        "Recherche parmi les souvenirs retenus sur l'utilisateur (préférences, "
        "faits durables), optionnellement filtrés par un mot-clé dans le contenu."
    )

    def __init__(self, db: Session) -> None:
        self._service = MemoryService(db)

    def run(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, keyword: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        memories = self._service.list_memories(tenant_id, user_id)
        if keyword:
            needle = keyword.strip().lower()
            memories = [m for m in memories if needle in m.content.lower()]
        return [
            {"memory_id": str(m.id), "category": m.category, "content": m.content}
            for m in memories
        ]
