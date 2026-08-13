"""Contexte Personnel domain (Personal Executive AI Assistant vision,
context/project-overview.md — "Domaine #1 : la base de tout").

No LLM call here, unlike MemoryService.extract_and_store — the profile is
entered directly by the user in Settings, not extracted from conversation.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from email_analyzer.db.models import Profile
from email_analyzer.repositories.profile_repository import ProfileRepository


class ProfileService:
    def __init__(self, db: Session) -> None:
        self._repo = ProfileRepository(db)

    def get_profile(self, tenant_id: uuid.UUID, user_id: uuid.UUID) -> Optional[Profile]:
        return self._repo.get(tenant_id, user_id)

    def update_profile(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, fields: Dict[str, Any]
    ) -> Profile:
        """`fields` ne doit contenir que les clés réellement fournies par
        l'appelant (ex. body Pydantic avec `exclude_unset=True`) — une clé
        absente laisse la valeur existante inchangée, une clé présente avec
        `None` efface explicitement le champ."""
        cleaned = {
            key: (value.strip() if isinstance(value, str) else value)
            for key, value in fields.items()
        }
        return self._repo.upsert(tenant_id, user_id, **cleaned)
