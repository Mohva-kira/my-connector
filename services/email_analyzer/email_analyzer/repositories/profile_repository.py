"""Data access for the `profiles` table (Contexte Personnel domain).

Unlike MemoryRepository (many rows per user), a profile is a singleton per
(tenant, user) — get-or-create + partial update, no list/delete-by-id.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from sqlalchemy.orm import Session

from email_analyzer.db.models import Profile


class ProfileRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def get(self, tenant_id: uuid.UUID, user_id: uuid.UUID) -> Optional[Profile]:
        return (
            self._db.query(Profile)
            .filter(Profile.tenant_id == tenant_id, Profile.user_id == user_id)
            .first()
        )

    def upsert(self, tenant_id: uuid.UUID, user_id: uuid.UUID, **fields: Any) -> Profile:
        """Crée le profil s'il n'existe pas encore, sinon met à jour
        uniquement les champs fournis (les clés absentes de `fields` ne
        sont pas touchées — mise à jour partielle, cohérent avec un body
        PATCH où tous les champs sont optionnels)."""
        profile = self.get(tenant_id, user_id)
        if profile is None:
            profile = Profile(tenant_id=tenant_id, user_id=user_id)
            self._db.add(profile)
        for key, value in fields.items():
            setattr(profile, key, value)
        self._db.commit()
        self._db.refresh(profile)
        return profile
