"""FindContactTool — thin wrapper over KnowledgeService, filtering
contacts by a name substring. Pure delegation, same rationale as
SearchMemoryTool."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from email_analyzer.services.knowledge_service import KnowledgeService


class FindContactTool:
    name = "find_contact"
    description = (
        "Recherche un contact connu (personne mentionnée dans les projets suivis) "
        "par nom, avec son organisation si connue."
    )

    def __init__(self, db: Session) -> None:
        self._service = KnowledgeService(db)

    def run(self, tenant_id: uuid.UUID, name: Optional[str] = None) -> List[Dict[str, Any]]:
        contacts = self._service.list_contacts(tenant_id)
        if name:
            needle = name.strip().lower()
            contacts = [c for c in contacts if needle in c.name.lower()]

        orgs_by_id = {o.id: o.name for o in self._service.list_organizations(tenant_id)}
        return [
            {
                "contact_id": str(c.id),
                "name": c.name,
                "organization": orgs_by_id.get(c.organization_id),
            }
            for c in contacts
        ]
