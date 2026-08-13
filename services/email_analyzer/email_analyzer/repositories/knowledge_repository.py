"""Data access for the Knowledge Layer (organizations, contacts, project
links). Same pattern as ConversationRepository/MemoryRepository — single
place issuing SQL for this concept, upserts done here so the service stays
focused on orchestration."""

from __future__ import annotations

import uuid
from typing import List, Optional

from sqlalchemy.orm import Session

from email_analyzer.db.models import Contact, Organization, ProjectContactLink


class KnowledgeRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def get_or_create_organization(self, tenant_id: uuid.UUID, name: str) -> Organization:
        org = (
            self._db.query(Organization)
            .filter(Organization.tenant_id == tenant_id, Organization.name == name)
            .first()
        )
        if org is not None:
            return org
        org = Organization(tenant_id=tenant_id, name=name)
        self._db.add(org)
        self._db.commit()
        self._db.refresh(org)
        return org

    def get_or_create_contact(
        self, tenant_id: uuid.UUID, name: str, organization_id: Optional[uuid.UUID] = None
    ) -> Contact:
        contact = (
            self._db.query(Contact)
            .filter(Contact.tenant_id == tenant_id, Contact.name == name)
            .first()
        )
        if contact is not None:
            if organization_id is not None and contact.organization_id is None:
                contact.organization_id = organization_id
                self._db.commit()
                self._db.refresh(contact)
            return contact
        contact = Contact(tenant_id=tenant_id, name=name, organization_id=organization_id)
        self._db.add(contact)
        self._db.commit()
        self._db.refresh(contact)
        return contact

    def link_project_contact(self, project_id: uuid.UUID, contact_id: uuid.UUID) -> None:
        existing = (
            self._db.query(ProjectContactLink)
            .filter(
                ProjectContactLink.project_id == project_id,
                ProjectContactLink.contact_id == contact_id,
            )
            .first()
        )
        if existing is not None:
            return
        self._db.add(ProjectContactLink(project_id=project_id, contact_id=contact_id))
        self._db.commit()

    def list_contacts_for_tenant(self, tenant_id: uuid.UUID, limit: int = 200) -> List[Contact]:
        return (
            self._db.query(Contact)
            .filter(Contact.tenant_id == tenant_id)
            .order_by(Contact.name.asc())
            .limit(limit)
            .all()
        )

    def list_contacts_for_project(self, project_id: uuid.UUID) -> List[Contact]:
        return (
            self._db.query(Contact)
            .join(ProjectContactLink, ProjectContactLink.contact_id == Contact.id)
            .filter(ProjectContactLink.project_id == project_id)
            .order_by(Contact.name.asc())
            .all()
        )

    def list_organizations_for_tenant(
        self, tenant_id: uuid.UUID, limit: int = 200
    ) -> List[Organization]:
        return (
            self._db.query(Organization)
            .filter(Organization.tenant_id == tenant_id)
            .order_by(Organization.name.asc())
            .limit(limit)
            .all()
        )
