"""Knowledge Layer foundation (Personal Executive AI Assistant pivot, Phase 6).

Deterministic pipeline, no new LLM call: Project -> ProjectSummary.
structured_content (already produced by the existing Fast-Track/batch
pipeline, email_analyzer.llm.ProjectSummaryLLM) -> Contact/Organization
rows + ProjectContactLink. "Entities"/"Facts"/"Relationships" (named in the
pivot plan) are represented today only as Contact/Organization/
ProjectContactLink — the concrete subset the codebase already has data
for; a generic Fact table would duplicate Memory (see memory_service.py),
so it is deliberately not introduced here.

Syncing is also available as an explicit action (api/routers/knowledge.py:
per-project and bulk-per-tenant) for manual/on-demand use, but the primary
path is now automatic: `_run_fasttrack_sync` (analysis_tasks.py) calls
`sync_from_project_summary` right after every analysis that produced new
emails, and `run_knowledge_sync_scan` (hourly arq cron) catches any project
whose summary changed without an immediate sync (e.g. transient failure).
"""

from __future__ import annotations

import logging
import uuid
from typing import Dict, List, Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from email_analyzer.db.models import Contact, Organization, Project, ProjectStatus, ProjectSummary
from email_analyzer.repositories.knowledge_repository import KnowledgeRepository

logger = logging.getLogger(__name__)


class KnowledgeService:
    def __init__(self, db: Session) -> None:
        self._db = db
        self._repo = KnowledgeRepository(db)

    def list_contacts(self, tenant_id: uuid.UUID, limit: int = 200) -> List[Contact]:
        return self._repo.list_contacts_for_tenant(tenant_id, limit)

    def list_organizations(self, tenant_id: uuid.UUID, limit: int = 200) -> List[Organization]:
        return self._repo.list_organizations_for_tenant(tenant_id, limit)

    def list_project_contacts(self, tenant_id: uuid.UUID, project_id: uuid.UUID) -> List[Contact]:
        project = self._get_project(tenant_id, project_id)
        return self._repo.list_contacts_for_project(project.id)

    def sync_all_for_tenant(self, tenant_id: uuid.UUID) -> Dict[str, int]:
        """Variante portefeuille de `sync_from_project_summary` — boucle sur
        tous les projets actifs du tenant (bouton "Synchroniser" du chat,
        POST /api/knowledge/sync). Échec isolé par projet (log + continue),
        jamais bloquant pour les autres, même discipline que
        analysis_tasks.run_scheduled_sync."""
        projects = (
            self._db.query(Project)
            .filter(Project.tenant_id == tenant_id, Project.status == ProjectStatus.active.value)
            .all()
        )
        synced = 0
        failed = 0
        for project in projects:
            try:
                self.sync_from_project_summary(tenant_id, project.id)
                synced += 1
            except Exception:
                logger.exception(
                    "knowledge.sync_all : échec project_id=%s tenant_id=%s", project.id, tenant_id
                )
                failed += 1
        return {"synced_projects": synced, "failed_projects": failed, "total_projects": len(projects)}

    def sync_from_project_summary(self, tenant_id: uuid.UUID, project_id: uuid.UUID) -> List[Contact]:
        """Relit `ProjectSummary.structured_content` du projet et upsert
        les Organization/Contact/ProjectContactLink correspondants. Retourne
        les contacts liés au projet après synchronisation (créés + déjà
        existants)."""
        project = self._get_project(tenant_id, project_id)
        logger.info("knowledge.sync start tenant=%s project=%s", tenant_id, project_id)

        summary = self._db.query(ProjectSummary).filter(ProjectSummary.project_id == project.id).first()
        structured = (summary.structured_content if summary else None) or {}
        company_names = [c.strip() for c in (structured.get("companies") or []) if c and c.strip()]
        participant_names = [
            p.strip() for p in (structured.get("participants") or []) if p and p.strip()
        ]

        # Une seule organisation par nom pour ce sync (v1 : pas de résolution
        # "quel participant appartient à quelle société" — le LLM ne le
        # fournit pas encore de façon fiable). Les contacts restent liés au
        # projet même sans organisation associée.
        primary_org: Optional[Organization] = None
        for name in company_names:
            org = self._repo.get_or_create_organization(tenant_id, name)
            if primary_org is None:
                primary_org = org

        for name in participant_names:
            contact = self._repo.get_or_create_contact(
                tenant_id, name, organization_id=primary_org.id if primary_org else None
            )
            self._repo.link_project_contact(project.id, contact.id)

        linked = self._repo.list_contacts_for_project(project.id)
        logger.info(
            "knowledge.sync done tenant=%s project=%s contacts=%d organizations=%d",
            tenant_id,
            project_id,
            len(linked),
            len(company_names),
        )
        return linked

    def _get_project(self, tenant_id: uuid.UUID, project_id: uuid.UUID) -> Project:
        project = (
            self._db.query(Project)
            .filter(Project.id == project_id, Project.tenant_id == tenant_id)
            .first()
        )
        if project is None:
            raise HTTPException(status_code=404, detail="Projet introuvable")
        return project
