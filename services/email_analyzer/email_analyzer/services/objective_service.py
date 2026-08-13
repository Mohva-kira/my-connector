"""Objectives foundation (Personal Executive AI Assistant pivot, Phase 11).

"Use existing projects whenever possible" (pivot plan): an Objective can
optionally link to a Project (validated to belong to the same tenant),
otherwise it stands alone as a personal goal. Progress is computed from
milestones on read — see ObjectiveRepository/Objective.__doc__ for why
there is no separate "Progress" table.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from email_analyzer.db.models import Milestone, Objective, ObjectiveStatus, Project
from email_analyzer.repositories.objective_repository import ObjectiveRepository

logger = logging.getLogger(__name__)

_VALID_STATUSES = {s.value for s in ObjectiveStatus}


@dataclass
class ObjectiveProgress:
    objective: Objective
    milestones: List[Milestone]
    completed_count: int
    total_count: int
    percent: int


class ObjectiveService:
    def __init__(self, db: Session) -> None:
        self._db = db
        self._repo = ObjectiveRepository(db)

    def create_objective(
        self,
        tenant_id: uuid.UUID,
        title: str,
        project_id: Optional[uuid.UUID] = None,
        target_date: Optional[datetime] = None,
    ) -> Objective:
        title = title.strip()
        if not title:
            raise HTTPException(status_code=422, detail="Titre d'objectif vide")
        if project_id is not None:
            project = (
                self._db.query(Project)
                .filter(Project.id == project_id, Project.tenant_id == tenant_id)
                .first()
            )
            if project is None:
                raise HTTPException(status_code=404, detail="Projet introuvable")

        logger.info("objective.create tenant=%s project=%s", tenant_id, project_id)
        return self._repo.create(tenant_id, title, project_id, target_date)

    def list_objectives(
        self, tenant_id: uuid.UUID, project_id: Optional[uuid.UUID] = None
    ) -> List[ObjectiveProgress]:
        return [self._with_progress(o) for o in self._repo.list_for_tenant(tenant_id, project_id)]

    def get_objective(self, tenant_id: uuid.UUID, objective_id: uuid.UUID) -> ObjectiveProgress:
        objective = self._get(tenant_id, objective_id)
        return self._with_progress(objective)

    def update_status(
        self,
        tenant_id: uuid.UUID,
        objective_id: uuid.UUID,
        status: str,
        blocking_reason: Optional[str] = None,
    ) -> ObjectiveProgress:
        if status not in _VALID_STATUSES:
            raise HTTPException(status_code=422, detail=f"Statut invalide : {status}")
        objective = self._get(tenant_id, objective_id)
        logger.info("objective.update_status tenant=%s objective=%s status=%s", tenant_id, objective_id, status)
        objective = self._repo.update_status(objective, status, blocking_reason)
        return self._with_progress(objective)

    def add_milestone(
        self,
        tenant_id: uuid.UUID,
        objective_id: uuid.UUID,
        title: str,
        due_date: Optional[datetime] = None,
    ) -> ObjectiveProgress:
        title = title.strip()
        if not title:
            raise HTTPException(status_code=422, detail="Titre de jalon vide")
        objective = self._get(tenant_id, objective_id)
        self._repo.add_milestone(objective.id, title, due_date)
        return self._with_progress(objective)

    def set_milestone_status(
        self, tenant_id: uuid.UUID, objective_id: uuid.UUID, milestone_id: uuid.UUID, status: str
    ) -> ObjectiveProgress:
        if status not in {"pending", "done"}:
            raise HTTPException(status_code=422, detail=f"Statut de jalon invalide : {status}")
        objective = self._get(tenant_id, objective_id)
        milestone = self._repo.get_milestone(objective.id, milestone_id)
        if milestone is None:
            raise HTTPException(status_code=404, detail="Jalon introuvable")
        self._repo.set_milestone_status(milestone, status)
        return self._with_progress(objective)

    def _get(self, tenant_id: uuid.UUID, objective_id: uuid.UUID) -> Objective:
        objective = self._repo.get(tenant_id, objective_id)
        if objective is None:
            raise HTTPException(status_code=404, detail="Objectif introuvable")
        return objective

    def _with_progress(self, objective: Objective) -> ObjectiveProgress:
        milestones = self._repo.list_milestones(objective.id)
        completed = sum(1 for m in milestones if m.status == "done")
        total = len(milestones)
        percent = round(100 * completed / total) if total else 0
        return ObjectiveProgress(
            objective=objective,
            milestones=milestones,
            completed_count=completed,
            total_count=total,
            percent=percent,
        )
