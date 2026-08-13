"""Data access for Objectives/Milestones. Same pattern as the other
repositories in this package — single place issuing SQL for this
concept."""

from __future__ import annotations

import uuid
from typing import List, Optional

from sqlalchemy.orm import Session

from email_analyzer.db.models import Milestone, Objective


class ObjectiveRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def create(
        self,
        tenant_id: uuid.UUID,
        title: str,
        project_id: Optional[uuid.UUID] = None,
        target_date=None,
    ) -> Objective:
        objective = Objective(
            tenant_id=tenant_id, project_id=project_id, title=title, target_date=target_date
        )
        self._db.add(objective)
        self._db.commit()
        self._db.refresh(objective)
        return objective

    def get(self, tenant_id: uuid.UUID, objective_id: uuid.UUID) -> Optional[Objective]:
        return (
            self._db.query(Objective)
            .filter(Objective.id == objective_id, Objective.tenant_id == tenant_id)
            .first()
        )

    def list_for_tenant(
        self, tenant_id: uuid.UUID, project_id: Optional[uuid.UUID] = None
    ) -> List[Objective]:
        query = self._db.query(Objective).filter(Objective.tenant_id == tenant_id)
        if project_id is not None:
            query = query.filter(Objective.project_id == project_id)
        return query.order_by(Objective.target_date.asc().nullslast(), Objective.created_at.desc()).all()

    def update_status(
        self, objective: Objective, status: str, blocking_reason: Optional[str] = None
    ) -> Objective:
        objective.status = status
        objective.blocking_reason = blocking_reason if status == "blocked" else None
        self._db.commit()
        self._db.refresh(objective)
        return objective

    def add_milestone(self, objective_id: uuid.UUID, title: str, due_date=None) -> Milestone:
        milestone = Milestone(objective_id=objective_id, title=title, due_date=due_date)
        self._db.add(milestone)
        self._db.commit()
        self._db.refresh(milestone)
        return milestone

    def list_milestones(self, objective_id: uuid.UUID) -> List[Milestone]:
        return (
            self._db.query(Milestone)
            .filter(Milestone.objective_id == objective_id)
            .order_by(Milestone.due_date.asc().nullslast(), Milestone.created_at.asc())
            .all()
        )

    def get_milestone(self, objective_id: uuid.UUID, milestone_id: uuid.UUID) -> Optional[Milestone]:
        return (
            self._db.query(Milestone)
            .filter(Milestone.id == milestone_id, Milestone.objective_id == objective_id)
            .first()
        )

    def set_milestone_status(self, milestone: Milestone, status: str) -> Milestone:
        milestone.status = status
        self._db.commit()
        self._db.refresh(milestone)
        return milestone
