"""ListObjectivesTool — objectifs personnels/professionnels (Objective/
Milestone, domaine "Contexte Objectifs"), avec leur progression et leurs
jalons en attente. Read-only, no LLM call — même discipline que
``list_tickets_tool.py``/``list_meetings_tool.py``. Contrairement au
résumé ambiant injecté par ``ContextService`` (borné, sans le détail des
jalons), ce tool retourne le détail complet d'un ou plusieurs objectifs sur
demande explicite du LLM."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from email_analyzer.services.objective_service import ObjectiveService


class ListObjectivesTool:
    name = "list_objectives"
    description = (
        "Liste les objectifs personnels/professionnels suivis par l'utilisateur, "
        "avec leur statut (sur la bonne voie / bloqué et pourquoi), leur progression "
        "et le détail de leurs jalons — optionnellement filtrés par projet."
    )

    def __init__(self, db: Session) -> None:
        self._db = db

    def run(self, tenant_id: uuid.UUID, project_id: Optional[uuid.UUID] = None) -> List[Dict[str, Any]]:
        progresses = ObjectiveService(self._db).list_objectives(tenant_id, project_id=project_id)
        return [
            {
                "objective_id": str(p.objective.id),
                "title": p.objective.title,
                "status": p.objective.status,
                "blocking_reason": p.objective.blocking_reason,
                "target_date": p.objective.target_date.isoformat() if p.objective.target_date else None,
                "percent": p.percent,
                "milestones": [
                    {
                        "title": m.title,
                        "status": m.status,
                        "due_date": m.due_date.isoformat() if m.due_date else None,
                    }
                    for m in p.milestones
                ],
            }
            for p in progresses
        ]
