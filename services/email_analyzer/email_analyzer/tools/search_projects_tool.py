"""SearchProjectsTool — list/search a tenant's projects, optionally
filtered by a keyword in the project name or its rules_matrix (sender
domain, client name...). Read-only, no LLM call."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from email_analyzer.classification import ProjectRules, rules_matrix_field_matches
from email_analyzer.db.models import Project, ProjectSummary


class SearchProjectsTool:
    name = "search_projects"
    description = (
        "Recherche les projets du tenant, optionnellement filtrés par un mot-clé "
        "présent dans le nom du projet ou dans ses règles de rattachement (domaine "
        "expéditeur, client, société...). Retourne nom, statut, sentiment et résumé "
        "courant pour chaque projet trouvé."
    )

    def __init__(self, db: Session) -> None:
        self._db = db

    def run(
        self, tenant_id: uuid.UUID, keyword: Optional[str] = None, limit: int = 20
    ) -> List[Dict[str, Any]]:
        query = self._db.query(Project).filter(Project.tenant_id == tenant_id)
        if keyword:
            # Filtrage en Python plutôt qu'en SQL : rules_matrix (JSONB) n'a pas
            # d'opérateur ILIKE natif ici, et le volume par tenant (dizaines à
            # quelques centaines de projets) rend le coût négligeable — voir
            # rules_matrix_field_matches (classification.py).
            needle = keyword.strip().lower()
            all_projects = query.all()
            matched = [
                p
                for p in all_projects
                if needle in p.name.lower()
                or rules_matrix_field_matches(ProjectRules.from_dict(p.rules_matrix), needle, exact=False)
            ]
            projects = sorted(matched, key=lambda p: p.name.lower())[:limit]
        else:
            projects = query.order_by(Project.name.asc()).limit(limit).all()

        results: List[Dict[str, Any]] = []
        for p in projects:
            summary = (
                self._db.query(ProjectSummary).filter(ProjectSummary.project_id == p.id).first()
            )
            results.append(
                {
                    "project_id": str(p.id),
                    "name": p.name,
                    "status": p.status,
                    "sentiment": summary.sentiment if summary else None,
                    "summary": summary.content if summary else None,
                }
            )
        return results
