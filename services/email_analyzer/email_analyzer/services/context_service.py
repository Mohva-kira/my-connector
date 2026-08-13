"""Context Engine foundation (Personal Executive AI Assistant pivot, Phase 7).

Single place that assembles the unified AI context (projects, memory,
knowledge) into the system prompt used by the LLM. Per the pivot plan this
should become "the only context provider for LLM requests" — a first step
towards that: ConversationService now delegates its prompt-building to
this service instead of doing it inline (see Unit 45/46/47 for the
pieces being unified here).

v1 scope: Projects + Memory + Knowledge. Profile (Contexte Personnel,
2026-07-30) added on top without changing this v1 boundary otherwise.
Conversation history is handled
separately (it's the message thread, not "context" text). Emails and
Actions are not added here yet — Actions already appear inside the
projects context (pending_action_descriptions, unchanged from before this
unit); Emails have no dedicated read model here, so adding them now would
mean inventing data that doesn't exist (contrary to ai-workflow-rules.md's
"no speculative logic"). Documents (2026-08-01) get an ambient *count* per
project (`document_count`, mirrors `open_ticket_count`) — enough for the
model to know documents exist and decide to call `search_documents` — but
never their extracted text, which stays tool-call-only (a document can be
up to 200k chars of extracted text; inlining that ambiently would blow the
prompt budget for no guaranteed benefit).

Deliberately does NOT trigger memory extraction or knowledge sync — it
only reads what has already been extracted/synced explicitly (Units 46,
47). Whether that becomes automatic is a product decision for a
dedicated session, not implied by "build the context engine".
"""

from __future__ import annotations

import uuid
from typing import Dict, List

from sqlalchemy import func
from sqlalchemy.orm import Session

from email_analyzer.db.models import Document, Email, Project, ProjectSummary, SuggestedAction
from email_analyzer.llm import (
    build_knowledge_context,
    build_memory_context,
    build_objectives_context,
    build_portfolio_chat_system_prompt,
    build_portfolio_context,
    build_profile_context,
)
from email_analyzer.services.knowledge_service import KnowledgeService
from email_analyzer.services.memory_service import MemoryService
from email_analyzer.services.objective_service import ObjectiveService
from email_analyzer.services.profile_service import ProfileService


# Bornes sur le contexte assemblé pour le chat interactif — sans elles, le
# nombre de requêtes DB (N+1 sur les projets) et la taille du system prompt
# (donc le temps de génération LLM) croissent sans limite avec le nombre de
# projets/actions du tenant. Le N+1 lui-même n'est pas éliminé ici (hors
# périmètre du budget de latence), seulement borné.
_MAX_PROJECTS_IN_CONTEXT = 15
_MAX_PENDING_ACTIONS_PER_PROJECT = 5
_MAX_OBJECTIVES_IN_CONTEXT = 10


class ContextService:
    def __init__(self, db: Session) -> None:
        self._db = db

    def build_portfolio_system_prompt(self, tenant_id: uuid.UUID, user_id: uuid.UUID) -> str:
        projects_context = self._build_projects_context(tenant_id)
        extra_sections = [
            self._build_profile_section(tenant_id, user_id),
            self._build_memory_section(tenant_id, user_id),
            self._build_knowledge_section(tenant_id),
            self._build_objectives_section(tenant_id),
        ]
        return build_portfolio_chat_system_prompt(projects_context, extra_sections)

    def quick_fallback_summary(self, tenant_id: uuid.UUID) -> str:
        """Résumé déterministe (pas de LLM) à partir des données déjà
        disponibles — utilisé par ConversationService quand l'appel LLM
        interactif dépasse son budget de latence, pour répondre quand même
        "selon les données disponibles" plutôt que de laisser l'utilisateur
        sans réponse."""
        rows = self._project_rows(tenant_id)
        if not rows:
            return "Je n'ai pas encore de projet analysé pour votre compte."

        lines = ["Voici ce que je sais déjà (données de la dernière analyse) :"]
        for row in rows:
            sentiment = f" ({row['sentiment']})" if row["sentiment"] else ""
            lines.append(f"- {row['name']}{sentiment}")
            for action in row["pending_action_descriptions"][:3]:
                lines.append(f"  · À faire : {action}")
        return "\n".join(lines)

    def _build_projects_context(self, tenant_id: uuid.UUID) -> str:
        total_count = self._db.query(Project).filter(Project.tenant_id == tenant_id).count()
        return build_portfolio_context(self._project_rows(tenant_id), total_count=total_count)

    def _project_rows(self, tenant_id: uuid.UUID) -> List[dict]:
        projects = (
            self._db.query(Project)
            .filter(Project.tenant_id == tenant_id)
            .order_by(Project.updated_at.desc())
            .limit(_MAX_PROJECTS_IN_CONTEXT)
            .all()
        )
        ticket_counts = self._open_ticket_counts([p.id for p in projects])
        document_counts = self._document_counts([p.id for p in projects])
        rows = []
        for p in projects:
            summary = self._db.query(ProjectSummary).filter(ProjectSummary.project_id == p.id).first()
            rows.append(
                {
                    "name": p.name,
                    "sentiment": summary.sentiment if summary else None,
                    "summary_content": summary.content if summary else None,
                    "pending_action_descriptions": self._pending_action_descriptions(p.id),
                    "open_ticket_count": ticket_counts.get(p.id, 0),
                    "document_count": document_counts.get(p.id, 0),
                }
            )
        return rows

    def _open_ticket_counts(self, project_ids: List[uuid.UUID]) -> Dict[uuid.UUID, int]:
        """Nombre de tickets distincts détectés dans les sujets (voir
        email_analyzer.subject_tickets, Email.ticket_label/ticket_number) par
        projet — une seule requête groupée (pas de N+1) pour garder le modèle
        proactivement informé du volume de réclamations à chaque tour de chat
        (même principe que pending_action_descriptions ci-dessus)."""
        if not project_ids:
            return {}
        rows = (
            self._db.query(Email.project_id, func.count(func.distinct(Email.ticket_number)))
            .filter(Email.project_id.in_(project_ids), Email.ticket_label.isnot(None))
            .group_by(Email.project_id)
            .all()
        )
        return dict(rows)

    def _document_counts(self, project_ids: List[uuid.UUID]) -> Dict[uuid.UUID, int]:
        """Nombre de documents disponibles (upload manuel ou pièce jointe
        d'email ingérée automatiquement, voir analysis_tasks.py::
        _ingest_new_email_attachments) par projet — une seule requête groupée,
        même principe que _open_ticket_counts. Ambiant seulement (le modèle
        sait qu'il existe N documents) : le contenu lui-même reste accessible
        à la demande via l'outil `search_documents`, pas injecté ici (voir
        context_service.py, docstring de module)."""
        if not project_ids:
            return {}
        rows = (
            self._db.query(Document.project_id, func.count(Document.id))
            .filter(Document.project_id.in_(project_ids))
            .group_by(Document.project_id)
            .all()
        )
        return dict(rows)

    def _pending_action_descriptions(self, project_id: uuid.UUID) -> List[str]:
        rows = (
            self._db.query(SuggestedAction)
            .filter(SuggestedAction.project_id == project_id, SuggestedAction.status == "pending")
            .order_by(SuggestedAction.deadline.asc().nullslast(), SuggestedAction.created_at.desc())
            .limit(_MAX_PENDING_ACTIONS_PER_PROJECT)
            .all()
        )
        return [a.description for a in rows]

    def _build_profile_section(self, tenant_id: uuid.UUID, user_id: uuid.UUID) -> str:
        profile = ProfileService(self._db).get_profile(tenant_id, user_id)
        if profile is None:
            return ""
        return build_profile_context(
            {
                "display_name": profile.display_name,
                "languages": profile.languages,
                "timezone": profile.timezone,
                "location": profile.location,
                "work_start_time": profile.work_start_time.strftime("%Hh%M") if profile.work_start_time else None,
                "work_end_time": profile.work_end_time.strftime("%Hh%M") if profile.work_end_time else None,
                "wake_time": profile.wake_time.strftime("%Hh%M") if profile.wake_time else None,
                "sleep_time": profile.sleep_time.strftime("%Hh%M") if profile.sleep_time else None,
                "sport_days": profile.sport_days,
                "focus_hours": profile.focus_hours,
                "family_members": profile.family_members,
                "preferences": profile.preferences,
            }
        )

    def _build_memory_section(self, tenant_id: uuid.UUID, user_id: uuid.UUID) -> str:
        memories = MemoryService(self._db).list_memories(tenant_id, user_id)
        return build_memory_context(
            [{"category": m.category, "content": m.content} for m in memories]
        )

    def _build_knowledge_section(self, tenant_id: uuid.UUID) -> str:
        service = KnowledgeService(self._db)
        contacts = service.list_contacts(tenant_id)
        orgs_by_id = {o.id: o.name for o in service.list_organizations(tenant_id)}
        return build_knowledge_context(
            [
                {"name": c.name, "organization": orgs_by_id.get(c.organization_id)}
                for c in contacts
            ]
        )

    def _build_objectives_section(self, tenant_id: uuid.UUID) -> str:
        """Objectifs suivis par le tenant (Objective/Milestone, domaine
        "Contexte Objectifs") — jusqu'ici absents de tout contexte ambiant ou
        tool du chat portefeuille malgré une UI/CRUD complets depuis les
        Units 54-56 ; l'assistant ne pouvait donc jamais en parler. Objectifs
        bloqués triés en premier, puis par échéance la plus proche, bornés
        par ``_MAX_OBJECTIVES_IN_CONTEXT`` (même logique de budget de
        latence que les projets/actions ci-dessus)."""
        progresses = ObjectiveService(self._db).list_objectives(tenant_id)
        progresses.sort(
            key=lambda p: (
                p.objective.status != "blocked",
                p.objective.target_date is None,
                p.objective.target_date,
            )
        )
        rows = []
        for p in progresses[:_MAX_OBJECTIVES_IN_CONTEXT]:
            next_milestone = next((m for m in p.milestones if m.status != "done"), None)
            rows.append(
                {
                    "title": p.objective.title,
                    "status": p.objective.status,
                    "blocking_reason": p.objective.blocking_reason,
                    "target_date": p.objective.target_date.strftime("%d/%m/%Y")
                    if p.objective.target_date
                    else None,
                    "percent": p.percent,
                    "next_milestone": next_milestone.title if next_milestone else None,
                }
            )
        return build_objectives_context(rows)
