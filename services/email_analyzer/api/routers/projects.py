"""CRUD minimal pour les projets persistés (Unit 10/11).

Sans ces endpoints, aucun `Project` ne peut jamais exister en base : le
Fast-Track (`POST /api/projects/{id}/refresh`, `analysis_tasks.py`) et la
persistance Unit 10 restent construits mais inaccessibles. Ce module ne fait
que créer/lister/consulter des projets manuellement — le clustering
automatique à l'onboarding (`evolution-plan.md` Phase 1, Historical
Clustering Worker) reste une unité séparée, non implémentée ici.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from email_analyzer.db.models import (
    Appointment,
    Email,
    Project,
    ProjectStatus,
    ProjectSummary,
    SuggestedAction,
)
from email_analyzer.db.session import get_db
from email_analyzer.saas_logic import authenticate_bearer, saas_enabled

router = APIRouter(prefix="/api/projects", tags=["projects"])


class CreateProjectBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    rules_matrix: Optional[dict] = None


class UpdateProjectBody(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    rules_matrix: Optional[dict] = None


class SuggestedActionOut(BaseModel):
    id: uuid.UUID
    description: str
    deadline: Optional[datetime]
    status: str
    created_at: datetime
    rationale: Optional[str]
    stakeholder: Optional[str]
    advice: Optional[str]


class EmailOut(BaseModel):
    id: uuid.UUID
    subject: Optional[str]
    received_at: Optional[datetime]
    recipient_status: str
    # 0-100, voir ai_intelligent.score_email_importance ; NULL pour les emails
    # persistés avant l'ajout de cette colonne (migration 004).
    importance_score: Optional[int]
    # Tags métier + priorité (email_analyzer.classification.derive_tags) et
    # score de confiance de classification (migration 005) ; NULL/vides pour
    # les emails persistés avant cette colonne.
    tags: Optional[List[str]]
    classification_score: Optional[int]


class ProjectOut(BaseModel):
    id: uuid.UUID
    name: str
    status: str
    created_at: datetime
    updated_at: datetime
    summary_content: Optional[str]
    sentiment: Optional[str]
    # Décisions/risques/échéances/prochaines étapes structurés (voir
    # email_analyzer.llm.ProjectSummaryLLM, ProjectSummary.structured_content) —
    # additif, None si le projet n'a pas encore été rafraîchi avec l'extraction
    # structurée ou si elle a échoué pour ce refresh.
    structured_content: Optional[dict]
    last_processed_email_timestamp: Optional[datetime]
    # "Outstanding actions" sur la carte projet (architecture.md, Project Hub) :
    # nombre d'actions suggérées encore `pending` (ni complétées ni écartées).
    pending_actions_count: int
    # Règles de classification multi-critères (email_analyzer.classification) :
    # keywords/sender_domains/sender_emails/client_names/company_names/
    # reference_numbers. Exposé en lecture pour préremplir l'éditeur frontend
    # (auparavant write-only via POST).
    rules_matrix: Optional[dict]


class ProjectDetailOut(ProjectOut):
    suggested_actions: List[SuggestedActionOut]
    # Les 10 emails les plus importants du projet (tri par importance_score,
    # cf. progress-tracker.md Unit 19) — permet un tri côté frontend sans
    # endpoint de liste d'emails paginé séparé (hors périmètre de cette unité).
    top_emails: List[EmailOut]


class DuplicateProjectOut(BaseModel):
    id: uuid.UUID
    name: str
    email_count: int
    updated_at: datetime


class DuplicateGroupOut(BaseModel):
    # Domaines expéditeurs (rules_matrix.sender_domains) communs à tous les
    # projets du groupe — seul signal de dédup retenu (cf. précédent Unit 30 :
    # client_names/company_names jugés peu fiables pour ce genre de filtre).
    shared_domains: List[str]
    projects: List[DuplicateProjectOut]


class MergeProjectsBody(BaseModel):
    source_ids: List[uuid.UUID] = Field(..., min_length=1)


_RULES_MATRIX_LIST_KEYS = (
    "keywords",
    "sender_domains",
    "sender_emails",
    "client_names",
    "company_names",
    "reference_numbers",
)


def _merge_rules_matrix(rules_list: List[Optional[dict]]) -> dict:
    """Union dédupliquée (insensible à la casse) des listes de
    ``rules_matrix`` fournies, dans l'ordre de première apparition."""
    merged: Dict[str, List[str]] = {key: [] for key in _RULES_MATRIX_LIST_KEYS}
    for rules in rules_list:
        if not isinstance(rules, dict):
            continue
        for key in _RULES_MATRIX_LIST_KEYS:
            values = rules.get(key)
            if not isinstance(values, list):
                continue
            for value in values:
                normalized = str(value).strip()
                if not normalized:
                    continue
                if normalized.lower() not in {v.lower() for v in merged[key]}:
                    merged[key].append(normalized)
    return {key: values for key, values in merged.items() if values}


def _require_saas() -> None:
    if not saas_enabled():
        raise HTTPException(status_code=503, detail="Mode SaaS non activé")


def _pending_actions_count(db: Session, project_id: uuid.UUID) -> int:
    return (
        db.query(SuggestedAction)
        .filter(SuggestedAction.project_id == project_id, SuggestedAction.status == "pending")
        .count()
    )


def _project_out(
    db: Session, project: Project, summary: Optional[ProjectSummary]
) -> ProjectOut:
    return ProjectOut(
        id=project.id,
        name=project.name,
        status=project.status,
        created_at=project.created_at,
        updated_at=project.updated_at,
        summary_content=summary.content if summary else None,
        sentiment=summary.sentiment if summary else None,
        structured_content=summary.structured_content if summary else None,
        last_processed_email_timestamp=(
            summary.last_processed_email_timestamp if summary else None
        ),
        pending_actions_count=_pending_actions_count(db, project.id),
        rules_matrix=project.rules_matrix,
    )


@router.post("", response_model=ProjectOut, status_code=201)
def create_project(
    body: CreateProjectBody,
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> ProjectOut:
    _require_saas()
    _, tenant, _ = authenticate_bearer(db, authorization)
    project = Project(tenant_id=tenant.id, name=body.name.strip(), rules_matrix=body.rules_matrix)
    db.add(project)
    db.commit()
    db.refresh(project)
    return _project_out(db, project, None)


@router.get("", response_model=List[ProjectOut])
def list_projects(
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> List[ProjectOut]:
    _require_saas()
    _, tenant, _ = authenticate_bearer(db, authorization)
    projects = (
        db.query(Project)
        .filter(Project.tenant_id == tenant.id)
        .order_by(Project.created_at.desc())
        .all()
    )
    return [_project_out(db, p, p.summary) for p in projects]


@router.get("/duplicates", response_model=List[DuplicateGroupOut])
def list_duplicate_projects(
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> List[DuplicateGroupOut]:
    """Regroupe les projets actifs du tenant dont ``rules_matrix.sender_domains``
    se chevauche (déclaré avant ``/{project_id}`` : un segment statique doit
    être enregistré avant un paramètre de chemin pour ne pas être intercepté
    par lui). Un projet partageant des domaines avec plusieurs autres est
    fusionné dans une seule composante connexe plutôt que dupliqué entre
    plusieurs groupes."""
    _require_saas()
    _, tenant, _ = authenticate_bearer(db, authorization)

    projects = (
        db.query(Project)
        .filter(Project.tenant_id == tenant.id, Project.status == ProjectStatus.active.value)
        .all()
    )

    domain_groups: Dict[str, List[Project]] = {}
    for project in projects:
        domains = (project.rules_matrix or {}).get("sender_domains") or []
        for domain in domains:
            normalized = str(domain).strip().lower()
            if normalized:
                domain_groups.setdefault(normalized, []).append(project)
    domain_groups = {d: ps for d, ps in domain_groups.items() if len(ps) > 1}
    if not domain_groups:
        return []

    # Union-find sur les project_id partageant un domaine, pour qu'un projet
    # à cheval sur plusieurs domaines n'apparaisse que dans un seul groupe.
    parent: Dict[uuid.UUID, uuid.UUID] = {}

    def find(pid: uuid.UUID) -> uuid.UUID:
        parent.setdefault(pid, pid)
        while parent[pid] != pid:
            parent[pid] = parent[parent[pid]]
            pid = parent[pid]
        return pid

    def union(a: uuid.UUID, b: uuid.UUID) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for ps in domain_groups.values():
        for p in ps[1:]:
            union(ps[0].id, p.id)

    projects_by_id = {p.id: p for ps in domain_groups.values() for p in ps}
    email_counts = dict(
        db.query(Email.project_id, func.count(Email.id))
        .filter(Email.project_id.in_(projects_by_id.keys()))
        .group_by(Email.project_id)
        .all()
    )

    components: Dict[uuid.UUID, List[uuid.UUID]] = {}
    for pid in projects_by_id:
        components.setdefault(find(pid), []).append(pid)

    result: List[DuplicateGroupOut] = []
    for root, pids in components.items():
        shared_domains = sorted(d for d, ps in domain_groups.items() if find(ps[0].id) == root)
        result.append(
            DuplicateGroupOut(
                shared_domains=shared_domains,
                projects=[
                    DuplicateProjectOut(
                        id=pid,
                        name=projects_by_id[pid].name,
                        email_count=email_counts.get(pid, 0),
                        updated_at=projects_by_id[pid].updated_at,
                    )
                    for pid in pids
                ],
            )
        )
    return result


@router.get("/{project_id}", response_model=ProjectDetailOut)
def get_project(
    project_id: uuid.UUID,
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> ProjectDetailOut:
    _require_saas()
    _, tenant, _ = authenticate_bearer(db, authorization)
    project = (
        db.query(Project)
        .filter(Project.id == project_id, Project.tenant_id == tenant.id)
        .first()
    )
    if project is None:
        raise HTTPException(status_code=404, detail="Projet introuvable")

    actions = (
        db.query(SuggestedAction)
        .filter(SuggestedAction.project_id == project.id)
        .order_by(SuggestedAction.created_at.desc())
        .all()
    )
    top_emails = (
        db.query(Email)
        .filter(Email.project_id == project.id)
        .order_by(Email.importance_score.desc().nullslast(), Email.received_at.desc())
        .limit(10)
        .all()
    )
    base = _project_out(db, project, project.summary)
    return ProjectDetailOut(
        **base.model_dump(),
        suggested_actions=[
            SuggestedActionOut(
                id=a.id,
                description=a.description,
                deadline=a.deadline,
                status=a.status,
                created_at=a.created_at,
                rationale=a.rationale,
                stakeholder=a.stakeholder,
                advice=a.advice,
            )
            for a in actions
        ],
        top_emails=[
            EmailOut(
                id=e.id,
                subject=e.subject,
                received_at=e.received_at,
                recipient_status=e.recipient_status,
                importance_score=e.importance_score,
                tags=e.tags,
                classification_score=e.classification_score,
            )
            for e in top_emails
        ],
    )


@router.post("/{project_id}/merge", response_model=ProjectDetailOut)
def merge_projects(
    project_id: uuid.UUID,
    body: MergeProjectsBody,
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> ProjectDetailOut:
    """Fusionne un ou plusieurs projets doublons (``body.source_ids``) dans
    ``project_id`` (la cible) : réassigne Email/SuggestedAction/Appointment,
    fusionne ``rules_matrix`` (union dédupliquée), supprime les projets
    sources (et leur ``ProjectSummary``, contrainte unique sur ``project_id``).

    Aucune réassignation d'``Email`` ne peut entrer en collision : la
    contrainte ``uq_email_tenant_external_id`` garantit qu'un email n'est déjà
    rattaché qu'à un seul projet (invariant 1:1, architecture.md). La
    réassignation de ``SuggestedAction``/``Appointment`` est sûre pour la même
    raison que leur wipe/rebuild à chaque Fast-Track (``_run_fasttrack_sync``) :
    le prochain rafraîchissement de la cible les régénère de toute façon.

    Ne déclenche **pas** de rafraîchissement automatique (pas d'appel LLM
    caché) — le résumé de la cible reste celui d'avant fusion tant que
    l'utilisateur ne relance pas un Fast-Track manuellement, idéalement avec
    ``force_days`` (le curseur ``last_processed_email_timestamp`` de la cible
    peut être postérieur à des emails plus anciens réassignés depuis une
    source, qu'un simple refresh delta ne verrait pas)."""
    _require_saas()
    _, tenant, _ = authenticate_bearer(db, authorization)

    source_ids = list(dict.fromkeys(body.source_ids))
    if project_id in source_ids:
        raise HTTPException(status_code=400, detail="Un projet ne peut pas être fusionné avec lui-même")

    target = (
        db.query(Project)
        .filter(Project.id == project_id, Project.tenant_id == tenant.id)
        .first()
    )
    if target is None:
        raise HTTPException(status_code=404, detail="Projet cible introuvable")

    sources = (
        db.query(Project)
        .filter(Project.id.in_(source_ids), Project.tenant_id == tenant.id)
        .all()
    )
    if len(sources) != len(source_ids):
        raise HTTPException(status_code=404, detail="Un ou plusieurs projets source introuvables")

    target.rules_matrix = _merge_rules_matrix([target.rules_matrix] + [s.rules_matrix for s in sources])

    db.query(Email).filter(Email.project_id.in_(source_ids)).update(
        {"project_id": target.id}, synchronize_session=False
    )
    db.query(SuggestedAction).filter(SuggestedAction.project_id.in_(source_ids)).update(
        {"project_id": target.id}, synchronize_session=False
    )
    db.query(Appointment).filter(Appointment.project_id.in_(source_ids)).update(
        {"project_id": target.id}, synchronize_session=False
    )
    db.query(ProjectSummary).filter(ProjectSummary.project_id.in_(source_ids)).delete(
        synchronize_session=False
    )
    for source in sources:
        db.delete(source)

    db.commit()
    return get_project(project_id, db=db, authorization=authorization)


@router.patch("/{project_id}", response_model=ProjectOut)
def update_project(
    project_id: uuid.UUID,
    body: UpdateProjectBody,
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> ProjectOut:
    _require_saas()
    _, tenant, _ = authenticate_bearer(db, authorization)
    project = (
        db.query(Project)
        .filter(Project.id == project_id, Project.tenant_id == tenant.id)
        .first()
    )
    if project is None:
        raise HTTPException(status_code=404, detail="Projet introuvable")
    if body.name is not None:
        project.name = body.name.strip()
    if body.rules_matrix is not None:
        project.rules_matrix = body.rules_matrix
    db.commit()
    db.refresh(project)
    return _project_out(db, project, project.summary)
