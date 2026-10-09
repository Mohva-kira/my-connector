"""Candidatures (``applications``), historique (``application_events``) et contacts (``company_contacts``)."""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ..models import Application, ApplicationEvent, CollectedItem, CompanyContact
from . import items as items_repo


def get(session: Session, user_id: UUID, ref: str, *, lock: bool = False) -> tuple[Application, CollectedItem] | None:
    """Candidature d'une offre ``job:<id>`` et son offre ; ``lock`` pose ``FOR UPDATE`` sur la candidature."""
    parts = items_repo.split_ref(ref)
    if parts is None:
        return None
    query = (
        select(Application, CollectedItem)
        .join(CollectedItem, CollectedItem.id == Application.item_id)
        .where(
            Application.user_id == user_id,
            CollectedItem.user_id == user_id,
            CollectedItem.kind == parts[0],
            CollectedItem.external_id == parts[1],
        )
    )
    if lock:
        query = query.with_for_update(of=Application)
    row = session.execute(query).first()
    return (row[0], row[1]) if row else None


def list_all(session: Session, user_id: UUID) -> list[tuple[Application, CollectedItem]]:
    rows = session.execute(
        select(Application, CollectedItem)
        .join(CollectedItem, CollectedItem.id == Application.item_id)
        .where(Application.user_id == user_id, CollectedItem.user_id == user_id)
        .order_by(Application.prepared_at.desc())
    )
    return [(app, item) for app, item in rows]


def ensure_item(session: Session, user_id: UUID, ref: str, *, title: str, company: str, url: str) -> UUID:
    """``id`` de l'offre ; crée une offre minimale si elle est absente de l'historique (dossier hérité)."""
    found = items_repo.find_item(session, user_id, ref)
    if found is not None:
        return found.id
    parts = items_repo.split_ref(ref)
    if parts is None or parts[0] != "job":
        raise ValueError(f"Identifiant d'offre invalide : {ref}")
    return session.execute(
        insert(CollectedItem)
        .values(
            user_id=user_id,
            source_code="linkedin",
            kind="job",
            external_id=parts[1],
            title=title,
            company=company,
            url=url,
            text="",
        )
        .returning(CollectedItem.id)
    ).scalar_one()


def upsert(session: Session, user_id: UUID, item_id: UUID, values: dict[str, Any]) -> UUID:
    """Crée la candidature de l'offre ou met à jour les colonnes fournies ; l'offre sort de la purge (I-10)."""
    statement = insert(Application).values(user_id=user_id, item_id=item_id, **values)
    application_id = session.execute(
        statement.on_conflict_do_update(
            constraint="uq_applications_user_id_item_id", set_={**values, "updated_at": func.now()}
        ).returning(Application.id)
    ).scalar_one()
    session.execute(
        update(CollectedItem)
        .where(CollectedItem.user_id == user_id, CollectedItem.id == item_id)
        .values(purge_after=None)
    )
    return application_id


def update_fields(session: Session, user_id: UUID, application_id: UUID, values: dict[str, Any]) -> None:
    session.execute(
        update(Application)
        .where(Application.user_id == user_id, Application.id == application_id)
        .values(**values, updated_at=func.now())
    )


def events(session: Session, user_id: UUID, application_ids: list[UUID]) -> dict[UUID, list[ApplicationEvent]]:
    found: dict[UUID, list[ApplicationEvent]] = {i: [] for i in application_ids}
    if not application_ids:
        return found
    rows = session.execute(
        select(ApplicationEvent)
        .join(Application, Application.id == ApplicationEvent.application_id)
        .where(Application.user_id == user_id, ApplicationEvent.application_id.in_(application_ids))
        .order_by(ApplicationEvent.id)
    ).scalars()
    for event in rows:
        found[event.application_id].append(event)
    return found


def add_event(
    session: Session, user_id: UUID, application_id: UUID, kind: str, details: dict[str, Any], at: datetime | None
) -> None:
    owned = session.execute(
        select(Application.id).where(Application.user_id == user_id, Application.id == application_id)
    ).scalar_one_or_none()
    if owned is None:
        raise LookupError("Candidature introuvable.")
    values: dict[str, Any] = {"application_id": application_id, "kind": kind, "details": details}
    if at is not None:
        values["at"] = at
    session.execute(insert(ApplicationEvent).values(**values))


def contacts(session: Session, user_id: UUID, application_ids: list[UUID]) -> dict[UUID, list[CompanyContact]]:
    found: dict[UUID, list[CompanyContact]] = {i: [] for i in application_ids}
    if not application_ids:
        return found
    rows = session.execute(
        select(CompanyContact)
        .where(CompanyContact.user_id == user_id, CompanyContact.application_id.in_(application_ids))
        .order_by(CompanyContact.collected_at, CompanyContact.id)
    ).scalars()
    for contact in rows:
        found[contact.application_id].append(contact)
    return found


def replace_contacts(session: Session, user_id: UUID, application_id: UUID, rows: list[dict[str, Any]]) -> None:
    """Remplace la liste des contacts. Un contact déjà connu (même URL de profil, sinon même email) est mis à jour
    en place : il garde son ``id`` (messages liés), sa date de collecte et sa date de purge (I-14)."""
    existing = contacts(session, user_id, [application_id])[application_id]

    def key(profile_url: str | None, email: str | None) -> str:
        return f"url:{profile_url}" if profile_url else f"mail:{(email or '').lower()}"

    by_key = {key(c.profile_url, c.email): c for c in existing}
    kept: set[UUID] = set()
    for row in rows:
        previous = by_key.get(key(row.get("profile_url"), row.get("email")))
        if previous is not None and previous.id not in kept:
            kept.add(previous.id)
            session.execute(
                update(CompanyContact)
                .where(CompanyContact.user_id == user_id, CompanyContact.id == previous.id)
                .values(**row)
            )
        else:
            session.execute(
                insert(CompanyContact).values(
                    **row, user_id=user_id, application_id=application_id, legal_basis="interet_legitime"
                )
            )
    stale = [c.id for c in existing if c.id not in kept]
    if stale:
        session.execute(delete(CompanyContact).where(CompanyContact.user_id == user_id, CompanyContact.id.in_(stale)))
