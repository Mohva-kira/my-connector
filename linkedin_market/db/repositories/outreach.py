"""Messages à l'entreprise (``outreach_messages``) : réservation du plafond quotidien, puis issue de l'envoi.

Réservation (``backend/03`` § 5.1, E-11) : dans **une** transaction, ``FOR UPDATE`` sur la candidature puis sur
``user_settings`` (sérialise le compteur entre candidatures), comptage du jour, insertion ``status='confirmé'``.
L'envoi (SMTP, LinkedIn) a lieu hors transaction ; une seconde transaction écrit ``envoyé`` ou ``échec``. Une ligne
``confirmé`` sans ``sent_at`` laissée par un arrêt est « à vérifier », jamais renvoyée automatiquement.
La ``suppression_list`` (I-13) s'ajoute à cette transaction en vague 4.
"""

from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ..models import CompanyContact, OutreachMessage
from . import activity as activity_repo

# Événements d'activité des messages envoyés avant la réservation en base (comptés pour la journée de transition).
MESSAGE_ACTIVITY_KINDS = ("message_email", "message_linkedin")
CONTACT_RETENTION = timedelta(days=3 * 365)  # I-14 : 3 ans après le dernier contact


def count_day(session: Session, user_id: UUID, start: datetime, end: datetime) -> int:
    """Messages qui occupent le plafond du jour : réservés ou envoyés (``échec`` libère la place).

    Les messages envoyés avant l'existence des réservations ne sont connus que par ``activity_events`` : on retient
    le plus grand des deux compteurs (chaque envoi réservé produit aussi un événement, jamais l'inverse).
    """
    reserved = session.execute(
        select(func.count())
        .select_from(OutreachMessage)
        .where(
            OutreachMessage.user_id == user_id,
            OutreachMessage.status != "échec",
            OutreachMessage.confirmed_at >= start,
            OutreachMessage.confirmed_at < end,
        )
    ).scalar_one()
    recorded = activity_repo.count_between(session, user_id, MESSAGE_ACTIVITY_KINDS, start, end)
    return max(reserved, recorded)


def find_contact(session: Session, user_id: UUID, application_id: UUID, recipient: str) -> UUID | None:
    """Contact de la candidature correspondant au destinataire (email sans casse, ou URL de profil)."""
    recipient = recipient.strip()
    url = recipient.split("?", 1)[0].rstrip("/")
    return session.execute(
        select(CompanyContact.id)
        .where(
            CompanyContact.user_id == user_id,
            CompanyContact.application_id == application_id,
            or_(
                func.lower(CompanyContact.email) == recipient.lower(),
                func.rtrim(CompanyContact.profile_url, "/") == url,
            ),
        )
        .order_by(CompanyContact.collected_at)
        .limit(1)
    ).scalar_one_or_none()


def insert_confirmed(session: Session, user_id: UUID, application_id: UUID, values: dict[str, Any]) -> UUID:
    """Réservation ``confirmé`` (``channel``, ``recipient``, ``subject``, ``body``, ``cv_attached``, ``contact_id``)."""
    return session.execute(
        insert(OutreachMessage)
        .values(**values, user_id=user_id, application_id=application_id, status="confirmé")
        .returning(OutreachMessage.id)
    ).scalar_one()


def get(session: Session, user_id: UUID, message_id: UUID, *, lock: bool = False) -> OutreachMessage | None:
    query = select(OutreachMessage).where(OutreachMessage.user_id == user_id, OutreachMessage.id == message_id)
    if lock:
        query = query.with_for_update()
    return session.execute(query).scalar_one_or_none()


def mark_sent(session: Session, user_id: UUID, message_id: UUID, smtp_message_id: str | None) -> None:
    """``envoyé`` + date ; le contact lié repousse sa date de purge à 3 ans (I-14)."""
    contact_id = session.execute(
        update(OutreachMessage)
        .where(OutreachMessage.user_id == user_id, OutreachMessage.id == message_id)
        .values(status="envoyé", sent_at=func.now(), smtp_message_id=smtp_message_id, error=None)
        .returning(OutreachMessage.contact_id)
    ).scalar_one_or_none()
    if contact_id is not None:
        session.execute(
            update(CompanyContact)
            .where(CompanyContact.user_id == user_id, CompanyContact.id == contact_id)
            .values(last_contacted_at=func.now(), purge_after=func.now() + CONTACT_RETENTION)
        )


def mark_failed(session: Session, user_id: UUID, message_id: UUID, error: str) -> None:
    """``échec`` : la place du plafond est libérée ; aucun renvoi automatique."""
    session.execute(
        update(OutreachMessage)
        .where(OutreachMessage.user_id == user_id, OutreachMessage.id == message_id, OutreachMessage.status != "envoyé")
        .values(status="échec", error=error[:1000])
    )


def note_uncertain(session: Session, user_id: UUID, message_id: UUID, error: str) -> None:
    """Issue inconnue (envoi peut-être parti) : reste ``confirmé`` (« à vérifier ») et compte dans le plafond."""
    session.execute(
        update(OutreachMessage)
        .where(OutreachMessage.user_id == user_id, OutreachMessage.id == message_id)
        .values(error=error[:1000])
    )


def release(session: Session, user_id: UUID, message_id: UUID) -> None:
    """Annule une réservation refusée avant toute tâche (conflit détecté après la réservation)."""
    session.execute(
        delete(OutreachMessage).where(
            OutreachMessage.user_id == user_id,
            OutreachMessage.id == message_id,
            OutreachMessage.status == "confirmé",
            OutreachMessage.sent_at.is_(None),
        )
    )
