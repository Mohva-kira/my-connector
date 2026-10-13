"""Purge de rétention (I-10, I-14) : suppressions par table, toutes filtrées par ``user_id`` (I-01).

Jamais purgés : ``suppression_list`` (empreintes, I-13), ``reports`` (immuables, I-02), ``applications``. Les
``labels`` suivent leur élément (``ON DELETE CASCADE``).
"""

from uuid import UUID

from sqlalchemy import delete, exists, func
from sqlalchemy.orm import Session

from ..models import (
    FINAL_TASK_STATUSES,
    ActivityEvent,
    Application,
    CollectedItem,
    CompanyContact,
    PlatformSession,
    Task,
)


def delete_expired_items(session: Session, user_id: UUID) -> int:
    """Éléments dont ``purge_after`` est dépassé et qu'aucune candidature ne référence (FK RESTRICT)."""
    has_application = exists().where(Application.item_id == CollectedItem.id, Application.user_id == user_id)
    result = session.execute(
        delete(CollectedItem).where(
            CollectedItem.user_id == user_id, CollectedItem.purge_after < func.now(), ~has_application
        )
    )
    return result.rowcount


def delete_expired_contacts(session: Session, user_id: UUID) -> int:
    """Contacts d'entreprise dont ``purge_after`` (dernier contact + 3 ans, SQL) est dépassé (I-14)."""
    result = session.execute(
        delete(CompanyContact).where(CompanyContact.user_id == user_id, CompanyContact.purge_after < func.now())
    )
    return result.rowcount


def delete_expired_sessions(session: Session, user_id: UUID) -> int:
    """Sessions de plateforme (cookies chiffrés) expirées."""
    result = session.execute(
        delete(PlatformSession).where(PlatformSession.user_id == user_id, PlatformSession.expires_at < func.now())
    )
    return result.rowcount


def delete_old_activity(session: Session, user_id: UUID) -> int:
    """Événements d'activité de plus de 12 mois."""
    result = session.execute(
        delete(ActivityEvent).where(
            ActivityEvent.user_id == user_id, ActivityEvent.at < func.now() - func.make_interval(0, 12)
        )
    )
    return result.rowcount


def delete_old_tasks(session: Session, user_id: UUID) -> int:
    """Tâches finales de plus de 12 mois ; une tâche non finale (dont la purge en cours) n'est jamais touchée."""
    result = session.execute(
        delete(Task).where(
            Task.user_id == user_id,
            Task.created_at < func.now() - func.make_interval(0, 12),
            Task.status.in_(FINAL_TASK_STATUSES),
        )
    )
    return result.rowcount
