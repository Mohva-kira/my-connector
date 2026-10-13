"""Service de rétention : purge des données arrivées à échéance (I-10, I-14, ``backend/03`` § 6).

``RetentionService.purge(user_id)`` supprime, en **une** transaction par utilisateur : les éléments collectés
expirés sans candidature, les contacts d'entreprise expirés, les sessions de plateforme expirées, les événements
d'activité et les tâches finales de plus de 12 mois. Jamais ``suppression_list``, ``reports`` ni ``applications``
(I-02, I-13). Les compteurs seuls sont journalisés (I-15) : aucun contenu.
"""

from __future__ import annotations

import logging
import threading
from uuid import UUID

from .db.repositories import retention as retention_repo
from .db.session import transaction
from .db.users import current_user_id

log = logging.getLogger(__name__)

PURGE_INTERVAL_SECONDS = 24 * 3600
PURGE_KIND = "purge"
PURGE_TITLE = "Purge de rétention"


class RetentionService:
    """Purge de rétention d'un utilisateur."""

    def purge(self, user_id: UUID) -> dict[str, dict[str, int]]:
        """Retourne ``{"deleted": {"collected_items": n, "company_contacts": n, "platform_sessions": n, ...}}``."""
        with transaction() as s:
            deleted = {
                "collected_items": retention_repo.delete_expired_items(s, user_id),
                "company_contacts": retention_repo.delete_expired_contacts(s, user_id),
                "platform_sessions": retention_repo.delete_expired_sessions(s, user_id),
                "activity_events": retention_repo.delete_old_activity(s, user_id),
                "tasks": retention_repo.delete_old_tasks(s, user_id),
            }
        log.info("Purge de rétention : %s", ", ".join(f"{table} {count}" for table, count in deleted.items()))
        return {"deleted": deleted}


def purge_current_user() -> dict[str, dict[str, int]]:
    """Purge de l'utilisateur de l'instance (CLI au démarrage, tâche ``purge`` de l'interface web)."""
    with transaction() as s:
        user_id = current_user_id(s)
    return RetentionService().purge(user_id)


def submit_purge(manager) -> bool:
    """Soumet la tâche ``purge`` (non navigateur) sauf si une purge est déjà active. Renvoie ``True`` si soumise."""
    if manager.has_active(PURGE_KIND):
        return False
    manager.submit(PURGE_KIND, PURGE_TITLE, lambda ctx: purge_current_user(), browser=False)
    return True


class PurgeScheduler:
    """Purge au démarrage (soumise tout de suite par ``start``) puis toutes les ``interval_seconds`` (thread daemon).

    Démarrée uniquement par ``python -m linkedin_market.web`` après ``prepare_database()``, jamais par
    ``create_app()``. Une erreur de soumission est journalisée et la minuterie continue.
    """

    def __init__(self, manager, interval_seconds: float = PURGE_INTERVAL_SECONDS) -> None:
        self._manager = manager
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="lm-purge", daemon=True)

    def _submit(self) -> None:
        try:
            submit_purge(self._manager)
        except Exception as exc:  # noqa: BLE001 - la minuterie ne doit jamais s'arrêter
            log.warning("Purge de rétention non soumise : %s", exc.__class__.__name__)

    def start(self) -> None:
        self._submit()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            self._submit()
