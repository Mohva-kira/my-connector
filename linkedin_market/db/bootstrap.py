"""Préparation de la base au démarrage (CLI et interface web, ``backend/03`` § 1).

Ordre : 1) ``alembic upgrade head`` ; 2) utilisateur unique créé si la base n'en a aucun ; 3) import des fichiers
historiques non migrés ; 4) tâches non finales passées en ``échec`` (arrêt de l'application). Base injoignable ou
non configurée ⇒ ``SystemExit`` avec un message français, mot de passe masqué : aucun repli sur les fichiers.
"""

from __future__ import annotations

import logging
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import func, update
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError, DBAPIError, OperationalError

from .. import config
from .import_json import import_legacy_files
from .models import FINAL_TASK_STATUSES, Task
from .session import dispose_engine, get_engine, transaction
from .users import ensure_initial_user

log = logging.getLogger(__name__)

ALEMBIC_INI = Path(__file__).resolve().parent.parent / "alembic.ini"
INTERRUPTED = "Tâche interrompue par l'arrêt de l'application."


def _masked_url() -> str:
    try:
        url = make_url(config.database_url())
    except config.DatabaseConfigError, ArgumentError:
        return "(URL invalide)"
    return url.render_as_string(hide_password=True)


def _unreachable(exc: Exception) -> SystemExit:
    try:
        password = make_url(config.database_url()).password or ""
    except config.DatabaseConfigError, ArgumentError:
        password = ""
    detail = str(getattr(exc, "orig", None) or exc.__class__.__name__).strip().splitlines()[0][:200]
    if password:
        detail = detail.replace(password, "***")
    return SystemExit(
        f"Base PostgreSQL injoignable ({_masked_url()}) : {detail}\n"
        "Démarre-la (« docker compose --env-file .env -f linkedin_market/compose.yaml up -d db » ou le service "
        "système) puis vérifie LM_DATABASE_URL dans le .env à la racine du dépôt."
    )


def upgrade_schema() -> None:
    """``alembic upgrade head`` programmatique, sur la connexion du moteur de l'application."""
    with get_engine().begin() as connection:
        cfg = Config(str(ALEMBIC_INI))
        cfg.attributes["connection"] = connection
        command.upgrade(cfg, "head")


def fail_interrupted_tasks() -> int:
    """Tâches restées actives lors du dernier arrêt -> ``échec`` (écriture durable). Renvoie leur nombre."""
    with transaction() as s:
        result = s.execute(
            update(Task)
            .where(Task.status.not_in(FINAL_TASK_STATUSES))
            .values(status="échec", error=INTERRUPTED, needs_input=None, finished_at=func.now())
        )
        return result.rowcount


def prepare_database(data_dir: Path | None = None) -> dict[str, int]:
    """Prépare la base ; renvoie les compteurs de l'import (et ``interrupted_tasks``)."""
    try:
        upgrade_schema()
        with transaction() as s:
            ensure_initial_user(s)
        imported = import_legacy_files(Path(data_dir) if data_dir is not None else config.DATA_DIR)
        interrupted = fail_interrupted_tasks()
    except config.DatabaseConfigError as exc:
        raise SystemExit(str(exc)) from None
    except (OperationalError, DBAPIError) as exc:
        if isinstance(exc, OperationalError) or exc.connection_invalidated:
            dispose_engine()
            raise _unreachable(exc) from None
        raise
    if interrupted:
        log.info("%d tâche(s) interrompue(s) par le dernier arrêt marquée(s) en échec.", interrupted)
    return {**imported, "interrupted_tasks": interrupted}
