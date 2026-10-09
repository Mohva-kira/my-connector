"""Tâches persistées en base et isolation du dossier de données (vague 1, sous-lot 1c)."""

import threading
import time
import uuid

import pytest

from linkedin_market import config
from linkedin_market.db import models, session
from linkedin_market.web.tasks import DONE, FAILED, INPUT, TaskManager

pytestmark = pytest.mark.db


def wait_for(predicate, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def row(task_id: str) -> models.Task:
    with session.transaction() as s:
        return s.get(models.Task, uuid.UUID(task_id))


def test_tache_persistee_avec_null_puis_relue():
    manager = TaskManager()
    gate = threading.Event()
    task = manager.submit("rescore", "Recalcul", lambda ctx: gate.wait(5) and {"jobs": 3}, meta={"key": "k"})
    assert uuid.UUID(task.id).version == 7  # C-7
    stored = row(task.id)
    assert stored is not None and stored.kind == "rescore" and stored.meta == {"key": "k"}
    public = task.public()
    assert public["finished_at"] is None and public["created_at"].endswith("Z")  # C-1, C-2
    gate.set()
    assert wait_for(lambda: task.status == DONE)
    manager.shutdown(wait=True)
    stored = row(task.id)
    assert stored.status == DONE and stored.result == {"jobs": 3} and stored.finished_at is not None
    reloaded = TaskManager()
    again = reloaded.get(task.id)
    assert again.status == DONE and again.result == {"jobs": 3} and again.finished_at.endswith("Z")
    reloaded.shutdown(wait=True)


def test_etat_intermediaire_persiste_et_tache_interrompue():
    manager = TaskManager()
    task = manager.submit("run", "Question", lambda ctx: ctx.ask("Années ?", "years"))
    assert wait_for(lambda: task.status == INPUT)
    assert wait_for(lambda: row(task.id).status == INPUT)
    # Redémarrage pendant l'attente : la tâche relue n'est plus active.
    reloaded = TaskManager()
    assert reloaded.get(task.id).status == FAILED and not reloaded.has_active("run")
    manager.answer(task.id, "12")
    assert wait_for(lambda: task.status == DONE)
    manager.shutdown(wait=True)
    reloaded.shutdown(wait=True)
    assert row(task.id).status == DONE and row(task.id).result == "12"


def test_erreur_persistee():
    manager = TaskManager()
    task = manager.submit("train", "Échec", lambda ctx: 1 / 0)
    assert wait_for(lambda: task.status == FAILED)
    manager.shutdown(wait=True)
    stored = row(task.id)
    assert "division" in stored.error and any("ZeroDivisionError" in line for line in stored.logs)


def test_filtre_user_id(db_user_id):
    from linkedin_market.db import users
    from linkedin_market.db.repositories import tasks as tasks_repo

    manager = TaskManager()
    task = manager.submit("run", "x", lambda ctx: None)
    assert wait_for(lambda: task.status == DONE)
    manager.shutdown(wait=True)
    with session.transaction() as s:
        other = users.ensure_user(s, "autre@example.org")
        assert tasks_repo.recent(s, other, 10) == []
        tasks_repo.update_task(s, other, uuid.UUID(task.id), {"status": FAILED})
    assert row(task.id).status == DONE  # la mise à jour d'un autre utilisateur ne touche rien (I-01)


def test_donnees_sous_tmp_path_pendant_tout_le_test(tmp_path):
    """Aucune écriture dans le vrai DATA_DIR : tous les chemins de config pointent sous tmp_path."""
    paths = [
        config.DATA_DIR,
        config.SESSION_FILE,
        config.HISTORY_FILE,
        config.LABELS_FILE,
        config.MODEL_FILE,
        config.REPORTS_DIR,
        config.APPLICATIONS_DIR,
        config.APPLICATIONS_FILE,
        config.LINKEDIN_PROFILE_CACHE,
        config.SETTINGS_FILE,
        config.PROFILE_SETTINGS_FILE,
        config.ACTIVITY_FILE,
        config.TASKS_FILE,
    ]
    for path in paths:
        assert path.resolve().is_relative_to(tmp_path.resolve()), path
    manager = TaskManager()
    task = manager.submit("run", "x", lambda ctx: ctx.log("ok"))
    assert wait_for(lambda: task.status == DONE)
    manager.shutdown(wait=True)
    assert config.DATA_DIR.resolve().is_relative_to(tmp_path.resolve())
    assert not config.TASKS_FILE.exists()  # les tâches ne vont plus dans tasks.jsonl
