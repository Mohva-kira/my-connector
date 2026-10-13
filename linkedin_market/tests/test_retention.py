"""Purge de rétention (vague 6, I-10, I-14, ``backend/03`` § 6)."""

import logging
import threading
import time
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from alembic import command
from sqlalchemy import func, insert, select, text
from sqlalchemy.exc import IntegrityError

from linkedin_market import retention
from linkedin_market.db import models, session, users
from linkedin_market.db.repositories import items as items_repo
from linkedin_market.retention import PurgeScheduler, RetentionService
from linkedin_market.tests.conftest import alembic_config
from linkedin_market.web import __main__ as web_main
from linkedin_market.web.app import create_app
from linkedin_market.web.tasks import DONE, TaskManager

pytestmark = pytest.mark.db


def ago(**delta) -> datetime:
    return datetime.now(UTC) - timedelta(**delta)


def make_item(s, uid: UUID, external_id: str, kind: str = "job", collected_days_ago: int = 0, **values) -> UUID:
    collected_at = ago(days=collected_days_ago)
    row = {
        "user_id": uid,
        "source_code": "linkedin",
        "kind": kind,
        "external_id": external_id,
        "text": None if kind == "person" else "texte",
        "collected_at": collected_at,
        "purge_after": items_repo.purge_after(kind, collected_at),
        **values,
    }
    return s.execute(insert(models.CollectedItem).values(**row).returning(models.CollectedItem.id)).scalar_one()


def make_application(s, uid: UUID, item_id: UUID) -> UUID:
    return s.execute(
        insert(models.Application)
        .values(user_id=uid, item_id=item_id, score=80, generator="local", folder_path="applications/x")
        .returning(models.Application.id)
    ).scalar_one()


def make_contact(s, uid: UUID, application_id: UUID, name: str, **values) -> UUID:
    return s.execute(
        insert(models.CompanyContact)
        .values(user_id=uid, application_id=application_id, full_name=name, **values)
        .returning(models.CompanyContact.id)
    ).scalar_one()


def make_task(s, uid: UUID, status: str, created_days_ago: int, kind: str = "run") -> UUID:
    return s.execute(
        insert(models.Task)
        .values(user_id=uid, kind=kind, status=status, created_at=ago(days=created_days_ago))
        .returning(models.Task.id)
    ).scalar_one()


def count(s, model, uid: UUID) -> int:
    return s.execute(select(func.count()).select_from(model).where(model.user_id == uid)).scalar_one()


def purge(uid: UUID) -> dict:
    return RetentionService().purge(uid)["deleted"]


def test_offre_expiree_sans_candidature_supprimee_avec_candidature_conservee(db_user_id):
    with session.transaction() as s:
        expired = make_item(s, db_user_id, "1", collected_days_ago=91)
        fresh = make_item(s, db_user_id, "2", collected_days_ago=10)
        applied = make_item(s, db_user_id, "3", collected_days_ago=200)
        make_application(s, db_user_id, applied)
        annotated = make_item(s, db_user_id, "4", collected_days_ago=200, purge_after=None)
        s.execute(insert(models.Label).values(item_id=annotated, user_id=db_user_id, value=1))
        gone = make_item(s, db_user_id, "5", collected_days_ago=200)
        s.execute(insert(models.Label).values(item_id=gone, user_id=db_user_id, value=1))
    deleted = purge(db_user_id)
    with session.transaction() as s:
        kept = set(s.execute(select(models.CollectedItem.id)).scalars())
        labels = set(s.execute(select(models.Label.item_id)).scalars())
    assert kept == {fresh, applied, annotated}
    assert expired not in kept and deleted["collected_items"] == 2
    assert labels == {annotated}  # le label de l'offre supprimée suit en cascade


def test_post_et_person_a_30_jours(db_user_id):
    with session.transaction() as s:
        old_post = make_item(s, db_user_id, "p1", kind="post", collected_days_ago=31)
        young_post = make_item(s, db_user_id, "p2", kind="post", collected_days_ago=29)
        old_person = make_item(s, db_user_id, "q1", kind="person", collected_days_ago=31)
        young_person = make_item(s, db_user_id, "q2", kind="person", collected_days_ago=29)
    purge(db_user_id)
    with session.transaction() as s:
        kept = set(s.execute(select(models.CollectedItem.id)).scalars())
    assert kept == {young_post, young_person} and not {old_post, old_person} & kept


def test_contacts_expires_supprimes_recents_conserves(db_user_id):
    with session.transaction() as s:
        app_id = make_application(s, db_user_id, make_item(s, db_user_id, "1"))
        old = make_contact(s, db_user_id, app_id, "Ancien", purge_after=ago(days=1))
        recent = make_contact(s, db_user_id, app_id, "Récent")
        within = make_contact(s, db_user_id, app_id, "Bientôt", purge_after=datetime.now(UTC) + timedelta(days=1))
    deleted = purge(db_user_id)
    with session.transaction() as s:
        kept = set(s.execute(select(models.CompanyContact.id)).scalars())
        default = s.execute(
            select(models.CompanyContact.purge_after, func.now()).where(models.CompanyContact.id == recent)
        )
        purge_after, now = default.one()
    assert kept == {recent, within} and old not in kept and deleted["company_contacts"] == 1
    assert timedelta(days=364 * 3) < purge_after - now < timedelta(days=367 * 3)  # intervalle SQL de 3 ans


def test_suppression_list_reports_et_candidatures_jamais_purges(db_user_id):
    with session.transaction() as s:
        s.execute(
            insert(models.SuppressionEntry).values(
                user_id=db_user_id, recipient_hash="h" * 64, reason="désinscription", created_at=ago(days=4000)
            )
        )
        s.execute(
            insert(models.Report).values(
                user_id=db_user_id,
                slug="r-2020",
                keyword="x",
                interpreter_used="local",
                html_path="reports/r-2020.html",
                created_at=ago(days=4000),
            )
        )
        item = make_item(s, db_user_id, "1", collected_days_ago=500)
        make_application(s, db_user_id, item)
    purge(db_user_id)
    with session.transaction() as s:
        assert count(s, models.SuppressionEntry, db_user_id) == 1
        assert count(s, models.Report, db_user_id) == 1
        assert count(s, models.Application, db_user_id) == 1
        assert count(s, models.CollectedItem, db_user_id) == 1


def test_sessions_activite_et_taches(db_user_id):
    with session.transaction() as s:
        for code, expires in (("linkedin", ago(hours=1)), ("malt", datetime.now(UTC) + timedelta(hours=1))):
            s.execute(
                insert(models.PlatformSession).values(
                    user_id=db_user_id,
                    source_code=code,
                    cookies_encrypted=b"x",
                    last_login_at=ago(days=1),
                    expires_at=expires,
                )
            )
        s.execute(insert(models.ActivityEvent).values(user_id=db_user_id, kind="run", at=ago(days=400)))
        s.execute(insert(models.ActivityEvent).values(user_id=db_user_id, kind="run", at=ago(days=300)))
        old_done = make_task(s, db_user_id, "terminée", 400)
        old_failed = make_task(s, db_user_id, "échec", 400)
        old_cancelled = make_task(s, db_user_id, "annulée", 400)
        recent = make_task(s, db_user_id, "terminée", 300)
        active = make_task(s, db_user_id, "en_cours", 400)
        waiting = make_task(s, db_user_id, "en_attente", 400)
        running_purge = make_task(s, db_user_id, "en_cours", 400, kind="purge")
    deleted = purge(db_user_id)
    with session.transaction() as s:
        sessions = set(s.execute(select(models.PlatformSession.source_code)).scalars())
        tasks = set(s.execute(select(models.Task.id)).scalars())
        events = count(s, models.ActivityEvent, db_user_id)
    assert sessions == {"malt"} and events == 1
    assert tasks == {recent, active, waiting, running_purge}
    assert not {old_done, old_failed, old_cancelled} & tasks
    assert deleted == {
        "collected_items": 0,
        "company_contacts": 0,
        "platform_sessions": 1,
        "activity_events": 1,
        "tasks": 3,
    }


def test_isolation_entre_utilisateurs(db_user_id):
    with session.transaction() as s:
        other = users.ensure_user(s, "autre@example.org")
        make_item(s, db_user_id, "1", collected_days_ago=200)
        make_item(s, other, "1", collected_days_ago=200)
        make_task(s, other, "terminée", 400)
        s.execute(insert(models.ActivityEvent).values(user_id=other, kind="run", at=ago(days=400)))
    purge(db_user_id)
    with session.transaction() as s:
        assert count(s, models.CollectedItem, db_user_id) == 0
        assert count(s, models.CollectedItem, other) == 1
        assert count(s, models.Task, other) == 1 and count(s, models.ActivityEvent, other) == 1


def test_idempotence(db_user_id):
    with session.transaction() as s:
        make_item(s, db_user_id, "1", collected_days_ago=200)
    assert purge(db_user_id)["collected_items"] == 1
    assert purge(db_user_id) == dict.fromkeys(
        ("collected_items", "company_contacts", "platform_sessions", "activity_events", "tasks"), 0
    )


def test_journal_sans_contenu(db_user_id, caplog):
    with session.transaction() as s:
        make_item(s, db_user_id, "1", collected_days_ago=200, title="Titre secret", company="Société secrète")
    with caplog.at_level(logging.INFO, logger="linkedin_market"):
        purge(db_user_id)
    assert "collected_items 1" in caplog.text
    assert "secret" not in caplog.text and "secrète" not in caplog.text


def test_tache_purge_executee_et_resultat(db_user_id):
    with session.transaction() as s:
        make_item(s, db_user_id, "1", collected_days_ago=200)
    manager = TaskManager()
    try:
        assert retention.submit_purge(manager) is True
        task = manager.list()[0]
        end = time.time() + 10
        while task.status != DONE and time.time() < end:
            time.sleep(0.02)
        assert task.kind == "purge" and task.title == "Purge de rétention" and task.browser is False
        assert task.result["deleted"]["collected_items"] == 1
        assert set(task.result["deleted"]) == {
            "collected_items",
            "company_contacts",
            "platform_sessions",
            "activity_events",
            "tasks",
        }
    finally:
        manager.shutdown(wait=True)


def test_pas_de_doublon_de_purge():
    manager = TaskManager(persist=False)
    gate = threading.Event()
    try:
        manager.submit("purge", "Purge de rétention", lambda ctx: gate.wait(5))
        assert manager.has_active("purge")
        assert retention.submit_purge(manager) is False
        assert [t.kind for t in manager.list()] == ["purge"]
    finally:
        gate.set()
        manager.shutdown(wait=True)


def test_check_tasks_kind_accepte_purge(db_user_id):
    with session.transaction() as s:
        make_task(s, db_user_id, "en_attente", 0, kind="purge")
    with pytest.raises(IntegrityError), session.transaction() as s:
        make_task(s, db_user_id, "en_attente", 0, kind="inconnue")


def test_migration_0004_aller_retour(db_schema):
    with db_schema.begin() as conn:
        cfg = alembic_config(conn)
        try:
            command.downgrade(cfg, "0003")
            columns = set(
                conn.execute(
                    text("SELECT column_name FROM information_schema.columns WHERE table_name='user_settings'")
                )
                .scalars()
                .all()
            )
            assert not {"home_location", "timezone"} & columns
        finally:
            command.upgrade(cfg, "head")
        command.check(alembic_config(conn))


# ------------------------------------------------------------------- démarrage
def test_create_app_ne_soumet_pas_de_purge():
    manager = TaskManager(persist=False)
    try:
        create_app(tasks=manager, token=None)
        assert manager.list() == []
    finally:
        manager.shutdown()


def test_demarrage_web_soumet_une_purge(monkeypatch, db_user_id):
    manager = TaskManager(persist=False)
    calls: list[str] = []
    monkeypatch.setattr(web_main, "prepare_database", lambda: calls.append("prepare"))
    monkeypatch.setattr(web_main, "create_app", lambda **kw: create_app(tasks=manager, token=None))
    monkeypatch.setattr(web_main.uvicorn, "run", lambda *a, **kw: calls.append(f"uvicorn:{len(manager.list())}"))
    monkeypatch.setattr(web_main.webbrowser, "open", lambda url: None)
    monkeypatch.setattr("sys.argv", ["linkedin_market.web", "--no-open", "--dev"])
    started: list[PurgeScheduler] = []
    original_start = PurgeScheduler.start

    def start(self):
        started.append(self)
        original_start(self)

    monkeypatch.setattr(PurgeScheduler, "start", start)
    try:
        web_main.main()
        assert calls == ["prepare", "uvicorn:1"]  # purge soumise après prepare_database, avant uvicorn.run
        assert [t.kind for t in manager.list()] == ["purge"]
    finally:
        started[0].stop()
        manager.shutdown(wait=True)


def test_minuterie_renouvelle_la_purge():
    manager = TaskManager(persist=False)
    submitted: list[bool] = []
    scheduler = PurgeScheduler(manager, interval_seconds=0.05)
    original = retention.submit_purge
    try:
        retention.submit_purge = lambda m: submitted.append(True) or original(m)
        scheduler.start()
        end = time.time() + 5
        while len(submitted) < 3 and time.time() < end:
            time.sleep(0.02)
        assert len(submitted) >= 3
    finally:
        retention.submit_purge = original
        scheduler.stop()
        manager.shutdown(wait=True)
