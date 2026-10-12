"""Candidature de bout en bout : API -> tâche -> Selenium réel sur une page locale -> confirmation utilisateur."""

import time

import pytest
from fastapi.testclient import TestClient

from linkedin_market import activity, application, config, session, settings, storage
from linkedin_market.models import Item
from linkedin_market.tests.browser_fixtures import url_of
from linkedin_market.web.app import create_app
from linkedin_market.web.tasks import TaskManager

pytestmark = pytest.mark.db  # réglages, historique et rapports en PostgreSQL (vague 1)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("LINKEDIN_EMAIL", "me@example.com")
    monkeypatch.setenv("LINKEDIN_PASSWORD", "x")
    manager = TaskManager(persist=False)
    with TestClient(create_app(tasks=manager, token=None), base_url="http://127.0.0.1:8765") as c:
        yield c
    # Isolation : aucune tâche d'un test ne doit survivre au TRUNCATE du suivant (annulation puis attente de fin).
    for task in manager.active():
        manager.cancel(task.id)
    manager.shutdown(wait=True)


def prepare_entry(workdir, monkeypatch, url, status="préparée"):
    apps_dir = workdir / "applications"
    folder = apps_dir / "20261007-acme-dev"
    folder.mkdir(parents=True)
    (folder / "cv.md").write_text("# Dev Python\n\n## Compétences\n\n- Python\n- FastAPI\n", encoding="utf-8")
    (folder / "message.txt").write_text("Bonjour", encoding="utf-8")
    monkeypatch.setattr(config, "APPLICATIONS_DIR", apps_dir)
    # L'offre doit être dans l'historique pour pouvoir être annotée (labels -> collected_items).
    storage.append_items(
        [
            Item(
                kind="job",
                id="job:1",
                url=url,
                title="Senior Python Developer",
                text="Python",
                keyword="python",
                company="Acme",
            )
        ]
    )
    application.save_applications(
        {
            "job:1": {
                "title": "Senior Python Developer",
                "company": "Acme",
                "url": url,
                "score": 90,
                "folder": str(folder),
                "generator": "local",
                "status": status,
                "prepared_at": "2026-10-07T10:00:00",
            }
        }
    )
    return folder


def wait_task(client, task_id, until, timeout=180):
    """Relève la tâche jusqu'à ``until`` ; délai large : sous forte charge, Firefox met longtemps à se fermer."""
    end = time.time() + timeout
    while time.time() < end:
        task = client.get(f"/api/tasks/{task_id}").json()
        if until(task):
            return task
        time.sleep(0.2)
    raise AssertionError(f"Délai dépassé : {task}")


def test_guards_before_launching(client, monkeypatch, workdir, accept_risks):
    accept_risks()
    prepare_entry(workdir, monkeypatch, "https://www.linkedin.com/jobs/view/1/")
    monkeypatch.delenv("LINKEDIN_EMAIL")
    assert client.post("/api/applications/job:1/apply").status_code == 409  # identifiants absents
    monkeypatch.setenv("LINKEDIN_EMAIL", "me@example.com")
    settings.save_settings({"apply_daily_max": 1})
    activity.record("apply", job="job:x")
    assert client.post("/api/applications/job:1/apply").status_code == 429  # plafond du jour
    settings.save_settings({"apply_daily_max": 5})
    activity.record("blocked", where="collecte")
    blocked = client.post("/api/applications/job:1/apply")
    assert blocked.status_code == 409 and "interrompu" in blocked.json()["detail"]
    assert client.get("/api/status").json()["blocked"]["where"] == "collecte"
    client.post("/api/block/clear")
    assert client.get("/api/status").json()["blocked"] is None
    application.set_status("job:1", "postulée")
    assert client.post("/api/applications/job:1/apply").status_code == 409  # déjà postulée
    assert client.post("/api/applications/job:404/apply").status_code == 404


def test_apply_end_to_end_with_user_confirmation(client, monkeypatch, workdir, accept_risks):
    accept_risks()
    import os

    os.environ["LINKEDIN_HEADLESS"] = "1"
    os.environ.setdefault("LINKEDIN_BROWSER", "firefox")
    prepare_entry(workdir, monkeypatch, url_of("easy_apply.html"))
    settings.save_settings({"contact_email": "me@example.com", "contact_phone": "+2250102030405", "caution": "prudent"})
    monkeypatch.setattr(session, "login", lambda *a, **k: None)
    monkeypatch.setattr(
        config, "CAUTION_LEVELS", {**config.CAUTION_LEVELS, "prudent": config.Caution("prudent", 1, 0.0, 0.0)}
    )

    task = client.post("/api/applications/job:1/apply").json()
    assert client.post("/api/applications/job:1/apply").status_code == 409  # déjà en cours

    asked = []
    while True:
        task = wait_task(
            client, task["id"], lambda t: t["needs_input"] or t["status"] in ("terminée", "échec", "annulée")
        )
        if not task["needs_input"]:
            break
        question = task["needs_input"]
        asked.append(question["field"] if question["field"] == "confirm_submit" else question["question"])
        if question["field"] == "confirm_submit":
            assert "Years: 12" in question["review"]
            assert application.get_application("job:1")["status"] == "préparée"  # rien n'est parti avant le clic
            answer = "Soumettre"
        else:
            answer = "12" if "years" in question["question"] else "Yes"
        assert client.post(f"/api/tasks/{task['id']}/answer", json={"value": answer}).status_code == 200
        # La réponse retire la question avant de rendre la main (TaskManager.answer) : le prochain relevé ne peut
        # pas revoir cette question et y répondre une seconde fois (cause de l'instabilité sous charge).
        assert client.get(f"/api/tasks/{task['id']}").json()["needs_input"] != question

    assert task["status"] == "terminée", task
    assert task["result"]["state"] == "postulée"
    entry = application.get_application("job:1")
    assert entry["status"] == "postulée" and entry["apply_state"] == "postulée"
    assert entry["history"][-1]["kind"] == "apply"
    assert storage.load_labels()["job:1"] == 1
    assert activity.count_today("apply") == 1
    # la réponse aux questions est mémorisée, la confirmation d'envoi ne l'est pas
    answers = settings.load_settings()["answers"]
    assert any("years of experience" in q for q in answers) and not any("soumettre" in q for q in answers)
    assert (workdir / "applications" / "20261007-acme-dev" / "cv.pdf").is_file()


def test_decline_confirmation_keeps_status(client, monkeypatch, workdir, accept_risks):
    accept_risks()
    import os

    os.environ["LINKEDIN_HEADLESS"] = "1"
    os.environ.setdefault("LINKEDIN_BROWSER", "firefox")
    prepare_entry(workdir, monkeypatch, url_of("easy_apply.html"))
    settings.save_settings({"contact_email": "me@example.com", "contact_phone": "+2250102030405"})
    settings.remember_answer("How many years of experience do you have with Python?", "9")
    settings.remember_answer("Are you legally authorized to work remotely? *", "Yes")
    monkeypatch.setattr(session, "login", lambda *a, **k: None)
    monkeypatch.setattr(
        config, "CAUTION_LEVELS", {**config.CAUTION_LEVELS, "prudent": config.Caution("prudent", 1, 0.0, 0.0)}
    )
    task = client.post("/api/applications/job:1/apply").json()
    task = wait_task(client, task["id"], lambda t: t["needs_input"])
    assert client.post(f"/api/tasks/{task['id']}/answer", json={"value": "Annuler"}).status_code == 200
    task = wait_task(client, task["id"], lambda t: t["status"] in ("terminée", "échec", "annulée"))
    assert task["result"]["state"] == "annulée"
    assert application.get_application("job:1")["status"] == "préparée"
    assert activity.count_today("apply") == 0
