import re
import time

import pytest
from fastapi.testclient import TestClient

from linkedin_market import activity, application, browser, config, outreach, session, settings
from linkedin_market.tests.browser_fixtures import url_of
from linkedin_market.web.app import create_app
from linkedin_market.web.tasks import TaskManager

pytestmark = pytest.mark.db  # réglages, historique et rapports en PostgreSQL (vague 1)


@pytest.fixture
def client(monkeypatch, workdir):
    monkeypatch.setenv("LINKEDIN_EMAIL", "me@example.com")
    monkeypatch.setenv("LINKEDIN_PASSWORD", "x")
    for var in ("SMTP_HOST", "SMTP_FROM", "SMTP_USER", "SMTP_PASSWORD", "SMTP_PORT", "SMTP_SECURITY"):
        monkeypatch.delenv(var, raising=False)
    apps_dir = workdir / "applications"
    folder = apps_dir / "20261007-acme-dev"
    folder.mkdir(parents=True)
    (folder / "cv.md").write_text("# Dev\n\n- Python\n", encoding="utf-8")
    (folder / "message.txt").write_text("Bonjour, je candidate.", encoding="utf-8")
    monkeypatch.setattr(config, "APPLICATIONS_DIR", apps_dir)
    application.save_applications(
        {
            "job:1": {
                "title": "Senior Python Developer",
                "company": "Acme",
                "url": url_of("hiring_team.html"),
                "score": 90,
                "folder": str(folder),
                "generator": "local",
                "status": "préparée",
                "prepared_at": "2026-10-07T10:00:00",
            }
        }
    )
    manager = TaskManager(persist=False)
    with TestClient(create_app(tasks=manager, token=None), base_url="http://127.0.0.1:8765") as c:
        yield c
    manager.shutdown()


def wait_done(client, task_id, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        task = client.get(f"/api/tasks/{task_id}").json()
        if task["status"] in ("terminée", "échec", "annulée"):
            return task
        time.sleep(0.2)
    raise AssertionError(task)


def use_smtp(monkeypatch, server):
    monkeypatch.setenv("SMTP_HOST", "127.0.0.1")
    monkeypatch.setenv("SMTP_PORT", str(server.server_address[1]))
    monkeypatch.setenv("SMTP_SECURITY", "none")
    monkeypatch.setenv("SMTP_FROM", "me@example.com")


MAIL = {"channel": "email", "to": "jane@acme.com", "subject": "Candidature", "body": "Bonjour Jane", "attach_cv": True}


def test_message_requires_confirmation_and_smtp(client, monkeypatch, smtp_server):
    assert client.post("/api/applications/job:1/messages", json=MAIL).status_code == 422  # pas de confirm
    assert (
        client.post("/api/applications/job:1/messages", json={**MAIL, "confirm": True}).status_code == 503
    )  # SMTP absent
    use_smtp(monkeypatch, smtp_server)
    assert (
        client.post("/api/applications/job:1/messages", json={**MAIL, "confirm": True, "to": "pas-un-mail"}).status_code
        == 422
    )
    assert (
        client.post(
            "/api/applications/job:1/messages", json={**MAIL, "confirm": True, "subject": "a\nBcc: x@y.z"}
        ).status_code
        == 422
    )
    assert smtp_server.__dict__.get("mails", []) == []  # rien n'est parti


def test_email_is_sent_logged_and_status_advances(client, monkeypatch, smtp_server):
    use_smtp(monkeypatch, smtp_server)
    draft = client.get("/api/applications/job:1/message-draft").json()
    assert draft["subject"].startswith("Candidature") and "je candidate" in draft["body"]
    task = client.post("/api/applications/job:1/messages", json={**MAIL, "confirm": True}).json()
    done = wait_done(client, task["id"])
    assert done["status"] == "terminée", done
    assert 'filename="cv.pdf"' in smtp_server.mails[0] and "Bonjour Jane" in smtp_server.mails[0]
    entry = application.get_application("job:1")
    assert (
        entry["status"] == "contactée"
        and entry["history"][-1]["kind"] == "message"
        and entry["history"][-1]["to"] == "jane@acme.com"
    )
    assert activity.count_today("message_email") == 1


def test_message_does_not_downgrade_a_posted_application(client, monkeypatch, smtp_server):
    use_smtp(monkeypatch, smtp_server)
    application.set_status("job:1", "postulée")
    task = client.post("/api/applications/job:1/messages", json={**MAIL, "confirm": True}).json()
    assert wait_done(client, task["id"])["status"] == "terminée"
    assert application.get_application("job:1")["status"] == "postulée"


def test_daily_message_quota(client, monkeypatch, smtp_server):
    use_smtp(monkeypatch, smtp_server)
    settings.save_settings({"message_daily_max": 1})
    activity.record("message_linkedin", job="job:2")
    assert client.post("/api/applications/job:1/messages", json={**MAIL, "confirm": True}).status_code == 429


def test_smtp_test_only_to_self(client, monkeypatch, smtp_server):
    assert client.post("/api/smtp/test", json={}).status_code == 503
    use_smtp(monkeypatch, smtp_server)
    assert client.post("/api/smtp/test", json={"to": "stranger@evil.io"}).status_code == 422
    assert client.post("/api/smtp/test", json={}).json()["to"] == "me@example.com"
    assert len(smtp_server.mails) == 1


def test_linkedin_message_end_to_end(client, monkeypatch):
    import os

    os.environ["LINKEDIN_HEADLESS"] = "1"
    os.environ.setdefault("LINKEDIN_BROWSER", "firefox")
    monkeypatch.setattr(outreach, "_PROFILE_URL", re.compile(r"^file://"))
    monkeypatch.setattr(session, "login", lambda *a, **k: None)
    monkeypatch.setattr(
        config, "CAUTION_LEVELS", {**config.CAUTION_LEVELS, "prudent": config.Caution("prudent", 1, 0.0, 0.0)}
    )
    holder = {}
    real = browser.create_driver
    monkeypatch.setattr(browser, "create_driver", lambda: holder.setdefault("d", real()))
    body = {"channel": "linkedin", "to": url_of("profile_first_degree.html"), "body": "Bonjour Jane", "confirm": True}
    task = client.post("/api/applications/job:1/messages", json=body).json()
    done = wait_done(client, task["id"])
    assert done["status"] == "terminée", done
    assert application.get_application("job:1")["status"] == "contactée"
    assert activity.count_today("message_linkedin") == 1


def test_contact_lookup_end_to_end(client, monkeypatch):
    import os

    os.environ["LINKEDIN_HEADLESS"] = "1"
    os.environ.setdefault("LINKEDIN_BROWSER", "firefox")
    from linkedin_market import contacts

    local = {
        "https://www.linkedin.com/in/jane-doe/": url_of("contact-jane.html"),
        "https://www.linkedin.com/in/john-roe/": url_of("contact-john.html"),
    }
    monkeypatch.setattr(contacts, "overlay_url", lambda u: local[u])
    original = contacts.find_hiring_team
    monkeypatch.setattr(
        contacts, "find_hiring_team", lambda *a, **k: original(*a, **{**k, "to_overlay": lambda u: local[u]})
    )
    monkeypatch.setattr(session, "login", lambda *a, **k: None)
    monkeypatch.setattr(
        config, "CAUTION_LEVELS", {**config.CAUTION_LEVELS, "prudent": config.Caution("prudent", 1, 0.0, 0.0)}
    )
    # contact saisi à la main : son email ne doit pas être écrasé
    application.update_application(
        "job:1",
        contacts=[
            {
                "full_name": "Jane",
                "profile_url": "https://www.linkedin.com/in/jane-doe/",
                "email": "j.manual@acme.com",
                "source": "manuel",
            }
        ],
    )
    task = client.post("/api/applications/job:1/contacts/lookup").json()
    done = wait_done(client, task["id"])
    assert done["status"] == "terminée", done
    assert done["result"] == {"found": 2, "emails": 1}
    saved = {c["full_name"]: c for c in application.get_application("job:1")["contacts"]}
    assert saved["Jane Doe"]["email"] == "j.manual@acme.com" and saved["John Roe"]["email"] is None
