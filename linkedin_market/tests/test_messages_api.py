import email
import email.policy
import re
import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, update

from linkedin_market import activity, application, browser, config, outreach, session, settings
from linkedin_market.db import models
from linkedin_market.db import session as db_session
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
    # Isolation : aucune tâche d'un test ne doit survivre au TRUNCATE du suivant (annulation puis attente de fin).
    for task in manager.active():
        manager.cancel(task.id)
    manager.shutdown(wait=True)


def wait_done(client, task_id, timeout=180):
    """Délai large : sous forte charge, la fermeture de Firefox (``driver.quit``) peut dépasser une minute."""
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
    assert client.put("/api/settings", json={"full_name": "Ada Lovelace"}).status_code == 200
    task = client.post("/api/applications/job:1/messages", json={**MAIL, "confirm": True}).json()
    done = wait_done(client, task["id"])
    assert done["status"] == "terminée", done
    # Pièce jointe nommée pour le recruteur (format ATS) : « Prénom-Nom-CV-Titre.pdf ».
    assert 'filename="Ada-Lovelace-CV-' in smtp_server.mails[0] and "Bonjour Jane" in smtp_server.mails[0]
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


def test_linkedin_message_end_to_end(client, monkeypatch, accept_risks):
    accept_risks()
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


def test_contact_lookup_end_to_end(client, monkeypatch, accept_risks):
    accept_risks()
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
    logs = "\n".join(done["logs"])  # G-9 : ni nom complet ni adresse de tiers dans le journal persisté
    assert "J. D. : j***@acme.com" in logs and not any(x in logs for x in ("Jane Doe", "John Roe", "jane@acme.com"))
    saved = {c["full_name"]: c for c in application.get_application("job:1")["contacts"]}
    assert saved["Jane Doe"]["email"] == "j.manual@acme.com" and saved["John Roe"]["email"] is None


# ------------------------------------------------- vague 4 : pied imposé et opposition
def parsed(raw: str):
    return email.message_from_string(raw, policy=email.policy.default)


def stored_messages() -> list[models.OutreachMessage]:
    with db_session.transaction() as s:
        return list(s.execute(select(models.OutreachMessage)).scalars())


def test_email_porte_le_pied_et_l_en_tete_list_unsubscribe(client, monkeypatch, smtp_server):
    use_smtp(monkeypatch, smtp_server)
    footer = client.get("/api/applications/job:1/message-draft", params={"to": "jane@acme.com"}).json()["footer"]
    task = client.post("/api/applications/job:1/messages", json={**MAIL, "confirm": True, "attach_cv": False}).json()
    assert wait_done(client, task["id"])["status"] == "terminée"
    sent = parsed(smtp_server.mails[0])
    assert sent["List-Unsubscribe"] == "<mailto:me@example.com?subject=STOP>"
    assert sent["List-Unsubscribe-Post"] is None
    text = sent.get_body().get_content().replace("\r\n", "\n")
    assert text.rstrip("\n") == "Bonjour Jane\n\n" + footer  # corps + « \n\n » + pied
    (row,) = stored_messages()
    assert row.body == "Bonjour Jane\n\n" + footer  # le texte réellement envoyé
    assert row.status == "envoyé"


def test_pied_saisi_par_le_client_n_est_pas_double(client, monkeypatch, smtp_server):
    use_smtp(monkeypatch, smtp_server)
    footer = client.get("/api/applications/job:1/message-draft", params={"to": "jane@acme.com"}).json()["footer"]
    body = {**MAIL, "confirm": True, "attach_cv": False, "body": "Bonjour Jane\n\n" + footer}
    task = client.post("/api/applications/job:1/messages", json=body).json()
    assert wait_done(client, task["id"])["status"] == "terminée"
    text = parsed(smtp_server.mails[0]).get_body().get_content().replace("\r\n", "\n")
    assert text.count("Pour ne plus recevoir") == 1 and text.rstrip("\n") == "Bonjour Jane\n\n" + footer
    assert stored_messages()[0].body.count("Pour ne plus recevoir") == 1


def test_pied_selon_l_origine_du_contact(client, monkeypatch, smtp_server):
    use_smtp(monkeypatch, smtp_server)
    application.update_application(
        "job:1",
        contacts=[{"full_name": "Jane", "email": "jane@acme.com", "source": "linkedin_equipe_recrutement"}],
    )
    task = client.post("/api/applications/job:1/messages", json={**MAIL, "confirm": True, "attach_cv": False}).json()
    assert wait_done(client, task["id"])["status"] == "terminée"
    assert "provient de l'équipe de recrutement affichée sur l'offre LinkedIn." in stored_messages()[0].body


def test_smtp_test_sans_list_unsubscribe(client, monkeypatch, smtp_server):
    use_smtp(monkeypatch, smtp_server)
    assert client.post("/api/smtp/test", json={}).status_code == 200
    sent = parsed(smtp_server.mails[0])
    assert sent["List-Unsubscribe"] is None and "STOP" not in sent.get_body().get_content()


def test_destinataire_oppose_409_sans_reservation_ni_envoi(client, monkeypatch, smtp_server):
    use_smtp(monkeypatch, smtp_server)
    application.record_opposition("Jane@Acme.com", "refus_oral")
    response = client.post("/api/applications/job:1/messages", json={**MAIL, "confirm": True})
    assert response.status_code == 409
    assert response.json()["detail"] == "Ce destinataire a demandé à ne plus être contacté."
    assert stored_messages() == [] and not getattr(smtp_server, "mails", [])
    assert activity.count_today("message_email") == 0


def test_409_oppose_avant_429_meme_plafond_atteint(client, monkeypatch, smtp_server):
    use_smtp(monkeypatch, smtp_server)
    settings.save_settings({"message_daily_max": 1})
    activity.record("message_linkedin", job="job:2")  # plafond atteint
    application.record_opposition("jane@acme.com", "désinscription")
    assert client.post("/api/applications/job:1/messages", json={**MAIL, "confirm": True}).status_code == 409
    other = {**MAIL, "to": "autre@acme.com", "confirm": True}
    assert client.post("/api/applications/job:1/messages", json=other).status_code == 429


def test_contact_lie_oppose_bloque_aussi(client, monkeypatch, smtp_server):
    use_smtp(monkeypatch, smtp_server)
    application.update_application("job:1", contacts=[{"full_name": "Jane", "email": "jane@acme.com"}])
    with db_session.transaction() as s:  # filet de sécurité : opposed_at posé sans empreinte
        s.execute(update(models.CompanyContact).values(opposed_at=func.now()))
    response = client.post("/api/applications/job:1/messages", json={**MAIL, "confirm": True})
    assert response.status_code == 409 and stored_messages() == []


def test_linkedin_pied_ajoute_par_le_serveur(client, monkeypatch, accept_risks):
    accept_risks()
    monkeypatch.setattr(session, "login", lambda *a, **k: None)

    class Driver:
        def quit(self) -> None:
            pass

    monkeypatch.setattr(browser, "create_driver", Driver)
    seen = {}

    def fake_send(driver, url, body, caution, *, footer, check, timeout=12.0):
        seen.update(url=url, body=body, footer=footer)
        return {"channel": "linkedin", "mode": "message", "to": url, "truncated": False}

    monkeypatch.setattr(outreach, "send_linkedin_message", fake_send)
    url = "https://www.linkedin.com/in/jane-doe/"
    footer = client.get("/api/applications/job:1/message-draft", params={"to": url}).json()["footer"]
    body = {"channel": "linkedin", "to": url, "body": "Bonjour Jane", "confirm": True}
    task = client.post("/api/applications/job:1/messages", json=body).json()
    assert wait_done(client, task["id"])["status"] == "terminée"
    assert (
        seen["footer"]
        == footer
        == "Au sujet de votre offre « Senior Python Developer ». Pour ne plus être contacté(e), répondez STOP."
    )
    assert seen["body"] == "Bonjour Jane"
    assert stored_messages()[0].body == "Bonjour Jane\n\n" + footer


def test_cv_exports_and_ats_issues(client):
    """Téléchargement PDF / Word nommé pour le recruteur ; écarts ATS exposés dans le détail de la candidature."""
    detail = client.get("/api/applications/job:1").json()
    assert any("Section manquante" in issue for issue in detail["ats_issues"])
    assert client.put("/api/settings", json={"full_name": "Ada Lovelace"}).status_code == 200
    pdf = client.get("/api/applications/job:1/cv.pdf")
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
    assert 'filename="Ada-Lovelace-CV-Senior-Python-Developer.pdf"' in pdf.headers["content-disposition"]
    docx = client.get("/api/applications/job:1/cv.docx")
    assert docx.status_code == 200 and docx.content.startswith(b"PK")
    assert client.get("/api/applications/job:1/cv.odt").status_code == 422
    assert client.get("/api/applications/inconnue/cv.pdf").status_code == 404
