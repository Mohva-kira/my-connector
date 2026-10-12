"""Plafonds quotidiens réservés en base (G-6, E-11), SMTP en clair (G-5) et journaux masqués (G-9)."""

import socket
import threading
import time
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from linkedin_market import activity, application, config, contacts, settings
from linkedin_market.db import models, session
from linkedin_market.db.repositories import settings as settings_repo
from linkedin_market.tests.browser_fixtures import url_of
from linkedin_market.web import services
from linkedin_market.web.app import create_app
from linkedin_market.web.tasks import TaskManager

pytestmark = pytest.mark.db

INSECURE = (
    "SMTP_SECURITY=none n'est permis que vers un serveur SMTP local (127.0.0.1) : "
    "choisis starttls ou ssl dans le fichier .env."
)


class FakeCtx:
    """Doublure de ``TaskContext`` pour appeler une fonction de tâche directement."""

    def __init__(self) -> None:
        self.logs: list[str] = []

    def log(self, message: str) -> None:
        self.logs.append(message)

    def check(self) -> None:
        pass

    def should_stop(self) -> bool:
        return False

    def manual_check(self, message: str) -> None:
        self.logs.append(message)

    def set_status(self, status: str) -> None:
        pass


@pytest.fixture
def manager():
    tasks = TaskManager(persist=False)
    yield tasks
    tasks.shutdown(wait=True)


@pytest.fixture
def client(manager):
    with TestClient(create_app(tasks=manager, token=None), base_url="http://127.0.0.1:8765") as c:
        yield c


def prepare(data_dir, count: int = 1) -> list[str]:
    """``count`` candidatures préparées (``job:1``…), dossiers sous le répertoire de données."""
    refs = []
    for i in range(1, count + 1):
        folder = data_dir / "applications" / f"20261009-acme-{i}"
        folder.mkdir(parents=True)
        (folder / "message.txt").write_text("Bonjour", encoding="utf-8")
        application.save_applications(
            {
                f"job:{i}": {
                    "title": f"Poste {i}",
                    "company": "Acme",
                    "url": f"https://www.linkedin.com/jobs/view/{i}/",
                    "score": 80,
                    "folder": str(folder),
                    "status": "préparée",
                }
            }
        )
        refs.append(f"job:{i}")
    return refs


def use_smtp(monkeypatch, port: int, host: str = "127.0.0.1") -> None:
    monkeypatch.setenv("SMTP_HOST", host)
    monkeypatch.setenv("SMTP_PORT", str(port))
    monkeypatch.setenv("SMTP_SECURITY", "none")
    monkeypatch.setenv("SMTP_FROM", "me@example.com")


def closed_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def mail(to: str = "jane.doe@acme.com") -> dict:
    return {"channel": "email", "to": to, "subject": "Candidature", "body": "Bonjour Jane", "attach_cv": False,
            "confirm": True}  # fmt: skip


def wait_done(manager: TaskManager, timeout: float = 30) -> None:
    end = time.time() + timeout
    while time.time() < end:
        if not manager.active():
            return
        time.sleep(0.05)
    raise AssertionError("tâches toujours actives")


def messages() -> list[models.OutreachMessage]:
    with session.transaction() as s:
        return list(s.execute(select(models.OutreachMessage).order_by(models.OutreachMessage.confirmed_at)).scalars())


# ------------------------------------------------------------------------ G-6
def test_g6_huit_envois_simultanes_plafond_trois(client, manager, data_dir, monkeypatch, smtp_server):
    use_smtp(monkeypatch, smtp_server.server_address[1])
    settings.save_settings({"message_daily_max": 3})
    refs = prepare(data_dir, 8)
    barrier = threading.Barrier(len(refs))
    responses: dict[str, object] = {}

    def send(ref: str) -> None:
        barrier.wait()
        responses[ref] = client.post(f"/api/applications/{ref}/messages", json=mail())

    threads = [threading.Thread(target=send, args=(ref,)) for ref in refs]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    codes = sorted(r.status_code for r in responses.values())
    assert codes == [202, 202, 202, 429, 429, 429, 429, 429]
    refused = [r for r in responses.values() if r.status_code == 429]
    assert all(r.json()["detail"].startswith("Plafond quotidien atteint (3 messages par jour)") for r in refused)
    assert all(int(r.headers["Retry-After"]) >= 1 for r in refused)
    wait_done(manager)
    rows = messages()
    assert len(rows) == 3 and {r.status for r in rows} == {"envoyé"}
    assert all(r.sent_at is not None and r.smtp_message_id for r in rows)
    assert activity.count_today("message_email") == 3
    assert services.quotas()["messages"] == {"used": 3, "limit": 3}


def test_g6_echec_smtp_libere_la_place(client, manager, data_dir, monkeypatch, smtp_server):
    settings.save_settings({"message_daily_max": 1})
    (ref,) = prepare(data_dir)
    use_smtp(monkeypatch, closed_port())  # serveur injoignable : l'envoi échoue
    task = client.post(f"/api/applications/{ref}/messages", json=mail()).json()
    wait_done(manager)
    failed = manager.get(task["id"])
    assert failed.status == "échec" and "Envoi impossible" in failed.error
    assert [r.status for r in messages()] == ["échec"]  # aucune relance automatique
    use_smtp(monkeypatch, smtp_server.server_address[1])
    retry = client.post(f"/api/applications/{ref}/messages", json=mail())
    assert retry.status_code == 202  # la place a été rendue
    wait_done(manager)
    assert [r.status for r in messages()] == ["échec", "envoyé"]
    assert client.post(f"/api/applications/{ref}/messages", json=mail()).status_code == 429


def test_g6_conflit_rend_la_reservation(client, manager, data_dir, monkeypatch):
    use_smtp(monkeypatch, closed_port())
    (ref,) = prepare(data_dir)
    gate = threading.Event()
    manager.submit("message", "occupé", lambda ctx: gate.wait(10), meta={"key": ref, "job_id": ref})
    response = client.post(f"/api/applications/{ref}/messages", json=mail())
    assert (
        response.status_code == 409
        and response.json()["detail"] == "Un envoi est déjà en cours pour cette candidature."
    )
    gate.set()
    assert messages() == []  # la réservation faite avant le contrôle a été annulée


def test_g6_plafond_message_revérifié_au_debut_de_la_tache(data_dir, monkeypatch, smtp_server):
    use_smtp(monkeypatch, smtp_server.server_address[1])
    (ref,) = prepare(data_dir)
    message_id = services.reserve_message(ref, channel="email", to="jane@acme.com", subject="s", body="b",
                                          attach_cv=False)  # fmt: skip
    settings.save_settings({"message_daily_max": 0})  # plafond abaissé entre l'API et le début de la tâche
    with pytest.raises(activity.QuotaExceeded):
        services.send_message(FakeCtx(), ref, message_id=message_id, channel="email", to="jane@acme.com",
                              subject="s", body="b", attach_cv=False)  # fmt: skip
    assert [r.status for r in messages()] == ["échec"]
    assert not getattr(smtp_server, "mails", [])


def test_g6_plafond_candidature_revérifié_sous_verrou(data_dir, monkeypatch, accept_risks):
    accept_risks()
    (ref,) = prepare(data_dir)
    settings.save_settings({"apply_daily_max": 1})
    activity.record("apply", job="job:x")  # candidature acceptée entre-temps par une autre tâche
    locks = []
    real = settings_repo.get_settings

    def spy(s, uid, *, lock=False):
        locks.append(lock)
        return real(s, uid, lock=lock)

    monkeypatch.setattr(settings_repo, "get_settings", spy)
    monkeypatch.setattr("linkedin_market.browser.create_driver", lambda: pytest.fail("aucun navigateur attendu"))
    with pytest.raises(activity.QuotaExceeded, match=r"\(1 candidatures par jour\)"):
        services.apply_job(FakeCtx(), ref)
    assert locks and locks[0] is True  # FOR UPDATE sur user_settings au début de la tâche


# ------------------------------------------------------------------------ G-5
def test_g5_revérifié_au_debut_de_la_tache(data_dir, monkeypatch):
    (ref,) = prepare(data_dir)
    use_smtp(monkeypatch, 25)
    message_id = services.reserve_message(ref, channel="email", to="jane@acme.com", subject="s", body="b",
                                          attach_cv=False)  # fmt: skip
    monkeypatch.setenv("SMTP_HOST", "smtp.exemple.fr")  # .env modifié après l'acceptation par l'API
    with pytest.raises(RuntimeError, match="SMTP_SECURITY=none") as raised:
        services.send_message(FakeCtx(), ref, message_id=message_id, channel="email", to="jane@acme.com",
                              subject="s", body="b", attach_cv=False)  # fmt: skip
    assert str(raised.value) == INSECURE
    assert [r.status for r in messages()] == ["échec"]


def test_g5_none_vers_relais_local_envoie(client, manager, data_dir, monkeypatch, smtp_server):
    use_smtp(monkeypatch, smtp_server.server_address[1])
    (ref,) = prepare(data_dir)
    assert client.post(f"/api/applications/{ref}/messages", json=mail()).status_code == 202
    wait_done(manager)
    assert [r.status for r in messages()] == ["envoyé"]


# ------------------------------------------------------------------------ G-9
def test_g9_journal_d_un_envoi_masque(client, manager, data_dir, monkeypatch, smtp_server):
    use_smtp(monkeypatch, smtp_server.server_address[1])
    (ref,) = prepare(data_dir)
    task = client.post(f"/api/applications/{ref}/messages", json=mail("jane.doe@acme.com")).json()
    wait_done(manager)
    done = client.get(f"/api/tasks/{task['id']}").json()
    assert done["status"] == "terminée", done
    logs = "\n".join(done["logs"])
    assert "j***@acme.com" in logs and "jane.doe@acme.com" not in logs and "Bonjour Jane" not in logs
    events = client.get("/api/monitor").json()["recent_events"]
    assert "jane.doe@acme.com" not in str(events) and any(e.get("to") == "j***@acme.com" for e in events)
    # L'historique de la candidature garde l'information complète.
    assert application.get_application(ref)["history"][-1]["to"] == "jane.doe@acme.com"
    assert messages()[0].recipient == "jane.doe@acme.com"


def test_g9_erreur_de_tache_masquee(client, manager, data_dir, monkeypatch):
    (ref,) = prepare(data_dir)
    use_smtp(monkeypatch, closed_port())
    task = client.post(f"/api/applications/{ref}/messages", json=mail("jane.doe@acme.com")).json()
    wait_done(manager)
    failed = client.get(f"/api/tasks/{task['id']}").json()
    assert failed["status"] == "échec"
    assert "jane.doe@acme.com" not in failed["error"] + "\n".join(failed["logs"])


def test_g9_recherche_de_contacts_masquee(driver):
    local = {
        "https://www.linkedin.com/in/jane-doe/": url_of("contact-jane.html"),
        "https://www.linkedin.com/in/john-roe/": url_of("contact-john.html"),
    }
    logs: list[str] = []
    found = contacts.find_hiring_team(
        driver, url_of("hiring_team.html"), config.Caution("test", 1, 0.0, 0.0), log_fn=logs.append,
        to_overlay=lambda u: local[u],
    )  # fmt: skip
    assert found[0]["full_name"] == "Jane Doe" and found[0]["email"] == "jane@acme.com"  # valeurs complètes
    joined = "\n".join(logs)
    assert "J. D. : j***@acme.com" in joined and "J. R. : email non affiché par LinkedIn" in joined
    assert "Jane Doe" not in joined and "jane@acme.com" not in joined and "John Roe" not in joined


def test_g9_journal_d_une_tache_contacts(manager):
    """Filet de sécurité du gestionnaire : adresses et profils masqués dans ``logs`` même s'ils y arrivent bruts."""

    def leak(ctx):
        ctx.log("Contact : jane.doe@acme.com https://www.linkedin.com/in/jane-doe/")
        raise RuntimeError("Le destinataire a été refusé : jane.doe@acme.com")

    task = manager.submit("contacts", "Chercher les recruteurs de l'offre", leak)
    wait_done(manager)
    text = "\n".join(task.logs) + task.error
    assert "jane.doe@acme.com" not in text and "jane-doe" not in text
    assert "j***@acme.com" in text and "linkedin.com/in/j***" in text


def test_reservation_liee_au_contact(data_dir):
    (ref,) = prepare(data_dir)
    application.update_application(
        ref, contacts=[{"full_name": "Jane Doe", "email": "Jane.Doe@acme.com", "source": "manuel"}]
    )
    message_id = services.reserve_message(ref, channel="email", to="jane.doe@acme.com", subject="s", body="b",
                                          attach_cv=False)  # fmt: skip
    with session.transaction() as s:
        row = s.get(models.OutreachMessage, UUID(str(message_id)))
        assert row.contact_id is not None and row.status == "confirmé"


# --------------------------------------------- envoi incertain, abandon avant démarrage
UNCERTAIN = "L'envoi n'a pas pu être confirmé : vérifie la conversation sur LinkedIn avant de réessayer."
PROFILE = "https://www.linkedin.com/in/jane-doe/"


@pytest.fixture
def linkedin_ready(monkeypatch):
    monkeypatch.setenv("LINKEDIN_EMAIL", "me@example.com")
    monkeypatch.setenv("LINKEDIN_PASSWORD", "x")
    monkeypatch.setattr("linkedin_market.session.login", lambda *a, **k: None)

    class Driver:
        def quit(self) -> None:
            pass

    monkeypatch.setattr("linkedin_market.browser.create_driver", Driver)


def test_envoi_linkedin_incertain_reste_confirme(client, manager, data_dir, monkeypatch, linkedin_ready, accept_risks):
    accept_risks()
    from linkedin_market import outreach

    (ref,) = prepare(data_dir)
    calls = []

    def uncertain(*args, **kwargs):
        calls.append(args)
        raise outreach.UncertainDelivery(UNCERTAIN)

    monkeypatch.setattr(outreach, "send_linkedin_message", uncertain)
    body = {"channel": "linkedin", "to": PROFILE, "body": "Bonjour Jane", "confirm": True}
    task = client.post(f"/api/applications/{ref}/messages", json=body).json()
    wait_done(manager)
    done = manager.get(task["id"])
    assert done.status == "échec" and done.error == UNCERTAIN
    (row,) = messages()
    assert row.status == "confirmé" and row.error == UNCERTAIN and row.sent_at is None  # « à vérifier »
    assert services.message_count_today() == 1  # la place reste occupée (I-07)
    time.sleep(0.3)
    assert len(calls) == 1 and len(messages()) == 1  # aucune nouvelle tentative
    assert application.get_application(ref)["status"] == "préparée"


def test_tache_annulee_avant_de_demarrer_rend_la_place(client, manager, data_dir, linkedin_ready, accept_risks):
    accept_risks()
    (ref,) = prepare(data_dir)
    settings.save_settings({"message_daily_max": 1})
    gate = threading.Event()
    holder = manager.submit("rescore", "occupe le navigateur", lambda ctx: gate.wait(10), browser=True)
    body = {"channel": "linkedin", "to": PROFILE, "body": "Bonjour Jane", "confirm": True}
    task = client.post(f"/api/applications/{ref}/messages", json=body).json()
    assert task["status"] == "en_attente"
    assert client.post(f"/api/tasks/{task['id']}/cancel").status_code == 200
    end = time.time() + 10
    while manager.get(task["id"]).status != "annulée" and time.time() < end:
        time.sleep(0.05)
    gate.set()
    wait_done(manager)
    assert manager.get(task["id"]).status == "annulée" and holder.status == "terminée"
    (row,) = messages()
    assert row.status == "échec" and row.error == "Tâche annulée avant l'envoi."
    assert services.message_count_today() == 0


def test_erreur_a_la_fermeture_smtp_ignoree(data_dir, monkeypatch, smtp_server):
    import smtplib

    from linkedin_market import outreach

    real_quit = smtplib.SMTP.quit

    def broken_quit(self):
        real_quit(self)  # QUIT bien envoyé (le serveur de test libère sa connexion), puis erreur de fermeture
        raise ConnectionResetError("connexion fermée par le serveur")

    monkeypatch.setattr(smtplib.SMTP, "quit", broken_quit)
    use_smtp(monkeypatch, smtp_server.server_address[1])
    result, message_id = outreach.deliver_email("jane@acme.com", "Candidature", "Bonjour")
    assert result["to"] == "jane@acme.com" and message_id.startswith("<")
    (ref,) = prepare(data_dir)
    reserved = services.reserve_message(ref, channel="email", to="jane@acme.com", subject="s", body="b",
                                        attach_cv=False)  # fmt: skip
    services.send_message(FakeCtx(), ref, message_id=reserved, channel="email", to="jane@acme.com", subject="s",
                          body="b", attach_cv=False)  # fmt: skip
    assert [r.status for r in messages()] == ["envoyé"]
