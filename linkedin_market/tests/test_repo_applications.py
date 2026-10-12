"""Candidatures, historique et contacts en base (vague 1, sous-lot 1c)."""

import re

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from linkedin_market import application, config, storage
from linkedin_market.db import models, session
from linkedin_market.models import Item
from linkedin_market.web.app import create_app
from linkedin_market.web.tasks import TaskManager

pytestmark = pytest.mark.db
ISO_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


@pytest.fixture
def entry(data_dir):
    folder = config.APPLICATIONS_DIR / "20261008-acme-dev"
    folder.mkdir(parents=True)
    (folder / "cv.md").write_text("# CV", encoding="utf-8")
    storage.append_items(
        [
            Item(
                kind="job",
                id="job:5",
                url="https://www.linkedin.com/jobs/view/5/",
                title="Dev",
                text="Python",
                keyword="python",
                company="Acme",
            )
        ]
    )
    application.save_applications(
        {
            "job:5": {
                "title": "Dev",
                "url": "u",
                "score": 81.5,
                "folder": str(folder),
                "generator": "local",
                "status": "préparée",
            }
        }
    )
    return folder


@pytest.fixture
def client():
    manager = TaskManager(persist=False)
    with TestClient(create_app(tasks=manager, token=None), base_url="http://127.0.0.1:8765") as c:
        yield c
    manager.shutdown()


def test_dossier_relatif_en_base_absolu_dans_l_api(entry):
    with session.transaction() as s:
        row = s.execute(select(models.Application)).scalar_one()
        assert row.folder_path == "applications/20261008-acme-dev" and row.score == 81.5
        item = s.get(models.CollectedItem, row.item_id)
        assert item.purge_after is None  # une offre candidatée n'est pas purgée (I-10)
    got = application.get_application("job:5")
    assert got["folder"] == str(entry) and application.read_files(got)["cv"] == "# CV"
    assert ISO_UTC.match(got["prepared_at"]) and got["updated_at"] is None  # C-1, C-2
    assert got["title"] == "Dev" and got["url"] == "https://www.linkedin.com/jobs/view/5/"


def test_i03_pas_de_regression_automatique_patch_libre(entry, client):
    application.advance_status("job:5", "postulée")
    application.advance_status("job:5", "contactée")
    assert application.get_application("job:5")["status"] == "postulée"
    application.advance_status("job:5", "entretien")
    assert application.get_application("job:5")["status"] == "entretien"
    patched = client.patch("/api/applications/job:5", json={"status": "préparée"}).json()
    assert patched["status"] == "préparée"  # seule voie de régression
    assert patched["history"][-1]["kind"] == "status" and patched["history"][-1]["status"] == "préparée"
    with pytest.raises(application.ApplicationNotFound):
        application.advance_status("job:404", "postulée")


def test_historique_status_apply_et_anciens_noms(entry):
    application.add_history("job:5", "candidature", state="postulée", detail="")
    application.add_history("job:5", "statut", status="entretien")
    application.add_history("job:5", "message", channel="email", to="a@b.co", extrait="Bonjour")
    history = application.get_application("job:5")["history"]
    assert [h["kind"] for h in history] == ["apply", "status", "message"]
    assert history[0]["state"] == "postulée" and all(ISO_UTC.match(h["at"]) for h in history)
    with pytest.raises(ValueError):
        application.add_history("job:5", "inconnu")


def test_contacts_full_name_et_sources(entry, client):
    body = {
        "contacts": [
            {
                "full_name": "Jane Doe",
                "profile_url": "https://www.linkedin.com/in/jane-doe/",
                "source": "linkedin_equipe_recrutement",
            },
            {"full_name": "Ana", "email": "ana@acme.com"},
        ]
    }
    contacts = client.patch("/api/applications/job:5", json=body).json()["contacts"]
    assert [c["full_name"] for c in contacts] == ["Jane Doe", "Ana"]
    assert contacts[0]["source"] == "linkedin_equipe_recrutement" and contacts[1]["source"] == "manuel"
    assert "name" not in contacts[0]
    with session.transaction() as s:
        jane_id = s.execute(
            select(models.CompanyContact.id).where(models.CompanyContact.full_name == "Jane Doe")
        ).scalar()
    # Mise à jour en place : le contact garde son id (messages liés) ; un contact retiré est supprimé.
    body["contacts"][0]["headline"] = "Talent Lead"
    contacts = client.patch("/api/applications/job:5", json={"contacts": body["contacts"][:1]}).json()["contacts"]
    assert len(contacts) == 1 and contacts[0]["headline"] == "Talent Lead"
    with session.transaction() as s:
        rows = s.execute(select(models.CompanyContact)).scalars().all()
        assert [r.id for r in rows] == [jane_id] and rows[0].legal_basis == "interet_legitime"
    # Ancien format interne (« name », source « linkedin ») converti.
    application.update_application("job:5", contacts=[{"name": "John", "source": "linkedin"}])
    john = application.get_application("job:5")["contacts"][0]
    assert ISO_UTC.match(john.pop("collected_at"))  # V4-8 : lecture seule
    assert john == {
        "full_name": "John",
        "profile_url": "",
        "email": None,
        "headline": "",
        "source": "linkedin_equipe_recrutement",
        "opposed": False,
    }


def test_preparer_a_nouveau_garde_statut_contacts_historique(entry):
    application.advance_status("job:5", "postulée")
    application.update_application("job:5", contacts=[{"full_name": "Jane"}])
    application.add_history("job:5", "note", texte="relance")
    item = storage.find_item("job:5")
    job = {"id": "job:5", "title": "Dev", "company": "Acme", "url": item.url, "score": 90.0}
    folder = application.write(application.Application("# CV 2", "Bonjour", "-", "local"), item, job)
    got = application.get_application("job:5")
    assert got["status"] == "postulée" and got["contacts"][0]["full_name"] == "Jane" and got["history"]
    assert got["folder"] == str(folder) and got["score"] == 90.0
    assert [r["id"] for r in application.list_applications()] == ["job:5"]


def test_filtre_user_id(entry, db_user_id):
    from linkedin_market.db import users
    from linkedin_market.db.repositories import applications as applications_repo

    with session.transaction() as s:
        other = users.ensure_user(s, "autre@example.org")
        assert applications_repo.get(s, other, "job:5") is None and applications_repo.list_all(s, other) == []
        assert applications_repo.get(s, db_user_id, "job:5") is not None


def test_i14_mark_sent_pose_last_contacted_at_et_purge_a_trois_ans(entry):
    """Non-régression (vague 4) : l'envoi repousse la purge du contact lié à 3 ans (I-14)."""
    from datetime import timedelta

    from linkedin_market.db.repositories import outreach as outreach_repo
    from linkedin_market.db.repositories import user_transaction

    application.update_application("job:5", contacts=[{"full_name": "Jane", "email": "jane@acme.com"}])
    with user_transaction() as (s, uid):
        contact = s.execute(select(models.CompanyContact)).scalar_one()
        assert contact.last_contacted_at is None
        message_id = outreach_repo.insert_confirmed(
            s,
            uid,
            contact.application_id,
            {"channel": "email", "recipient": "jane@acme.com", "subject": "s", "body": "b", "contact_id": contact.id},
        )
        outreach_repo.mark_sent(s, uid, message_id, "<id@x>")
        s.expire_all()
        contact = s.execute(select(models.CompanyContact)).scalar_one()
        assert contact.last_contacted_at is not None
        assert (
            contact.purge_after - contact.last_contacted_at
            == outreach_repo.CONTACT_RETENTION
            == timedelta(days=3 * 365)
        )
