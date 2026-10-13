"""Import manuel d'une offre Indeed (vague 5, R-6) : ``POST /api/items/import``, aucun appel réseau."""

import hashlib
import json
import socket
import time
import urllib.request
from datetime import timedelta

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from linkedin_market import config, settings
from linkedin_market.db import models, session
from linkedin_market.web.app import create_app
from linkedin_market.web.routers.items import external_id_of, normalize_url
from linkedin_market.web.tasks import TaskManager

pytestmark = pytest.mark.db

URL = "https://fr.indeed.com/viewjob?jk=abc123def456&tk=suivi&from=serp#details"
TEXT = (
    "Nous recherchons un Data Engineer freelance pour construire des pipelines Python, Spark et Airflow sur AWS. "
    "Mission de six mois en télétravail, TJM selon profil, démarrage rapide, équipe data de dix personnes."
)
BODY = {
    "source_code": "indeed",
    "url": URL,
    "title": "Data Engineer",
    "company": "Acme",
    "location": "Paris",
    "text": TEXT,
}


@pytest.fixture
def client():
    manager = TaskManager(persist=False)
    with TestClient(create_app(tasks=manager, token=None), base_url="http://127.0.0.1:8765") as c:
        yield c
    manager.shutdown(wait=True)


@pytest.fixture
def profile(monkeypatch, tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "profil.json").write_text(
        json.dumps({"title": "Data Engineer", "skills": ["python", "spark"], "text": "Data Engineer Python"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "PROJECT_ROOT", root)


@pytest.fixture
def no_network(monkeypatch):
    """Toute tentative de sortie vers Internet (HTTP, sockets sortants, navigateur) fait échouer le test."""
    attempts: list[str] = []

    def refuse(name):
        def _refuse(*args, **kwargs):
            attempts.append(name)
            raise AssertionError(f"appel réseau interdit : {name}")

        return _refuse

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse("httpx"))
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", refuse("httpx async"))
    monkeypatch.setattr(urllib.request, "urlopen", refuse("urllib"))
    monkeypatch.setattr(socket, "getaddrinfo", refuse("dns"))
    monkeypatch.setattr("linkedin_market.browser.create_driver", refuse("navigateur"))
    yield attempts
    assert attempts == []


def import_item(client, **overrides):
    return client.post("/api/items/import", json={**BODY, **overrides})


def stored(external_id: str) -> models.CollectedItem:
    with session.transaction() as s:
        row = s.execute(
            select(models.CollectedItem).where(models.CollectedItem.external_id == external_id)
        ).scalar_one()
        s.expunge(row)
        return row


def wait_final(client, task_id: str) -> dict:
    end = time.time() + 30
    while time.time() < end:
        task = client.get(f"/api/tasks/{task_id}").json()
        if task["status"] in ("terminée", "échec", "annulée"):
            return task
        time.sleep(0.05)
    raise AssertionError(task)


# ----------------------------------------------------------------- création
def test_import_cree_l_offre_sans_appel_reseau(client, no_network):
    settings.save_settings({"keyword": "Data Engineer"})
    response = import_item(client)
    expected = "indeed-" + hashlib.sha256(b"https://fr.indeed.com/viewjob?jk=abc123def456").hexdigest()[:16]
    assert response.status_code == 201, response.text
    assert response.json() == {
        "id": f"job:{expected}",
        "source_code": "indeed",
        "title": "Data Engineer",
        "company": "Acme",
        "location": "Paris",
        "url": "https://fr.indeed.com/viewjob?jk=abc123def456",
        "keyword": "Data Engineer",
    }
    row = stored(expected)
    assert (row.source_code, row.kind, row.keywords) == ("indeed", "job", ["data engineer"])
    assert row.text == TEXT and {"python", "spark", "airflow"} <= set(row.skills)
    assert timedelta(days=90) - (row.purge_after - row.collected_at) < timedelta(seconds=1)  # I-10 : 90 jours
    assert row.posted_at is not None


def test_doublon_409_sans_reecriture(client, no_network):
    settings.save_settings({"keyword": "Data Engineer"})
    first = import_item(client)
    assert first.status_code == 201
    # Même offre malgré une adresse écrite autrement (hôte en majuscules, suivi, fragment) : refusée, rien réécrit.
    again = import_item(client, url="https://FR.indeed.com/viewjob?utm_x=1&jk=abc123def456#x", text=TEXT + " v2")
    assert again.status_code == 409 and again.json()["detail"] == "Cette offre est déjà dans l'historique."
    assert stored(first.json()["id"].removeprefix("job:")).text == TEXT


def test_normalisation_de_l_adresse():
    assert (
        normalize_url("HTTPS://Fr.Indeed.COM/viewjob?vjk=2&jk=1&tk=x#frag")
        == "https://fr.indeed.com/viewjob?jk=1&vjk=2"
    )
    assert normalize_url("https://www.indeed.com/cmp/Acme/jobs/Data-Engineer-1?from=serp#x") == (
        "https://www.indeed.com/cmp/Acme/jobs/Data-Engineer-1"
    )
    assert external_id_of(URL) == external_id_of("https://fr.indeed.com/viewjob?jk=abc123def456")
    assert external_id_of(URL) != external_id_of("https://fr.indeed.com/viewjob?jk=autre")
    assert len(external_id_of(URL)) == len("indeed-") + 16


def test_l_offre_importee_est_scoree_a_la_prochaine_analyse(client, profile):
    created = import_item(client, keyword="Data Engineer").json()
    task = wait_final(
        client,
        client.post("/api/runs", json={"keyword": "Data Engineer", "scrape": False, "interpreter": "local"}).json()[
            "id"
        ],
    )
    assert task["status"] == "terminée" and task["result"]["jobs"] == 1
    (job,) = client.get("/api/jobs").json()["jobs"]
    assert job["id"] == created["id"] and job["source_code"] == "indeed"
    assert [j["id"] for j in client.get("/api/jobs", params={"source": "indeed"}).json()["jobs"]] == [created["id"]]


# --------------------------------------------------------------------- mot-clé
def test_priorite_du_mot_cle(client, profile):
    # Aucun mot-clé nulle part : 422.
    refused = import_item(client)
    assert refused.status_code == 422
    assert refused.json()["detail"] == "keyword : renseigne le mot-clé du poste visé (Paramètres ou ce formulaire)."
    # Réglages.
    settings.save_settings({"keyword": "Mot réglages"})
    assert import_item(client, url="https://fr.indeed.com/viewjob?jk=1").json()["keyword"] == "Mot réglages"
    # Dernier rapport (avant les réglages).
    task = wait_final(
        client,
        client.post("/api/runs", json={"keyword": "Mot rapport", "scrape": False, "interpreter": "local"}).json()["id"],
    )
    assert task["status"] == "terminée"
    assert import_item(client, url="https://fr.indeed.com/viewjob?jk=2").json()["keyword"] == "Mot rapport"
    # Mot-clé du corps : prioritaire.
    assert (
        import_item(client, url="https://fr.indeed.com/viewjob?jk=3", keyword="Mot corps").json()["keyword"]
        == "Mot corps"
    )


# ------------------------------------------------------------------ validation
@pytest.mark.parametrize(
    ("override", "detail"),
    [
        ({"source_code": "linkedin"}, "source_code : doit être l'un de : indeed"),
        ({"url": "http://fr.indeed.com/viewjob?jk=1"}, "url : colle l'adresse https de l'offre Indeed."),
        ({"url": "https://www.linkedin.com/jobs/view/1/"}, "url : colle l'adresse https de l'offre Indeed."),
        ({"url": "https://indeed.com.evil.org/viewjob?jk=1"}, "url : colle l'adresse https de l'offre Indeed."),
        ({"url": "https://evil.org/?u=indeed.com"}, "url : colle l'adresse https de l'offre Indeed."),
        ({"url": "pas une adresse"}, "url : colle l'adresse https de l'offre Indeed."),
        ({"url": "https://fr.indeed.com/" + "a" * 2000}, "url : colle l'adresse https de l'offre Indeed."),
        ({"text": "trop court"}, "text : colle le texte complet de l'offre (200 caractères minimum)."),
        ({"text": "x" * 199}, "text : colle le texte complet de l'offre (200 caractères minimum)."),
        ({"text": "x" * 20001}, "text : texte trop long (20 000 caractères maximum)."),
        ({"title": "   "}, "title : renseigne le titre de l'offre (150 caractères maximum)."),
        ({"title": "t" * 151}, "title : renseigne le titre de l'offre (150 caractères maximum)."),
        ({"company": "c" * 81}, "company : entreprise trop longue (80 caractères maximum)."),
        ({"location": "l" * 121}, "location : lieu trop long (120 caractères maximum)."),
        ({"keyword": "k" * 81}, "keyword : mot-clé trop long (80 caractères maximum)."),
    ],
)
def test_validation_422(client, override, detail):
    settings.save_settings({"keyword": "Data Engineer"})
    response = import_item(client, **override)
    assert response.status_code == 422 and response.json()["detail"] == detail


def test_bornes_acceptees(client):
    settings.save_settings({"keyword": "Data Engineer"})
    body = {"text": "x" * 200, "title": "t" * 150, "company": "c" * 80, "location": "l" * 120, "keyword": "k" * 80}
    assert import_item(client, **body).status_code == 201
    assert import_item(client, url="https://indeed.com/viewjob?jk=2", text="y" * 20000).status_code == 201
    assert import_item(client, url="https://indeed.com/viewjob?jk=3", company="", location="").status_code == 201


def test_champs_requis(client):
    response = client.post("/api/items/import", json={"source_code": "indeed"})
    assert response.status_code == 422
    assert "url : champ requis" in response.json()["detail"] and "text : champ requis" in response.json()["detail"]
