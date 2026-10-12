"""Sources et consentements aux risques (I-16, vague 3) : ``backend/01`` § 5.8 et § 5.8.1."""

import pytest
from fastapi.testclient import TestClient

from linkedin_market import __main__ as cli
from linkedin_market import activity, consent
from linkedin_market.web.app import create_app
from linkedin_market.web.tasks import TaskManager

pytestmark = pytest.mark.db

VERSION = "2026-10"


@pytest.fixture
def client():
    manager = TaskManager(persist=False)
    with TestClient(create_app(tasks=manager, token=None), base_url="http://127.0.0.1:8765") as c:
        yield c
    manager.shutdown()


@pytest.fixture
def credentials(monkeypatch):
    monkeypatch.setenv("LINKEDIN_EMAIL", "me@example.com")
    monkeypatch.setenv("LINKEDIN_PASSWORD", "x")


def source(client, code: str) -> dict:
    return next(row for row in client.get("/api/sources").json() if row["code"] == code)


def test_liste_des_sources_sans_consentement(client):
    rows = {row["code"]: row for row in client.get("/api/sources").json()}
    assert set(rows) == {"linkedin", "malt", "wttj", "france_travail", "indeed"}
    assert rows["linkedin"]["enabled"] is True and rows["linkedin"]["consent"] is None
    assert rows["linkedin"]["consent_required"] is True and rows["linkedin"]["blocked"] is None
    assert rows["indeed"]["access_mode"] == "import_manuel" and rows["indeed"]["consent_required"] is False
    assert "configured" in rows["france_travail"]  # seul champ propre à France Travail


def test_accepter_puis_revoquer(client):
    response = client.post("/api/sources/linkedin/consent", json={"risk_version": VERSION, "accepted": True})
    assert response.status_code == 201
    body = response.json()
    assert body["consent_required"] is False and body["consent"]["risk_version"] == VERSION
    assert body["consent"]["accepted_at"].endswith("Z")
    # Ré-accepter la même version remplace le consentement actif sans erreur.
    assert (
        client.post("/api/sources/linkedin/consent", json={"risk_version": VERSION, "accepted": True}).status_code
        == 201
    )
    assert client.delete("/api/sources/linkedin/consent").status_code == 204
    assert source(client, "linkedin")["consent_required"] is True
    assert client.delete("/api/sources/linkedin/consent").status_code == 404


@pytest.mark.parametrize(
    ("code", "body", "status", "detail"),
    [
        ("inconnue", {"risk_version": VERSION, "accepted": True}, 404, "Source inconnue."),
        ("indeed", {"risk_version": VERSION, "accepted": True}, 422, "Cette source ne demande pas de consentement."),
        ("malt", {"risk_version": VERSION, "accepted": True}, 409, "La source « Malt » n'est pas encore disponible."),
        (
            "linkedin",
            {"risk_version": VERSION, "accepted": False},
            422,
            "Accepte explicitement les risques (accepted: true).",
        ),
        ("linkedin", {"risk_version": VERSION}, 422, "Accepte explicitement les risques (accepted: true)."),
        (
            "linkedin",
            {"risk_version": "2025-01", "accepted": True},
            409,
            "Le texte des risques a changé : relis la nouvelle version.",
        ),
    ],
)
def test_erreurs_du_consentement(client, code, body, status, detail):
    response = client.post(f"/api/sources/{code}/consent", json=body)
    assert response.status_code == status and response.json()["detail"] == detail


def test_revoquer_une_source_inconnue(client):
    assert client.delete("/api/sources/inconnue/consent").status_code == 404


def test_run_avec_collecte_sans_consentement_repond_428(client, credentials):
    response = client.post("/api/runs", json={"keyword": "python"})
    assert response.status_code == 428
    assert response.json() == {
        "detail": (
            f"Accepte d'abord les risques liés à LinkedIn (version {VERSION}) avant toute action dans le navigateur."
        ),
        "source": "linkedin",
        "risk_version": VERSION,
    }


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/api/profile/import-linkedin"),
    ],
)
def test_actions_navigateur_sans_consentement_repondent_428(client, credentials, method, path):
    assert getattr(client, method)(path).status_code == 428


def test_ordre_des_controles_422_avant_428_avant_409(client, monkeypatch):
    monkeypatch.delenv("LINKEDIN_EMAIL", raising=False)
    # 422 : la source est invalide, même sans consentement ni identifiants.
    assert client.post("/api/runs", json={"keyword": "python", "sources": ["inconnue"]}).status_code == 422
    # 428 : consentement manquant, même sans identifiants.
    assert client.post("/api/runs", json={"keyword": "python"}).status_code == 428
    # 409 : identifiants absents une fois le consentement donné.
    client.post("/api/sources/linkedin/consent", json={"risk_version": VERSION, "accepted": True})
    assert client.post("/api/runs", json={"keyword": "python"}).status_code == 409


@pytest.mark.parametrize(
    ("sources", "detail"),
    [
        (["inconnue"], "sources : source inconnue « inconnue »."),
        (["indeed"], "sources : Indeed s'importe à la main, elle ne se collecte pas."),
        (["malt"], "sources : la source « Malt » n'est pas encore disponible."),
    ],
)
def test_run_refuse_les_sources_invalides(client, sources, detail):
    response = client.post("/api/runs", json={"keyword": "python", "sources": sources})
    assert response.status_code == 422 and detail in response.json()["detail"]


def test_run_sans_collecte_ignore_les_sources_et_le_consentement(client):
    response = client.post("/api/runs", json={"scrape": False, "sources": ["indeed"], "keyword": "python"})
    assert response.status_code != 428 and response.status_code != 422


def test_run_exige_au_moins_une_source(client):
    assert client.post("/api/runs", json={"keyword": "python", "sources": []}).status_code == 422


def test_revocation_entre_la_creation_et_le_demarrage_de_la_tache(client):
    client.post("/api/sources/linkedin/consent", json={"risk_version": VERSION, "accepted": True})
    consent.require("linkedin")
    client.delete("/api/sources/linkedin/consent")
    with pytest.raises(consent.SourceConsentRequired):  # revérifié au début de la tâche
        consent.require("linkedin")


def test_blocage_propre_a_chaque_source(client):
    activity.record("blocked", where="collecte", source="linkedin")
    assert source(client, "linkedin")["blocked"]["where"] == "collecte"
    assert source(client, "malt")["blocked"] is None
    with pytest.raises(activity.BlockedByLinkedIn):
        activity.guard_not_blocked("linkedin")
    activity.guard_not_blocked("malt")  # un blocage LinkedIn ne bloque pas une autre source
    assert client.post("/api/block/clear", json={"source": "inconnue"}).status_code == 404
    assert client.post("/api/block/clear", json={"source": "linkedin"}).json() == {"ok": True}
    assert source(client, "linkedin")["blocked"] is None


def test_block_clear_sans_corps_leve_le_blocage_de_linkedin(client):
    activity.record("blocked", where="collecte")  # sans ``source`` : compatibilité, LinkedIn
    assert source(client, "linkedin")["blocked"] is not None
    assert client.post("/api/block/clear").json() == {"ok": True}
    assert source(client, "linkedin")["blocked"] is None


def test_cli_avec_collecte_refuse_de_demarrer_sans_consentement(monkeypatch):
    monkeypatch.setattr(cli, "collect_request", lambda args: cli.SearchRequest(keyword="python"))
    with pytest.raises(SystemExit) as exc:
        cli.main(["-k", "python"])
    assert str(exc.value) == (
        f"Accepte d'abord les risques liés à LinkedIn (version {VERSION}) dans l'interface web : "
        "Paramètres → Sources et consentements."
    )
