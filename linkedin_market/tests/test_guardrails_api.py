"""Garde-fous de l'API (vague 2, contrat ``backend/01`` § 1.2) : G-1, G-2, G-3, G-4, G-5, G-11, G-12."""

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from linkedin_market import activity, application, config, interpreter, pipeline, settings
from linkedin_market.__main__ import parse_args
from linkedin_market.models import SearchRequest
from linkedin_market.web import guards, services
from linkedin_market.web.app import create_app
from linkedin_market.web.tasks import TaskManager

pytestmark = pytest.mark.db

MAIL = {"channel": "email", "to": "jane@acme.com", "subject": "Candidature", "body": "Bonjour", "confirm": True}
INSECURE = (
    "SMTP_SECURITY=none n'est permis que vers un serveur SMTP local (127.0.0.1) : "
    "choisis starttls ou ssl dans le fichier .env."
)


@pytest.fixture
def client():
    manager = TaskManager(persist=False)
    with TestClient(create_app(tasks=manager, token=None), base_url="http://127.0.0.1:8765") as c:
        yield c
    manager.shutdown()


@pytest.fixture
def prepared(data_dir):
    """Une candidature préparée ``job:1`` (dossier sous le répertoire de données)."""
    folder = data_dir / "applications" / "20261009-acme-dev"
    folder.mkdir(parents=True)
    (folder / "message.txt").write_text("Bonjour", encoding="utf-8")
    application.save_applications(
        {
            "job:1": {
                "title": "Senior Python Developer",
                "company": "Acme",
                "url": "https://www.linkedin.com/jobs/view/1/",
                "score": 80,
                "folder": str(folder),
                "status": "préparée",
            }
        }
    )
    return "job:1"


# ----------------------------------------------------------------------- G-1
def test_g1_champ_manquant_et_plusieurs_erreurs(client):
    response = client.post("/api/applications/job:1/messages", json={"channel": "email", "body": "x"})
    assert response.status_code == 422
    assert response.json() == {"detail": "to : champ requis"}
    several = client.post("/api/applications/job:1/messages", json={"channel": "sms"})
    assert (
        several.json()["detail"]
        == "channel : doit être l'un de : email, linkedin ; to : champ requis ; body : champ requis"
    )


def test_g1_literal(client):
    response = client.post("/api/applications/job:1/messages", json={**MAIL, "channel": "sms"})
    assert response.status_code == 422
    assert response.json()["detail"] == "channel : doit être l'un de : email, linkedin"


def test_g1_entier_invalide(client):
    response = client.post("/api/runs", json={"keyword": "x", "scrape": False, "max_items": "beaucoup"})
    assert response.status_code == 422
    assert response.json()["detail"] == "max_items : doit être un entier"
    label = client.put("/api/jobs/job:1/label", json={"value": "oui"})
    assert label.status_code == 422 and label.json()["detail"] == "value : doit être un entier"


def test_g1_field_validator_sans_prefixe(client):
    response = client.patch("/api/applications/job:1", json={"contacts": [{"email": "pas-un-email"}]})
    assert response.status_code == 422
    assert response.json()["detail"] == "email : Adresse email invalide."  # message du validateur, point conservé


def test_g1_json_mal_forme(client):
    response = client.post("/api/runs", content=b'{"keyword": ', headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert response.json() == {"detail": "Corps JSON invalide."}


def test_g1_traductions_unitaires():
    from linkedin_market.web.app import validation_detail

    errors = [
        {"type": "bool_parsing", "loc": ("body", "scrape"), "msg": "x"},
        {"type": "float_parsing", "loc": ("query", "min_score"), "msg": "x"},
        {"type": "string_type", "loc": ("body", "contacts", 0, "full_name"), "msg": "x"},
        {"type": "greater_than_equal", "loc": ("body", "n"), "msg": "x", "ctx": {"ge": 1}},
        {"type": "less_than_equal", "loc": ("body", "n"), "msg": "x", "ctx": {"le": 60}},
        {"type": "dict_type", "loc": ("body",), "msg": "x"},
    ]
    assert validation_detail(errors) == (
        "scrape : doit valoir true ou false ; min_score : doit être un nombre ; full_name : doit être un texte ; "
        "n : doit être ≥ 1 ; n : doit être ≤ 60 ; corps de la requête : doit être un objet JSON"
    )


# ----------------------------------------------------------------------- G-2
def test_g2_secondes_jusqu_a_minuit():
    assert guards.seconds_until_local_midnight(datetime(2026, 10, 9, 23, 0, 0).astimezone()) == 3600
    assert guards.seconds_until_local_midnight(datetime(2026, 10, 9, 23, 59, 59, 900000).astimezone()) == 1
    assert guards.seconds_until_local_midnight(datetime(2026, 10, 9, 0, 0, 0).astimezone()) == 86400


@pytest.fixture
def frozen_clock(monkeypatch):
    monkeypatch.setattr(guards, "_now", lambda: datetime(2026, 10, 9, 22, 30, 0).astimezone())
    return 5400


def test_g2_retry_after_candidature(client, prepared, monkeypatch, frozen_clock):
    monkeypatch.setenv("LINKEDIN_EMAIL", "me@example.com")
    monkeypatch.setenv("LINKEDIN_PASSWORD", "x")
    settings.save_settings({"apply_daily_max": 1})
    activity.record("apply", job="job:x")
    response = client.post(f"/api/applications/{prepared}/apply")
    assert response.status_code == 429
    assert response.json()["detail"] == (
        "Plafond quotidien atteint (1 candidatures par jour) : réessaie demain ou relève-le dans les paramètres."
    )
    assert response.headers["Retry-After"] == str(frozen_clock)


def test_g2_retry_after_messages(client, prepared, monkeypatch, frozen_clock):
    monkeypatch.setenv("SMTP_HOST", "127.0.0.1")
    monkeypatch.setenv("SMTP_FROM", "me@example.com")
    monkeypatch.setenv("SMTP_SECURITY", "none")
    settings.save_settings({"message_daily_max": 0})
    response = client.post(f"/api/applications/{prepared}/messages", json=MAIL)
    assert response.status_code == 429
    assert response.json()["detail"].startswith("Plafond quotidien atteint (0 messages par jour)")
    assert response.headers["Retry-After"] == str(frozen_clock)
    # Canal LinkedIn : même en-tête.
    monkeypatch.setenv("LINKEDIN_EMAIL", "me@example.com")
    monkeypatch.setenv("LINKEDIN_PASSWORD", "x")
    linkedin = {
        "channel": "linkedin",
        "to": "https://www.linkedin.com/in/jane-doe/",
        "body": "Bonjour",
        "confirm": True,
    }
    response = client.post(f"/api/applications/{prepared}/messages", json=linkedin)
    assert response.status_code == 429 and response.headers["Retry-After"] == str(frozen_clock)


# ----------------------------------------------------------------- G-3, G-11
def test_g3_run_max_items_plafonne_par_la_prudence(client):
    body = {"keyword": "x", "scrape": False, "caution": "prudent", "max_items": 16}
    response = client.post("/api/runs", json=body)
    assert response.status_code == 422
    assert response.json()["detail"] == "max_items ne peut pas dépasser 15 en prudence « prudent »."
    settings.save_settings({"caution": "normal"})
    response = client.post("/api/runs", json={"keyword": "x", "scrape": False, "max_items": 31})
    assert response.json()["detail"] == "max_items ne peut pas dépasser 30 en prudence « normal »."


def test_g3_defauts_et_min_defensif():
    assert SearchRequest(keyword="x").caution == "prudent" and SearchRequest(keyword="x").max_items == 15
    assert pipeline.capped(SearchRequest(keyword="x", caution="prudent", max_items=200)).max_items == 15
    assert pipeline.capped(SearchRequest(keyword="x", caution="agressif", max_items=60)).max_items == 60
    assert pipeline.capped(SearchRequest(keyword="x", caution="inconnue", max_items=60)).max_items == 15
    # Réglage enregistré pour une prudence plus large : ramené au plafond de la prudence demandée.
    settings.save_settings({"keyword": "x", "caution": "agressif", "max_items": 60})
    assert services.build_request({"caution": "prudent"}).max_items == 15


def test_g11_minimum_max_items(client):
    for response in (
        client.put("/api/settings", json={"max_items": 0}),
        client.post("/api/runs", json={"keyword": "x", "scrape": False, "max_items": 0}),
    ):
        assert response.status_code == 422
        assert response.json()["detail"] == "max_items doit être au moins 1."
    assert client.put("/api/settings", json={"max_items": "abc"}).json()["detail"] == "max_items doit être un entier."
    assert client.put("/api/settings", json={"caution": "prudent", "max_items": 1}).status_code == 200


# ----------------------------------------------------------------------- G-4
def test_g4_api_refuse_none(client):
    for response in (
        client.put("/api/settings", json={"interpreter": "none"}),
        client.post("/api/runs", json={"keyword": "x", "scrape": False, "interpreter": "none"}),
    ):
        assert response.status_code == 422
        assert response.json()["detail"] == "interpreter doit être l'un de : claude, local."


def test_g4_cli_refuse_none():
    with pytest.raises(SystemExit):
        parse_args(["--interpreter", "none"])
    assert parse_args(["--interpreter", "local"]).interpreter == "local"


def test_g4_variable_none_lue_local(monkeypatch, caplog):
    monkeypatch.setenv("LINKEDIN_INTERPRETER", "none")
    with caplog.at_level("WARNING"):
        assert config.interpreter_name() == "local"
    assert "LINKEDIN_INTERPRETER=none" in caplog.text
    summary = {"counts": {"job": 1}, "gap": {}, "model": {}, "rising": {}}
    text, used = interpreter.interpret(summary)
    assert used == "local" and text  # le mode local produit toujours une interprétation (I-11)
    assert interpreter.interpret(summary, "none")[1] == "local"


# ----------------------------------------------------------------------- G-5
def test_g5_none_vers_serveur_distant_refuse(client, prepared, monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "smtp.exemple.fr")
    monkeypatch.setenv("SMTP_FROM", "me@example.com")
    monkeypatch.setenv("SMTP_SECURITY", "none")
    response = client.post(f"/api/applications/{prepared}/messages", json=MAIL)
    assert response.status_code == 503 and response.json()["detail"] == INSECURE
    test = client.post("/api/smtp/test", json={})
    assert test.status_code == 503 and test.json()["detail"] == INSECURE
    monkeypatch.setenv("SMTP_SECURITY", "starttls")
    assert config.smtp_security_error(config.smtp_config()) is None


def test_g5_none_vers_relais_local_admis():
    for host in ("127.0.0.1", "localhost", "::1", "[::1]"):
        assert config.smtp_security_error({"host": host, "security": "none"}) is None
    assert config.smtp_security_error({"host": "smtp.exemple.fr", "security": "none"}) == INSECURE
    unknown = config.smtp_security_error({"host": "127.0.0.1", "security": "plain"})
    assert unknown == (
        "SMTP_SECURITY=plain n'est pas reconnu : choisis starttls ou ssl dans le fichier .env "
        "(none seulement pour un serveur SMTP local, 127.0.0.1)."
    )


# ---------------------------------------------------------------------- G-12
def test_g12_contact_email(client):
    response = client.put("/api/settings", json={"contact_email": "pas-un-mail"})
    assert response.status_code == 422
    assert response.json()["detail"] == "contact_email doit être une adresse email valide."
    assert client.put("/api/settings", json={"contact_email": "a@b.io, c@d.io"}).status_code == 422
    assert client.put("/api/settings", json={"contact_email": " me@example.com "}).json()["contact_email"] == (
        "me@example.com"
    )
    assert client.put("/api/settings", json={"contact_email": ""}).json()["contact_email"] == ""
