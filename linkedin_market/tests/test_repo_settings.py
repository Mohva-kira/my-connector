"""Réglages, réponses mémorisées et profil en base (vague 1, sous-lot 1b)."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import update

from linkedin_market import config, settings, storage
from linkedin_market.db import models, session, users
from linkedin_market.db.repositories import settings as settings_repo
from linkedin_market.web.app import create_app
from linkedin_market.web.tasks import TaskManager

pytestmark = pytest.mark.db


@pytest.fixture
def client():
    manager = TaskManager(persist=False)
    with TestClient(create_app(tasks=manager, token=None), base_url="http://127.0.0.1:8765") as c:
        yield c
    manager.shutdown()


def test_defaut_prudent():
    assert config.DEFAULT_CAUTION == "prudent"
    values = settings.load_settings()
    assert values["caution"] == "prudent" and values["max_items"] is None and values["answers"] == {}


def test_aller_retour_reglages_et_reponses():
    saved = settings.save_settings(
        {
            "keyword": " Data Engineer ",
            "caution": "normal",
            "max_items": 30,
            "apply_min_score": "72.5",
            "answers": {"Années d'expérience ?": "12"},
            "inconnue": 1,
        }
    )
    assert saved["keyword"] == "Data Engineer" and saved["max_items"] == 30 and saved["apply_min_score"] == 72.5
    assert saved["answers"] == {"annees d'experience ?": "12"}
    settings.remember_answer("Are you legally authorized?", "Yes")
    assert settings.find_answer("ARE YOU LEGALLY AUTHORIZED?") == "Yes"
    assert settings.find_answer("Autre question") is None
    reloaded = settings.load_settings()
    assert reloaded["keyword"] == "Data Engineer" and len(reloaded["answers"]) == 2
    # answers remplace l'ensemble (contrat backend/01 § 5.2)
    assert settings.save_settings({"answers": {"q": "r"}})["answers"] == {"q": "r"}


def test_max_items_plafonne_par_la_prudence(client):
    response = client.put("/api/settings", json={"caution": "prudent", "max_items": 16})
    assert response.status_code == 422
    assert response.json()["detail"] == "max_items ne peut pas dépasser 15 en prudence « prudent »."
    assert client.put("/api/settings", json={"caution": "agressif", "max_items": 60}).status_code == 200
    # Revenir en prudent avec max_items = 60 est refusé, le réglage reste intact.
    assert client.put("/api/settings", json={"caution": "prudent"}).status_code == 422
    assert client.get("/api/settings").json()["caution"] == "agressif"
    assert client.put("/api/settings", json={"max_items": 61}).status_code == 422
    assert client.put("/api/settings", json={"caution": "prudent", "max_items": None}).json()["max_items"] is None


def test_interpreteur_none_refuse_et_lu_local(client, monkeypatch):
    response = client.put("/api/settings", json={"interpreter": "none"})
    assert response.status_code == 422
    assert response.json()["detail"] == "interpreter doit être l'un de : claude, local."
    assert client.post("/api/runs", json={"keyword": "x", "scrape": False, "interpreter": "none"}).status_code == 422
    # Valeur héritée (fichier importé, autre version) : lue comme local.
    real = settings_repo.get_settings
    monkeypatch.setattr(
        settings_repo, "get_settings", lambda s, uid, lock=False: {**real(s, uid), "interpreter": "none"}
    )
    assert settings.load_settings()["interpreter"] == "local"
    assert settings.read_interpreter("claude") == "claude"


def test_surcharges_de_profil():
    settings.add_skill("Rust")
    settings.add_skill("Go")
    settings.remove_skill("python")
    settings.save_profile_settings({"title": "Data Engineer", "years_experience": 12, "seniority": "Senior"})
    overrides = settings.load_profile_settings()
    assert overrides["added_skills"] == ["rust", "go"] and overrides["removed_skills"] == ["python"]
    assert (overrides["title"], overrides["years_experience"], overrides["seniority"]) == (
        "Data Engineer",
        12,
        "senior",
    )
    settings.remove_skill("rust")  # le dernier choix gagne
    overrides = settings.load_profile_settings()
    assert overrides["added_skills"] == ["go"] and set(overrides["removed_skills"]) == {"python", "rust"}
    with pytest.raises(settings.SettingsError):
        settings.save_profile_settings({"seniority": "dieu"})


def test_cache_du_profil_linkedin():
    assert storage.load_linkedin_profile_text() == ""
    storage.save_linkedin_profile_text("Data Engineer\nPython, Spark et SQL")
    assert storage.load_linkedin_profile_text().startswith("Data Engineer")
    with session.transaction() as s:
        uid = users.current_user_id(s)
        row = s.query(models.CandidateProfile).filter_by(user_id=uid).one()
        assert row.origin == "linkedin" and "python" in row.skills and row.synced_at is not None


def test_utilisateur_unique_a5():
    with session.transaction() as s:
        uid = users.current_user_id(s)
        assert users.ensure_initial_user(s) == uid  # base non vide : aucune création
        s.execute(update(models.User).values(email="autre@example.org"))
    with session.transaction() as s:
        s.execute(models.User.__table__.delete())
    with pytest.raises(users.UserResolutionError, match="Aucun utilisateur"), session.transaction() as s:
        users.current_user_id(s)
    with session.transaction() as s:
        users.ensure_initial_user(s)
        assert users.current_user_id(s)
