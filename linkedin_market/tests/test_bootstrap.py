"""Préparation de la base au démarrage (vague 1, sous-lot 1d)."""

import uuid

import pytest
from sqlalchemy import select

from linkedin_market.db import bootstrap, models, session, users

pytestmark = pytest.mark.db


def test_demarrage_passe_en_echec_une_tache_en_cours(db_user_id, data_dir):
    with session.transaction() as s:
        for status in ("en_cours", "attente_saisie", "terminée"):
            s.add(models.Task(id=uuid.uuid7(), user_id=db_user_id, kind="run", title=status, status=status))
    result = bootstrap.prepare_database(data_dir)
    assert result["interrupted_tasks"] == 2
    with session.transaction() as s:
        rows = {t.title: t for t in s.execute(select(models.Task)).scalars()}
    assert rows["en_cours"].status == "échec" and rows["en_cours"].error == bootstrap.INTERRUPTED
    assert rows["en_cours"].finished_at is not None and rows["terminée"].status == "terminée"
    assert bootstrap.prepare_database(data_dir)["interrupted_tasks"] == 0  # idempotent


def test_utilisateur_cree_si_base_vide(data_dir, monkeypatch):
    with session.transaction() as s:
        s.execute(models.User.__table__.delete())
    monkeypatch.setenv("LINKEDIN_EMAIL", "Alex.Exemple@Exemple.TEST")
    bootstrap.prepare_database(data_dir)
    with session.transaction() as s:
        assert s.execute(select(models.User.email)).scalars().all() == ["alex.exemple@exemple.test"]
        assert users.current_user_id(s)
    monkeypatch.setenv("LINKEDIN_EMAIL", "autre@exemple.test")
    bootstrap.prepare_database(data_dir)  # base non vide : aucun second utilisateur (A5)
    with session.transaction() as s:
        assert len(s.execute(select(models.User.id)).all()) == 1
    # Le test suivant retrouve l'utilisateur de test (TRUNCATE + création par conftest).


@pytest.mark.db
def test_base_injoignable_sans_mot_de_passe(monkeypatch, data_dir):
    secret = "MotDePasseTresSecret42"
    monkeypatch.setenv("LM_DATABASE_URL", f"postgresql+psycopg://lm:{secret}@127.0.0.1:1/linkedin_market_test")
    session.dispose_engine()
    try:
        with pytest.raises(SystemExit) as caught:
            bootstrap.prepare_database(data_dir)
    finally:
        monkeypatch.undo()
        session.dispose_engine()
    message = str(caught.value.code)
    assert "injoignable" in message and secret not in message and "***" in message
    assert "LM_DATABASE_URL" in message


def test_base_non_configuree(monkeypatch, data_dir):
    monkeypatch.delenv("LM_DATABASE_URL")
    session.dispose_engine()
    try:
        with pytest.raises(SystemExit, match="LM_DATABASE_URL n'est pas définie"):
            bootstrap.prepare_database(data_dir)
    finally:
        monkeypatch.undo()
        session.dispose_engine()
