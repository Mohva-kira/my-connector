"""Fixtures communes : fichiers de données isolés dans ``tmp_path`` et base PostgreSQL de test jetable.

Base de test (``backend/03`` § 8) : ``LM_TEST_DATABASE_URL`` désigne une base dont le nom finit par ``_test``,
distincte de ``LM_DATABASE_URL`` ; sinon la session pytest s'arrête. Pendant les tests, ``LM_DATABASE_URL`` est
remplacée par l'URL de test (ou retirée) : aucun code ne peut atteindre la vraie base. ``LM_SECRET_KEY`` est une
clé éphémère générée pour la session : jamais celle du ``.env``, et rien n'est écrit dans le ``.env``. Les
identifiants LinkedIn, SMTP et la clé Anthropic du ``.env`` sont retirés de l'environnement : sans eux, aucune tâche
lancée par un test ne peut se connecter à un vrai service.
"""

import os
from pathlib import Path
from uuid import UUID

import pytest
from alembic import command
from alembic.config import Config
from cryptography.fernet import Fernet
from sqlalchemy import Connection, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

from linkedin_market import config
from linkedin_market import consent as consent_mod
from linkedin_market.db import session, users

# Fixtures partagées (navigateur réel, dossier de travail, serveur SMTP local) : disponibles dans tous les tests.
from linkedin_market.tests.browser_fixtures import driver, workdir  # noqa: F401
from linkedin_market.tests.test_outreach import smtp_server  # noqa: F401

TEST_URL_VAR = "LM_TEST_DATABASE_URL"
SKIP_MESSAGE = "Tests PostgreSQL ignorés : définis LM_TEST_DATABASE_URL (base dont le nom finit par _test)."
ALEMBIC_INI = Path(config.__file__).resolve().parent / "alembic.ini"
_LOCAL_HOSTS = {"", "localhost", "127.0.0.1", "::1"}
# Tables conservées entre deux tests : données de référence et version du schéma.
_KEPT_TABLES = {"sources", "alembic_version"}
# Secrets du .env de l'utilisateur jamais visibles des tests (aucun appel réel à LinkedIn, SMTP ou Claude) : un test
# qui en a besoin pose une valeur factice avec monkeypatch.setenv.
_REAL_SECRETS = (
    "LINKEDIN_EMAIL",
    "LINKEDIN_PASSWORD",
    "SMTP_HOST",
    "SMTP_PORT",
    "SMTP_USER",
    "SMTP_PASSWORD",
    "SMTP_FROM",
    "SMTP_SECURITY",
    "ANTHROPIC_API_KEY",
    "FRANCE_TRAVAIL_CLIENT_ID",
    "FRANCE_TRAVAIL_CLIENT_SECRET",
)


def _target(url: str) -> tuple[str, int, str]:
    parsed = make_url(url)
    host = (parsed.host or "").lower()
    return ("local" if host in _LOCAL_HOSTS else host, parsed.port or 5432, parsed.database or "")


def check_test_database_url(test_url: str, real_url: str | None) -> str | None:
    """Message d'erreur si ``test_url`` n'est pas une base de test sûre, ``None`` sinon."""
    try:
        database = make_url(test_url).database or ""
    except ArgumentError:
        return f"{TEST_URL_VAR} n'est pas une URL SQLAlchemy valide (postgresql+psycopg://…/<nom>_test)."
    if not database.endswith("_test"):
        return (
            f"{TEST_URL_VAR} désigne la base « {database} » : son nom doit finir par _test "
            "(protection de ta vraie base, elle est vidée à chaque session de tests)."
        )
    if real_url:
        try:
            same = _target(test_url) == _target(real_url)
        except ArgumentError:
            same = False
        if same:
            return f"{TEST_URL_VAR} désigne la même base que LM_DATABASE_URL : utilise une base de test dédiée."
    return None


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "db: test sur la base PostgreSQL de test (LM_TEST_DATABASE_URL)")
    # Le .env racine a déjà été chargé à l'import de linkedin_market.config : on neutralise la vraie base.
    real_url = os.environ.pop("LM_DATABASE_URL", None)
    for name in _REAL_SECRETS:
        os.environ.pop(name, None)
    os.environ["LM_SECRET_KEY"] = Fernet.generate_key().decode("ascii")
    test_url = (os.environ.get(TEST_URL_VAR) or "").strip()
    if not test_url:
        return
    error = check_test_database_url(test_url, real_url)
    if error:
        pytest.exit(error, returncode=4)
    os.environ["LM_DATABASE_URL"] = test_url


def alembic_config(connection: Connection) -> Config:
    """Configuration Alembic du projet, branchée sur ``connection`` (aucune URL lue dans un fichier)."""
    cfg = Config(str(ALEMBIC_INI))
    cfg.attributes["connection"] = connection
    return cfg


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    root = tmp_path / "data"
    paths = {
        "DATA_DIR": root,
        "SESSION_FILE": root / "session.json",
        "HISTORY_FILE": root / "history.jsonl",
        "LABELS_FILE": root / "labels.json",
        "MODEL_FILE": root / "matching_model.json",
        "REPORTS_DIR": root / "reports",
        "APPLICATIONS_DIR": root / "applications",
        "APPLICATIONS_FILE": root / "applications.json",
        "LINKEDIN_PROFILE_CACHE": root / "linkedin_profile.txt",
        "SETTINGS_FILE": root / "settings.json",
        "PROFILE_SETTINGS_FILE": root / "profile_settings.json",
        "ACTIVITY_FILE": root / "activity.jsonl",
        "TASKS_FILE": root / "tasks.jsonl",
    }
    for name, value in paths.items():
        monkeypatch.setattr(config, name, value)
    config.ensure_data_dirs()
    return root


@pytest.fixture(scope="session")
def db_schema():
    """Schéma recréé une fois par session : ``DROP SCHEMA public`` puis ``alembic upgrade head`` (teste 0001)."""
    if not (os.environ.get(TEST_URL_VAR) or "").strip():
        pytest.skip(SKIP_MESSAGE)
    engine = session.get_engine()
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
        command.upgrade(alembic_config(conn), "head")
    yield engine
    session.dispose_engine()


@pytest.fixture(autouse=True)
def _db_isolation(request: pytest.FixtureRequest) -> None:
    """Avant chaque test marqué ``db`` : ``TRUNCATE`` (hors ``sources``) puis création de l'utilisateur de test."""
    if request.node.get_closest_marker("db") is None:
        return
    engine = request.getfixturevalue("db_schema")
    with engine.begin() as conn:
        tables = conn.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename")
        ).scalars()
        names = [name for name in tables if name not in _KEPT_TABLES]
        if names:
            conn.execute(text(f"TRUNCATE TABLE {', '.join(names)} RESTART IDENTITY CASCADE"))
    with session.transaction() as s:
        users.ensure_user(s, "test@example.org")


@pytest.fixture
def db_user_id() -> UUID:
    """``id`` de l'utilisateur de test (test marqué ``db``)."""
    with session.transaction() as s:
        return users.current_user_id(s)


@pytest.fixture
def accept_risks():
    """Accepte les risques d'une source ``navigateur`` (I-16) pour les tests qui lancent une action navigateur.

    Les tests existants, antérieurs à la popup de risques, n'ont pas de consentement enregistré : sans cet appel,
    toute action navigateur (collecte, candidature, contacts, message LinkedIn, import de profil, compétence) y
    répondrait désormais ``428`` à la place du comportement attendu.
    """

    def _accept(source_code: str = "linkedin") -> None:
        risk_version = consent_mod.get_public(source_code)["risk_version"]
        consent_mod.accept(source_code, risk_version=risk_version, accepted=True)

    return _accept
