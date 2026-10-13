"""Socle base de données (vague 1, sous-lot 1a) : migration 0001, contraintes, helpers, garde-fous de test."""

import os
import stat
import subprocess
import sys
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest
from alembic import command
from cryptography.fernet import Fernet
from sqlalchemy import delete, insert, inspect, select, text, update
from sqlalchemy.exc import DBAPIError, IntegrityError

from linkedin_market import config
from linkedin_market.db import crypto, models, session, users, utc_iso, utcnow
from linkedin_market.tests.conftest import alembic_config, check_test_database_url

REPO_ROOT = Path(config.__file__).resolve().parent.parent
TEST_URL = "postgresql+psycopg://lm:lm@127.0.0.1:55432/linkedin_market_test"


# ----------------------------------------------------------------- hors base
def test_utc_iso_format_et_none():
    paris = timezone(timedelta(hours=2))
    assert utc_iso(datetime(2026, 10, 8, 14, 32, 5, 987654, tzinfo=paris)) == "2026-10-08T12:32:05Z"
    assert utc_iso(None) is None
    now = utcnow()
    assert now.tzinfo is not None and now.utcoffset() == timedelta(0)


def test_database_url_absente_message_clair(monkeypatch):
    monkeypatch.delenv("LM_DATABASE_URL", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://email_analyzer/base")  # jamais lue (P-1)
    with pytest.raises(config.DatabaseConfigError, match="LM_DATABASE_URL n'est pas définie"):
        config.database_url()


def test_database_url_driver_psycopg(monkeypatch):
    monkeypatch.setenv("LM_DATABASE_URL", "postgresql://u:p@127.0.0.1:5432/linkedin_market")
    assert config.database_url() == "postgresql+psycopg://u:p@127.0.0.1:5432/linkedin_market"
    monkeypatch.setenv("LM_DATABASE_URL", "postgresql+psycopg://u:p@127.0.0.1/x")
    assert config.database_url() == "postgresql+psycopg://u:p@127.0.0.1/x"


def test_base_de_test_sans_suffixe_refusee():
    error = check_test_database_url("postgresql+psycopg://lm:lm@127.0.0.1:5432/linkedin_market", None)
    assert error and "_test" in error
    assert check_test_database_url("postgresql+psycopg://lm:lm@127.0.0.1:5432/linkedin_market_test", None) is None


def test_base_de_test_identique_a_la_vraie_refusee():
    real = "postgresql+psycopg://a:b@localhost:5432/linkedin_market_test"
    error = check_test_database_url("postgresql+psycopg://lm:lm@127.0.0.1/linkedin_market_test", real)
    assert error and "même base" in error
    assert check_test_database_url(TEST_URL, "postgresql+psycopg://a:b@127.0.0.1:5432/linkedin_market") is None


def test_session_pytest_arretee_si_base_sans_suffixe():
    env = {**os.environ, "LM_TEST_DATABASE_URL": "postgresql+psycopg://lm:lm@127.0.0.1:55432/linkedin_market"}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "--co", "-q", __file__],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 4
    assert "doit finir par _test" in result.stdout + result.stderr


def test_cle_de_test_vient_de_l_environnement():
    key = os.environ["LM_SECRET_KEY"]
    Fernet(key.encode())  # clé valide
    token = crypto.encrypt(b'{"li_at": "x"}')
    assert crypto.decrypt(token) == b'{"li_at": "x"}'


def test_generation_de_cle_refusee_dans_le_vrai_env_en_test(monkeypatch):
    monkeypatch.delenv("LM_SECRET_KEY")
    before = crypto.ENV_FILE.read_bytes() if crypto.ENV_FILE.exists() else None
    with pytest.raises(crypto.SecretKeyError, match="absente en test"):
        crypto.secret_key()
    assert (crypto.ENV_FILE.read_bytes() if crypto.ENV_FILE.exists() else None) == before


def test_generation_de_cle_dans_un_env_temporaire(tmp_path, monkeypatch):
    monkeypatch.delenv("LM_SECRET_KEY")
    env_file = tmp_path / ".env"
    env_file.write_text("AUTRE=1", encoding="utf-8")
    key = crypto.secret_key(env_file)
    assert env_file.read_text(encoding="utf-8") == f"AUTRE=1\nLM_SECRET_KEY={key}\n"
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
    assert os.environ["LM_SECRET_KEY"] == key
    assert crypto.secret_key(env_file) == key  # pas de seconde génération


def test_decrypt_avec_autre_cle_message_lisible(monkeypatch):
    token = crypto.encrypt(b"cookies")
    monkeypatch.setenv("LM_SECRET_KEY", Fernet.generate_key().decode())
    with pytest.raises(crypto.SecretKeyError, match="reconnecte-toi"):
        crypto.decrypt(token)


def test_plafonds_de_prudence_alignes_sur_config():
    assert models.CAUTION_MAX_ITEMS == {name: c.max_items for name, c in config.CAUTION_LEVELS.items()}


# ----------------------------------------------------------------- base PostgreSQL


def _task(s, user_id: UUID) -> UUID:
    return s.execute(insert(models.Task).values(user_id=user_id, kind="run").returning(models.Task.id)).scalar_one()


def _item(s, user_id: UUID, kind: str = "job", text_value: str | None = "texte", external_id: str = "1") -> UUID:
    return s.execute(
        insert(models.CollectedItem)
        .values(user_id=user_id, source_code="linkedin", kind=kind, external_id=external_id, text=text_value)
        .returning(models.CollectedItem.id)
    ).scalar_one()


def _report(s, user_id: UUID, task_id: UUID | None = None, slug: str = "data-engineer-2026-10-08") -> UUID:
    return s.execute(
        insert(models.Report)
        .values(
            user_id=user_id,
            task_id=task_id,
            slug=slug,
            keyword="Data Engineer",
            interpreter_used="local",
            html_path=f"reports/{slug}.html",
        )
        .returning(models.Report.id)
    ).scalar_one()


@pytest.mark.db
def test_migrations_upgrade_downgrade_upgrade(db_schema):
    with db_schema.begin() as conn:
        cfg = alembic_config(conn)
        try:
            command.downgrade(cfg, "base")
            assert set(inspect(conn).get_table_names()) == {"alembic_version"}
            functions = conn.execute(text("SELECT count(*) FROM pg_proc WHERE proname = 'reports_immutable'"))
            assert functions.scalar_one() == 0
        finally:
            command.upgrade(cfg, "head")
        tables = set(inspect(conn).get_table_names())
    assert {t.name for t in models.Base.metadata.sorted_tables} | {"alembic_version"} == tables


@pytest.mark.db
def test_alembic_check_sans_ecart(db_schema):
    with db_schema.connect() as conn:
        command.check(alembic_config(conn))  # lève AutogenerateDiffsDetected en cas d'écart


@pytest.mark.db
def test_revision_0003_active_puis_desactive_les_nouvelles_sources(db_schema):
    def enabled(conn) -> dict[str, bool]:
        return dict(conn.execute(select(models.Source.code, models.Source.enabled)).all())

    with db_schema.begin() as conn:
        cfg = alembic_config(conn)
        try:
            command.downgrade(cfg, "0002")
            assert enabled(conn) == {
                "linkedin": True,
                "malt": False,
                "wttj": False,
                "france_travail": False,
                "indeed": False,
            }
        finally:
            command.upgrade(cfg, "head")
        assert all(enabled(conn).values())
        command.check(alembic_config(conn))  # modèles == migrations : alembic check vide


@pytest.mark.db
def test_cinq_sources_de_reference_toutes_activees():
    with session.transaction() as s:
        rows = s.execute(
            select(models.Source.code, models.Source.access_mode, models.Source.risk_version, models.Source.enabled)
        ).all()
    assert sorted(rows) == sorted(
        [
            ("linkedin", "navigateur", "2026-10", True),
            ("malt", "navigateur", "2026-10", True),
            ("wttj", "navigateur", "2026-10", True),
            ("france_travail", "api_officielle", None, True),
            ("indeed", "import_manuel", None, True),
        ]
    )


@pytest.mark.db
def test_utilisateur_unique_et_reglages_par_defaut(db_user_id):
    with session.transaction() as s:
        assert users.ensure_user(s, "  TEST@example.org ") == db_user_id  # idempotent, minuscules
        assert users.current_user_id(s) == db_user_id
        settings = s.get(models.UserSettings, db_user_id)
        assert (settings.caution, settings.max_items, settings.interpreter) == ("prudent", None, "claude")
        assert settings.apply_min_score == 70.0 and isinstance(settings.apply_min_score, float)
        user = s.get(models.User, db_user_id)
        assert user.created_at.tzinfo is not None
        assert utc_iso(user.created_at).endswith("Z")


@pytest.mark.db
def test_plusieurs_utilisateurs_refuses(db_user_id):
    with session.transaction() as s:
        users.ensure_user(s, "autre@example.org")
    with pytest.raises(users.UserResolutionError, match="plusieurs utilisateurs"), session.transaction() as s:
        users.current_user_id(s)


@pytest.mark.db
def test_rapport_immuable(db_user_id):
    with session.transaction() as s:
        report_id = _report(s, db_user_id)
    with pytest.raises(DBAPIError, match="immuable"), session.transaction() as s:
        s.execute(update(models.Report).where(models.Report.id == report_id).values(interpretation="réécrit"))
    with session.transaction() as s:
        assert s.get(models.Report, report_id).interpretation == ""


@pytest.mark.db
def test_purge_d_une_tache_detache_le_rapport_sans_le_modifier(db_user_id):
    with session.transaction() as s:
        task_id = _task(s, db_user_id)
        report_id = _report(s, db_user_id, task_id)
    with session.transaction() as s:
        s.execute(delete(models.Task).where(models.Task.id == task_id))
    with session.transaction() as s:
        report = s.get(models.Report, report_id)
        assert report.task_id is None and report.slug == "data-engineer-2026-10-08"
    with pytest.raises(DBAPIError, match="immuable"), session.transaction() as s:
        s.execute(update(models.Report).where(models.Report.id == report_id).values(task_id=None, slug="autre"))


@pytest.mark.db
def test_slug_unique_par_utilisateur(db_user_id):
    with session.transaction() as s:
        _report(s, db_user_id)
    with pytest.raises(IntegrityError), session.transaction() as s:
        _report(s, db_user_id)


@pytest.mark.db
def test_chemin_absolu_refuse(db_user_id):
    for path in ("/home/x/reports/a.html", "C:\\data\\a.html", "\\\\serveur\\a.html"):
        with pytest.raises(IntegrityError, match="html_path_relative"), session.transaction() as s:
            s.execute(
                insert(models.Report).values(user_id=db_user_id, slug="a", interpreter_used="local", html_path=path)
            )


@pytest.mark.db
def test_person_avec_texte_refuse(db_user_id):
    with pytest.raises(IntegrityError, match="person_without_text"), session.transaction() as s:
        _item(s, db_user_id, kind="person", text_value="biographie")
    with session.transaction() as s:
        _item(s, db_user_id, kind="person", text_value=None)


@pytest.mark.db
def test_label_binaire(db_user_id):
    with session.transaction() as s:
        item_id = _item(s, db_user_id)
    with pytest.raises(IntegrityError, match="ck_labels_value"), session.transaction() as s:
        s.execute(insert(models.Label).values(item_id=item_id, user_id=db_user_id, value=2))
    with session.transaction() as s:
        s.execute(insert(models.Label).values(item_id=item_id, user_id=db_user_id, value=1))


@pytest.mark.db
@pytest.mark.parametrize(
    ("caution", "max_items", "ok"),
    [
        ("prudent", 15, True),
        ("prudent", 16, False),
        ("normal", 30, True),
        ("normal", 31, False),
        ("agressif", 60, True),
        ("agressif", 61, False),
        ("prudent", 0, False),
        ("prudent", None, True),
    ],
)
def test_max_items_plafonne_par_la_prudence(db_user_id, caution, max_items, ok):
    statement = (
        update(models.UserSettings)
        .where(models.UserSettings.user_id == db_user_id)
        .values(caution=caution, max_items=max_items)
    )
    if ok:
        with session.transaction() as s:
            s.execute(statement)
    else:
        with pytest.raises(IntegrityError, match="ck_user_settings_max_items"), session.transaction() as s:
            s.execute(statement)


@pytest.mark.db
def test_interpreteur_none_refuse(db_user_id):
    with pytest.raises(IntegrityError, match="ck_user_settings_interpreter"), session.transaction() as s:
        s.execute(
            update(models.UserSettings).where(models.UserSettings.user_id == db_user_id).values(interpreter="none")
        )


@pytest.mark.db
def test_un_seul_consentement_actif_par_source(db_user_id):
    consent = {"user_id": db_user_id, "source_code": "linkedin", "risk_version": "2026-10"}
    with session.transaction() as s:
        first = s.execute(insert(models.SourceConsent).values(**consent).returning(models.SourceConsent.id)).scalar()
        s.execute(insert(models.SourceConsent).values({**consent, "source_code": "malt"}))
    with pytest.raises(IntegrityError, match="ix_source_consents_user_id_source_code"), session.transaction() as s:
        s.execute(insert(models.SourceConsent).values(**consent))
    with session.transaction() as s:
        s.execute(
            update(models.SourceConsent).where(models.SourceConsent.id == first).values(revoked_at=datetime.now(UTC))
        )
        s.execute(insert(models.SourceConsent).values(**consent))
        active = s.execute(
            select(models.SourceConsent).where(
                models.SourceConsent.source_code == "linkedin", models.SourceConsent.revoked_at.is_(None)
            )
        ).all()
    assert len(active) == 1


@pytest.mark.db
def test_source_inconnue_refusee(db_user_id):
    with pytest.raises(IntegrityError), session.transaction() as s:
        s.execute(
            insert(models.SourceConsent).values(user_id=db_user_id, source_code="indeed_scraping", risk_version="x")
        )


@pytest.mark.db
def test_contact_purge_after_par_defaut_trois_ans(db_user_id):
    with session.transaction() as s:
        item_id = _item(s, db_user_id)
        application_id = s.execute(
            insert(models.Application)
            .values(user_id=db_user_id, item_id=item_id, score=72.5, generator="local", folder_path="applications/x")
            .returning(models.Application.id)
        ).scalar_one()
        contact = s.execute(
            insert(models.CompanyContact)
            .values(user_id=db_user_id, application_id=application_id, full_name="Recruteur")
            .returning(models.CompanyContact.purge_after, models.CompanyContact.collected_at)
        ).one()
        score = s.get(models.Application, application_id).score
    assert contact.purge_after - contact.collected_at >= timedelta(days=3 * 365)
    assert score == 72.5 and isinstance(score, float)


@pytest.mark.db
def test_contact_opposed_at_present_et_nul_par_defaut(db_user_id):
    assert "opposed_at" in models.CompanyContact.__table__.columns
    assert models.CompanyContact.__table__.columns["opposed_at"].nullable
    with session.transaction() as s:
        item_id = _item(s, db_user_id)
        application_id = s.execute(
            insert(models.Application)
            .values(user_id=db_user_id, item_id=item_id, score=1, generator="local", folder_path="applications/x")
            .returning(models.Application.id)
        ).scalar_one()
        opposed_at = s.execute(
            insert(models.CompanyContact)
            .values(user_id=db_user_id, application_id=application_id, full_name="Recruteur")
            .returning(models.CompanyContact.opposed_at)
        ).scalar_one()
    assert opposed_at is None


@pytest.mark.db
def test_revision_0002_ajoute_puis_retire_opposed_at(db_schema):
    with db_schema.begin() as conn:
        cfg = alembic_config(conn)
        try:
            command.downgrade(cfg, "0001")
            assert "opposed_at" not in {c["name"] for c in inspect(conn).get_columns("company_contacts")}
        finally:
            command.upgrade(cfg, "head")
        assert "opposed_at" in {c["name"] for c in inspect(conn).get_columns("company_contacts")}
