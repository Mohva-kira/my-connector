"""Activité (plafonds, blocages) et session LinkedIn chiffrée en base (vague 1, sous-lot 1c)."""

import logging
import time
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select, update

from linkedin_market import activity, config
from linkedin_market import session as lm_session
from linkedin_market.db import models, session, users

pytestmark = pytest.mark.db


@pytest.fixture
def paris(monkeypatch):
    """Fuseau local de la machine : Europe/Paris (UTC+2 en octobre)."""
    monkeypatch.setenv("TZ", "Europe/Paris")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def _event(kind: str, at: datetime, **data) -> None:
    with session.transaction() as s:
        s.add(models.ActivityEvent(user_id=users.current_user_id(s), kind=kind, data=data, at=at))


def test_plafond_au_passage_de_minuit_local(paris):
    today = datetime.now().astimezone().date()
    midnight = datetime.combine(today, datetime.min.time()).astimezone()
    _event("apply", midnight - timedelta(minutes=1))  # hier, heure locale (mais « aujourd'hui » en UTC)
    _event("apply", midnight + timedelta(minutes=1))
    _event("message_email", midnight + timedelta(minutes=2))
    assert activity.count_today("apply") == 1
    assert activity.count_today("apply", today=today - timedelta(days=1)) == 1
    assert activity.count_today("message_email", "message_linkedin") == 1
    with pytest.raises(activity.QuotaExceeded, match="Plafond quotidien atteint"):
        activity.check_quota(("apply",), 1, "candidatures")
    days = {d["date"]: d for d in activity.per_day(3)}
    assert days[today.isoformat()].get("apply") == 1 and days[(today - timedelta(days=1)).isoformat()]["apply"] == 1


def test_record_et_load_en_utc():
    event = activity.record("run", keyword="python", jobs=3)
    assert event["kind"] == "run" and event["jobs"] == 3 and event["at"].endswith("Z")
    assert activity.load(1)[0]["keyword"] == "python"


def test_blocked_puis_unblocked():
    assert activity.block_status() is None
    activity.record("blocked", where="collecte")
    blocked = activity.block_status()
    assert blocked["where"] == "collecte" and blocked["at"].endswith("Z") and blocked["until"].endswith("Z")
    with pytest.raises(activity.BlockedByLinkedIn, match="interrompu la navigation"):
        activity.guard_not_blocked()
    activity.record("unblocked")
    assert activity.block_status() is None
    activity.guard_not_blocked()
    # Un blocage de plus de 6 h n'empêche plus rien.
    _event("blocked", datetime.now(UTC) - timedelta(hours=7), where="ancien")
    assert activity.block_status() is None


class FakeDriver:
    def __init__(self, cookies):
        self._cookies = cookies

    def get_cookies(self):
        return self._cookies


SECRET = "AQEDAR-secret-cookie-value"


def test_cookies_chiffres_absents_des_logs_et_72h(caplog):
    caplog.set_level(logging.DEBUG)
    lm_session._save_session(FakeDriver([{"name": "li_at", "value": SECRET}]))
    with session.transaction() as s:
        row = s.execute(select(models.PlatformSession)).scalar_one()
        assert row.source_code == "linkedin" and SECRET.encode() not in row.cookies_encrypted
        assert row.expires_at - row.last_login_at == timedelta(hours=config.SESSION_TTL_HOURS)
    assert lm_session._load_session() == [{"name": "li_at", "value": SECRET}]
    assert 0 <= lm_session.session_age_hours() < 0.1
    # 72 h plus tard : session supprimée, jamais prolongée.
    with session.transaction() as s:
        s.execute(
            update(models.PlatformSession).values(
                last_login_at=datetime.now(UTC) - timedelta(hours=73), expires_at=datetime.now(UTC) - timedelta(hours=1)
            )
        )
    assert lm_session._load_session() is None and lm_session.session_age_hours() is None
    assert SECRET not in caplog.text and "li_at" not in caplog.text


def test_session_illisible_avec_autre_cle(monkeypatch, caplog):
    from cryptography.fernet import Fernet

    lm_session._save_session(FakeDriver([{"name": "li_at", "value": SECRET}]))
    monkeypatch.setenv("LM_SECRET_KEY", Fernet.generate_key().decode())
    assert lm_session._load_session() is None
    assert "illisible" in caplog.text and SECRET not in caplog.text
    with session.transaction() as s:
        assert s.execute(select(models.PlatformSession)).first() is None


def test_aucun_fichier_de_session():
    lm_session._save_session(FakeDriver([{"name": "li_at", "value": SECRET}]))
    assert not config.SESSION_FILE.exists()
    assert lm_session._load_session() == [{"name": "li_at", "value": SECRET}]
