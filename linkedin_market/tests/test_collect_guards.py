"""Collecte et connexion LinkedIn sur doublures de pilote : G-3 (``min()``), G-7, G-8, G-10 (vague 2)."""

import json
import re
import time

import pytest
from fastapi.testclient import TestClient

from linkedin_market import activity, config, pipeline, scraper, session, storage
from linkedin_market.models import SearchRequest
from linkedin_market.web.app import create_app
from linkedin_market.web.tasks import FAILED, TaskManager

pytestmark = pytest.mark.db

CHECKPOINT = "https://www.linkedin.com/checkpoint/challenge/AgF"
VERIFICATION = (
    "LinkedIn demande une vérification que tu n'as pas terminée à temps : les actions navigateur sont suspendues 6 h. "
    "Termine-la sur LinkedIn puis lève le blocage depuis le tableau de bord."
)


class FakeLinkedIn:
    """Pilote factice : une page de recherche d'offres puis une page par offre ; ``on_get`` simule l'utilisateur
    (annulation) ou LinkedIn (checkpoint) au chargement d'une page."""

    def __init__(self, job_ids: list[int], on_get=None):
        self.job_ids, self.on_get = job_ids, on_get
        self.current_url, self.title = "about:blank", ""
        self.gets: list[str] = []

    def get(self, url: str) -> None:
        self.gets.append(url)
        self.current_url = url
        match = re.search(r"/jobs/view/(\d+)/", url)
        self.title = f"Data Engineer {match.group(1)} | Acme | LinkedIn" if match else ""
        if self.on_get:
            self.on_get(self, url)

    @property
    def detail_gets(self) -> int:
        return sum("/jobs/view/" in url for url in self.gets)

    @property
    def page_source(self) -> str:
        if "/jobs/search/" in self.current_url and "start=0" in self.current_url:
            return " ".join(f'<a href="/jobs/view/{job_id}/">' for job_id in self.job_ids)
        return ""

    def execute_script(self, script, *args):
        if script is scraper._JS_TEXT:
            return "Python Spark Airflow pour notre plateforme de données."
        if script is scraper._JS_POSTS:
            return []
        if script is scraper._JS_PEOPLE:
            return {}
        return None

    def quit(self) -> None:
        pass


@pytest.fixture
def fast(monkeypatch, tmp_path):
    """Pauses nulles, connexion factice, profil disponible, identifiants factices."""
    monkeypatch.setattr(
        config,
        "CAUTION_LEVELS",
        {name: config.Caution(name, level.max_items, 0.0, 0.0) for name, level in config.CAUTION_LEVELS.items()},
    )
    monkeypatch.setattr(session, "login", lambda *a, **k: None)
    root = tmp_path / "project"
    root.mkdir()
    (root / "profil.json").write_text(
        json.dumps({"title": "Data Engineer", "skills": ["python", "spark"], "text": "Data Engineer Python"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "PROJECT_ROOT", root)
    monkeypatch.setenv("LINKEDIN_EMAIL", "me@example.com")
    monkeypatch.setenv("LINKEDIN_PASSWORD", "x")


@pytest.fixture
def manager():
    tasks = TaskManager(persist=False)
    yield tasks
    tasks.shutdown(wait=True)


@pytest.fixture
def client(manager):
    with TestClient(create_app(tasks=manager, token=None), base_url="http://127.0.0.1:8765") as c:
        yield c


def use_driver(monkeypatch, driver: FakeLinkedIn) -> None:
    monkeypatch.setattr("linkedin_market.browser.create_driver", lambda: driver)


def wait_final(client, task_id: str, timeout: float = 30) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        task = client.get(f"/api/tasks/{task_id}").json()
        if task["status"] in ("terminée", "échec", "annulée"):
            return task
        time.sleep(0.05)
    raise AssertionError(task)


RUN = {"keyword": "Data Engineer", "caution": "prudent", "interpreter": "local"}
IDS = [4000001, 4000002, 4000003, 4000004, 4000005]


# ----------------------------------------------------------------------- G-10
def test_run_blocked_pendant_la_collecte(client, fast, monkeypatch, accept_risks):
    accept_risks()

    def checkpoint_at_third_job(driver, url):
        if driver.detail_gets == 3:
            driver.current_url = CHECKPOINT

    driver = FakeLinkedIn(IDS, checkpoint_at_third_job)
    use_driver(monkeypatch, driver)
    task = wait_final(client, client.post("/api/runs", json=RUN).json()["id"])
    assert task["status"] == "terminée", task
    result = task["result"]
    assert set(result) == {"report", "jobs", "counts", "blocked", "blocked_where", "blocked_sources", "sources"}
    assert result["blocked_sources"] == ["linkedin"]
    assert result["sources"] == [
        {"code": "linkedin", "items": 2, "status": "bloquée", "detail": result["sources"][0]["detail"]}
    ]
    assert "repos de 6 h" in result["sources"][0]["detail"]
    assert re.fullmatch(r"data-engineer-\d{4}-\d{2}-\d{2}", result["report"])
    assert result["jobs"] == 2 and result["counts"] == {"job": 2, "post": 0, "person": 0}
    assert result["blocked"] is True and result["blocked_where"] == "collecte"
    # I-06 : arrêt immédiat (aucune page après le checkpoint), événement blocked, repos de 6 h, aucune relance.
    assert driver.gets[-1].endswith(f"/jobs/view/{IDS[2]}/") and driver.detail_gets == 3
    assert activity.block_status()["where"] == "collecte"
    again = client.post("/api/runs", json=RUN)
    assert again.status_code == 409 and "interrompu" in again.json()["detail"]
    assert client.get(f"/api/reports/{result['report']}").status_code == 200  # rapport partiel enregistré
    assert any("Collecte interrompue par LinkedIn" in line for line in task["logs"])


def test_run_blocked_false_sans_blocage(client, fast, monkeypatch, accept_risks):
    accept_risks()
    use_driver(monkeypatch, FakeLinkedIn(IDS))
    task = wait_final(client, client.post("/api/runs", json=RUN).json()["id"])
    assert task["status"] == "terminée", task
    assert set(task["result"]) == {"report", "jobs", "counts", "blocked", "blocked_sources", "sources"}
    assert "blocked_where" not in task["result"]
    assert task["result"]["blocked_sources"] == []
    assert task["result"]["sources"] == [{"code": "linkedin", "items": 5, "status": "ok", "detail": ""}]
    assert task["result"]["blocked"] is False and task["result"]["jobs"] == 5
    assert activity.block_status() is None


def test_run_blocked_false_sans_collecte(client, fast, monkeypatch, accept_risks):
    accept_risks()
    use_driver(monkeypatch, FakeLinkedIn(IDS))
    first = wait_final(client, client.post("/api/runs", json=RUN).json()["id"])
    assert first["status"] == "terminée"
    task = wait_final(client, client.post("/api/runs", json={**RUN, "scrape": False}).json()["id"])
    assert task["result"]["blocked"] is False and "blocked_where" not in task["result"]
    assert task["result"]["sources"] == [] and task["result"]["blocked_sources"] == []  # aucune collecte


# ------------------------------------------------------------------------ G-8
def test_annulation_au_milieu_d_une_page(client, manager, fast, monkeypatch, accept_risks):
    accept_risks()

    def cancel_after_second_job(driver, url):
        if driver.detail_gets == 2:
            (run,) = [t for t in manager.active() if t.kind == "run"]
            assert manager.cancel(run.id)

    driver = FakeLinkedIn(IDS, cancel_after_second_job)
    use_driver(monkeypatch, driver)
    task = wait_final(client, client.post("/api/runs", json=RUN).json()["id"])
    assert task["status"] == "annulée" and task["error"] == "Tâche annulée."
    assert driver.detail_gets == 2  # plus aucune page après l'annulation
    kept = [item.id for item in storage.load_items("Data Engineer")]
    assert kept == [f"job:{IDS[0]}", f"job:{IDS[1]}"]  # éléments déjà collectés conservés
    assert activity.block_status() is None


# ------------------------------------------------------------------------ G-3
def test_scrape_applique_le_plafond_de_la_prudence(fast, monkeypatch, accept_risks):
    accept_risks()
    driver = FakeLinkedIn(list(range(4000001, 4000021)))  # 20 offres affichées
    use_driver(monkeypatch, driver)
    outcome = pipeline.scrape(SearchRequest(keyword="python", caution="prudent", max_items=200), False)
    assert outcome.blocked is False
    assert driver.detail_gets == 15 and len(storage.load_items("python")) == 15


# ------------------------------------------------------------------------ G-7
class FakeElement:
    def is_displayed(self) -> bool:
        return True

    def clear(self) -> None:
        pass

    def send_keys(self, *keys) -> None:
        pass


class FakeLoginDriver:
    """Page de connexion qui reste sur un checkpoint (vérification jamais terminée) ; cookies factices."""

    def __init__(self, landing: str = CHECKPOINT):
        self.landing = landing
        self.current_url = "about:blank"
        self.gets: list[str] = []

    def get(self, url: str) -> None:
        self.gets.append(url)
        self.current_url = self.landing if url in (session.FEED_URL, "https://www.linkedin.com/feed/") else url

    def find_elements(self, by, selector):
        return [FakeElement()]

    def add_cookie(self, cookie) -> None:
        pass

    def get_cookies(self) -> list[dict]:
        return [{"name": "li_at", "value": "secret"}]


@pytest.fixture
def short_waits(monkeypatch):
    monkeypatch.setattr(session, "FEED_TIMEOUT", 0.1)
    monkeypatch.setattr(session, "MANUAL_CHECK_TIMEOUT", 0.1)
    monkeypatch.setattr(session, "RESTORE_PAUSE", 0)
    monkeypatch.setenv("LINKEDIN_EMAIL", "me@example.com")
    monkeypatch.setenv("LINKEDIN_PASSWORD", "x")


def test_g7_verification_non_terminee_a_la_connexion(short_waits):
    driver = FakeLoginDriver()  # pas de session enregistrée : connexion par identifiants, /feed jamais atteint
    manager = TaskManager(persist=False)
    try:
        notes: list[str] = []

        def login_task(ctx):
            session.login(driver, interactive=False, on_manual_check=lambda m: notes.append(m) or ctx.manual_check(m))

        task = manager.submit("import_profile", "Importer le profil LinkedIn", login_task, browser=True)
        end = time.time() + 10
        while task.status not in ("terminée", "échec", "annulée") and time.time() < end:
            time.sleep(0.02)
    finally:
        manager.shutdown(wait=True)
    assert notes and "vérification" in notes[0]  # attente manuelle proposée à l'utilisateur d'abord
    assert task.status == FAILED and task.error == VERIFICATION
    blocked = activity.block_status()
    assert blocked is not None and blocked["where"] == "connexion"
    assert session.session_age_hours() is None  # aucune session enregistrée


def test_g7_checkpoint_pendant_la_reprise_par_cookies(short_waits, monkeypatch):
    session._save_session(FakeLoginDriver())  # session valide de moins de 72 h
    attempts: list[bool] = []
    monkeypatch.setattr(session, "_login_with_password", lambda *a, **k: attempts.append(True))
    driver = FakeLoginDriver(landing=CHECKPOINT)
    with pytest.raises(session.VerificationNotDoneError) as raised:
        session.login(driver, interactive=False)
    assert str(raised.value) == VERIFICATION
    assert attempts == []  # aucune nouvelle tentative par identifiants (I-06)
    assert activity.block_status()["where"] == "connexion"


def test_g7_reprise_par_cookies_normale(short_waits):
    session._save_session(FakeLoginDriver())
    session.login(FakeLoginDriver(landing="https://www.linkedin.com/feed/"), interactive=False)
    assert activity.block_status() is None
