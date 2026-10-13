"""Collecte multi-sources (vague 5) : ``POST /api/runs`` avec France Travail, Malt et WTTJ, ``GET /api/jobs``,
``/apply`` et ``/contacts/lookup`` sur offre non-LinkedIn. Pilote et API factices, base de test."""

import json
import time
from datetime import UTC, datetime

import httpx
import pytest
from fastapi.testclient import TestClient

from linkedin_market import activity, application, config, pipeline, sources, storage
from linkedin_market.models import Item, SearchRequest
from linkedin_market.sources import base
from linkedin_market.tests.test_source_browser import FakeDriver, html, malt_driver, wttj_driver
from linkedin_market.tests.test_source_france_travail import FakeApi, FakeClock, ok
from linkedin_market.web import services
from linkedin_market.web.app import create_app
from linkedin_market.web.tasks import TaskManager

pytestmark = pytest.mark.db

VERSION = "2026-10"
RUN = {"keyword": "Data Engineer", "caution": "prudent", "interpreter": "local"}
NO_FT = (
    "Identifiants France Travail absents : renseigne FRANCE_TRAVAIL_CLIENT_ID et "
    "FRANCE_TRAVAIL_CLIENT_SECRET dans le fichier .env."
)
TODAY = datetime.now(UTC).date().isoformat()


def ft_offer(offer_id: str = "194ZXYZ") -> dict:
    return {
        "id": offer_id,
        "intitule": "Data Engineer",
        "description": "Pipelines Python, Spark et Airflow en télétravail.",
        "dateCreation": f"{TODAY}T08:12:45.000Z",
        "lieuTravail": {"libelle": "75 - Paris"},
        "entreprise": {"nom": "Acme"},
        "origineOffre": {"urlOrigine": f"https://candidat.francetravail.fr/offres/recherche/detail/{offer_id}"},
    }


@pytest.fixture(autouse=True)
def fast(monkeypatch, tmp_path):
    """Pauses nulles, profil disponible, aucun identifiant réel ; les identifiants sont posés par chaque test."""
    monkeypatch.setattr(base, "human_pause", lambda *a, **k: None)
    root = tmp_path / "project"
    root.mkdir()
    (root / "profil.json").write_text(
        json.dumps({"title": "Data Engineer", "skills": ["python", "spark"], "text": "Data Engineer Python"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "PROJECT_ROOT", root)


@pytest.fixture
def manager():
    tasks = TaskManager(persist=False)
    yield tasks
    for task in tasks.active():
        tasks.cancel(task.id)
    tasks.shutdown(wait=True)


@pytest.fixture
def client(manager):
    with TestClient(create_app(tasks=manager, token=None), base_url="http://127.0.0.1:8765") as c:
        yield c


def wait_final(client, task_id: str, timeout: float = 30) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        task = client.get(f"/api/tasks/{task_id}").json()
        if task["status"] in ("terminée", "échec", "annulée"):
            return task
        time.sleep(0.05)
    raise AssertionError(task)


def with_france_travail(monkeypatch, searches: list) -> FakeApi:
    monkeypatch.setenv("FRANCE_TRAVAIL_CLIENT_ID", "id")
    monkeypatch.setenv("FRANCE_TRAVAIL_CLIENT_SECRET", "secret")
    clock = FakeClock()
    api = FakeApi(clock, searches)
    monkeypatch.setattr(
        sources,
        "france_travail_source",
        lambda: sources.FranceTravailSource(
            "id", "secret", transport=httpx.MockTransport(api), sleep=clock.sleep, clock=clock.now
        ),
    )
    return api


def with_drivers(monkeypatch, *drivers: FakeDriver) -> None:
    queue = iter(drivers)
    monkeypatch.setattr("linkedin_market.browser.create_driver", lambda: next(queue))


def accept(client, code: str) -> None:
    assert (
        client.post(f"/api/sources/{code}/consent", json={"risk_version": VERSION, "accepted": True}).status_code == 201
    )


def post_run(client, **body):
    return client.post("/api/runs", json={**RUN, **body})


# ------------------------------------------------------- France Travail seule
def test_france_travail_seule_sans_identifiants_linkedin_ni_consentement(client, manager, monkeypatch):
    with_france_travail(monkeypatch, [ok([ft_offer("A1"), ft_offer("A2")])])
    response = post_run(client, sources=["france_travail"])
    assert response.status_code == 202
    assert response.json()["browser"] is False  # aucun navigateur : pas de verrou
    task = wait_final(client, response.json()["id"])
    assert task["status"] == "terminée", task
    result = task["result"]
    assert result["blocked"] is False and result["blocked_sources"] == [] and "blocked_where" not in result
    assert result["sources"] == [{"code": "france_travail", "items": 2, "status": "ok", "detail": ""}]
    assert result["jobs"] == 2 and result["counts"]["job"] == 2
    item = storage.find_item("job:ft-A1")
    assert item is not None and item.source_code == "france_travail"
    assert activity.block_status() is None


def test_scrape_false_ignore_les_sources(client):
    response = post_run(client, scrape=False, sources=["malt", "france_travail"])
    assert response.status_code == 202 and response.json()["browser"] is False
    task = wait_final(client, response.json()["id"])
    assert task["result"]["sources"] == [] and task["result"]["blocked_sources"] == []


# ------------------------------------------------------- ordre des contrôles
def test_ordre_des_controles_404_422_428_409_identifiants_409_blocage(client, monkeypatch):
    monkeypatch.setenv("LINKEDIN_EMAIL", "me@example.com")
    monkeypatch.setenv("LINKEDIN_PASSWORD", "x")
    # 422 : mot-clé / source invalides avant tout le reste.
    assert post_run(client, keyword="  ", sources=["malt"]).status_code == 422
    assert post_run(client, sources=["malt", "indeed"]).status_code == 422
    # 428 : première source navigateur sans consentement, dans l'ordre de la liste, avant les identifiants.
    response = post_run(client, sources=["france_travail", "wttj", "malt"])
    assert response.status_code == 428 and response.json()["source"] == "wttj"
    accept(client, "wttj")
    response = post_run(client, sources=["france_travail", "wttj", "malt"])
    assert response.status_code == 428 and response.json()["source"] == "malt"
    accept(client, "malt")
    # 409 identifiants : France Travail absent (LinkedIn n'est pas demandé, ses identifiants ne comptent pas).
    response = post_run(client, sources=["france_travail", "wttj", "malt"])
    assert response.status_code == 409 and response.json()["detail"] == NO_FT
    # 409 blocage : premier code navigateur de la liste en repos ; chaque source a son repos.
    monkeypatch.setenv("FRANCE_TRAVAIL_CLIENT_ID", "id")
    monkeypatch.setenv("FRANCE_TRAVAIL_CLIENT_SECRET", "secret")
    activity.record("blocked", where="collecte", source="malt")
    activity.record("blocked", where="collecte", source="wttj")
    response = post_run(client, sources=["malt", "wttj"])
    assert response.status_code == 409 and response.json()["detail"].startswith("Malt a interrompu")
    response = post_run(client, sources=["wttj", "malt"])
    assert response.status_code == 409 and response.json()["detail"].startswith("Welcome to the Jungle a interrompu")
    client.post("/api/block/clear", json={"source": "wttj"})
    assert post_run(client, sources=["wttj", "malt"]).json()["detail"].startswith("Malt a interrompu")


def test_identifiants_linkedin_exiges_seulement_si_linkedin_est_demandee(client, monkeypatch):
    accept(client, "linkedin")
    monkeypatch.delenv("LINKEDIN_EMAIL", raising=False)
    response = post_run(client, sources=["linkedin"])
    assert response.status_code == 409 and response.json()["detail"].startswith("Identifiants LinkedIn absents")
    response = post_run(client, sources=["linkedin", "france_travail"])
    assert response.status_code == 409 and response.json()["detail"].startswith("Identifiants LinkedIn absents")


# --------------------------------------------------------------- multi-sources
def test_multi_sources_une_source_bloquee_n_arrete_pas_les_autres(client, monkeypatch):
    accept(client, "malt")
    accept(client, "wttj")
    with_france_travail(monkeypatch, [ok([ft_offer()])])
    challenge = FakeDriver({"": html("blocked_challenge.html")})
    wttj = wttj_driver()
    with_drivers(monkeypatch, challenge, wttj)  # un navigateur par source navigateur, dans l'ordre de la liste
    response = post_run(client, sources=["france_travail", "malt", "wttj"])
    assert response.status_code == 202 and response.json()["browser"] is True
    task = wait_final(client, response.json()["id"])
    assert task["status"] == "terminée", task
    result = task["result"]
    assert result["blocked"] is True and result["blocked_where"] == "collecte"
    assert result["blocked_sources"] == ["malt"]
    statuses = {s["code"]: (s["items"], s["status"]) for s in result["sources"]}
    assert [s["code"] for s in result["sources"]] == ["france_travail", "malt", "wttj"]  # ordre de la demande
    assert statuses == {"france_travail": (1, "ok"), "malt": (0, "bloquée"), "wttj": (3, "ok")}
    assert next(s for s in result["sources"] if s["code"] == "malt")["detail"].startswith("Malt a demandé")
    assert result["jobs"] == 4 and result["counts"]["job"] == 4
    assert challenge.quit_called == 1 and wttj.quit_called == 1  # un driver par source, fermé en finally
    assert len(challenge.gets) == 1  # aucune page chargée après le défi (I-06)
    # Repos propre à Malt : événement blocked avec data.source ; WTTJ et LinkedIn ne sont pas bloquées.
    assert activity.block_status(source="malt")["where"] == "collecte"
    assert activity.block_status(source="wttj") is None and activity.block_status() is None
    assert any("Collecte interrompue par Malt" in line for line in task["logs"])
    assert post_run(client, sources=["malt"]).status_code == 409  # repos de 6 h, aucune relance automatique
    # GET /api/jobs : source_code par offre et filtre exact.
    listing = client.get("/api/jobs").json()
    assert sorted(j["source_code"] for j in listing["jobs"]) == ["france_travail", "wttj", "wttj", "wttj"]
    only = client.get("/api/jobs", params={"source": "france_travail"}).json()["jobs"]
    assert [j["id"] for j in only] == ["job:ft-194ZXYZ"]
    assert client.get("/api/jobs", params={"source": "inconnue"}).json()["jobs"] == []
    report = client.get(f"/api/reports/{result['report']}").json()
    assert {j["source_code"] for j in report["top_jobs"]} == {"france_travail", "wttj"}
    detail = client.get("/api/jobs/job:ft-194ZXYZ").json()
    assert detail["source_code"] == "france_travail"


def test_france_travail_en_echec_est_ignoree_les_autres_continuent(client, monkeypatch):
    accept(client, "wttj")
    with_france_travail(monkeypatch, [httpx.Response(503)] * 3)
    with_drivers(monkeypatch, wttj_driver())
    task = wait_final(client, post_run(client, sources=["france_travail", "wttj"]).json()["id"])
    assert task["status"] == "terminée", task
    ft, wttj = task["result"]["sources"]
    assert ft["code"] == "france_travail" and ft["status"] == "indisponible" and "3 essais" in ft["detail"]
    assert wttj["status"] == "ok" and wttj["items"] == 3
    assert task["result"]["blocked"] is False  # une source indisponible n'est pas un blocage
    assert any("France Travail ignorée" in line for line in task["logs"])


def test_france_travail_sans_identifiants_au_demarrage_de_la_tache(manager, monkeypatch):
    task = manager.submit(
        "run",
        "t",
        lambda ctx: services.run_analysis(
            ctx,
            SearchRequest(keyword="Data Engineer"),
            scrape=True,
            interpreter_name="local",
            sources=["france_travail"],
        ),
    )
    end = time.time() + 30
    while task.status not in ("terminée", "échec", "annulée") and time.time() < end:
        time.sleep(0.05)
    assert task.status == "terminée", task.error
    assert task.result["sources"] == [{"code": "france_travail", "items": 0, "status": "indisponible", "detail": NO_FT}]


def test_consentement_revoque_entre_la_creation_et_le_demarrage_de_la_tache(manager):
    task = manager.submit(
        "run",
        "t",
        lambda ctx: services.run_analysis(
            ctx, SearchRequest(keyword="Data Engineer"), scrape=True, interpreter_name="local", sources=["wttj"]
        ),
        browser=True,
    )
    end = time.time() + 30
    while task.status not in ("terminée", "échec", "annulée") and time.time() < end:
        time.sleep(0.05)
    assert task.status == "échec"
    assert task.error == (
        f"Accepte d'abord les risques liés à Welcome to the Jungle (version {VERSION}) avant toute action "
        "dans le navigateur."
    )


def test_repos_revérifie_au_demarrage_de_la_tache(manager, accept_risks):
    accept_risks("malt")
    activity.record("blocked", where="collecte", source="malt")
    with pytest.raises(activity.BlockedByLinkedIn, match="Malt a interrompu"):
        pipeline.collect(SearchRequest(keyword="python"), ["malt"], False)


def test_annulation_pendant_malt_enregistre_l_existant_puis_relance(accept_risks, monkeypatch):
    accept_risks("malt")
    driver = malt_driver()
    with_drivers(monkeypatch, driver)

    def stop() -> bool:
        return sum("/mission/" in u for u in driver.gets) >= 1  # après la première mission lue

    with pytest.raises(pipeline.Cancelled):
        pipeline.collect(SearchRequest(keyword="Data Engineer"), ["malt"], False, should_stop=stop)
    assert driver.quit_called == 1
    assert [i.id for i in storage.load_items("Data Engineer")] == ["job:malt-abc123"]
    assert activity.block_status(source="malt") is None


def test_collect_dispatch_dans_l_ordre_et_plafond(accept_risks, monkeypatch):
    accept_risks("malt")
    accept_risks("wttj")
    with_drivers(monkeypatch, wttj_driver(), malt_driver())
    outcome = pipeline.collect(SearchRequest(keyword="Data Engineer", max_items=500), ["wttj", "malt"], True)
    assert [(r.code, r.items, r.status) for r in outcome.sources] == [("wttj", 3, "ok"), ("malt", 2, "ok")]
    assert outcome.blocked is False and outcome.profile_text == ""
    assert {i.source_code for i in storage.load_items("Data Engineer")} == {"wttj", "malt"}


def test_on_ready_des_le_demarrage(monkeypatch):
    with_france_travail(monkeypatch, [ok([ft_offer()])])
    calls: list[int] = []
    pipeline.collect(
        SearchRequest(keyword="Data Engineer"), ["france_travail"], False, on_ready=lambda: calls.append(1)
    )
    assert calls == [1]


# --------------------------------------------------- apply / contacts / prepare
def prepare_entry(workdir, monkeypatch, job_id: str, source_code: str, *, in_history: bool = True, status="préparée"):
    apps_dir = workdir / "applications"
    folder = apps_dir / f"20261007-{job_id.replace(':', '-')}"
    folder.mkdir(parents=True)
    (folder / "cv.md").write_text("# Dev\n", encoding="utf-8")
    (folder / "message.txt").write_text("Bonjour", encoding="utf-8")
    monkeypatch.setattr(config, "APPLICATIONS_DIR", apps_dir)
    if in_history:
        storage.append_items(
            [
                Item(
                    kind="job",
                    id=job_id,
                    url="https://exemple.org/offre",
                    title="Dev",
                    text="Python",
                    keyword="python",
                    source_code=source_code,
                )
            ]
        )
    application.save_applications(
        {
            job_id: {
                "title": "Dev",
                "company": "Acme",
                "url": "https://exemple.org/offre",
                "score": 90,
                "folder": str(folder),
                "generator": "local",
                "status": status,
                "prepared_at": "2026-10-07T10:00:00",
            }
        }
    )


@pytest.mark.parametrize(
    ("job_id", "source_code", "label", "in_history"),
    [
        ("job:ft-77", "france_travail", "France Travail", True),
        ("job:malt-x1", "malt", "Malt", True),
        ("job:wttj-x1", "wttj", "Welcome to the Jungle", True),
        ("job:indeed-0123456789abcdef", "indeed", "Indeed", True),
        (
            "job:malt-purge",
            "malt",
            "Malt",
            False,
        ),  # offre créée a minima (source linkedin par défaut) : le préfixe fait foi
    ],
)
def test_apply_et_lookup_refuses_sur_offre_non_linkedin(
    client, monkeypatch, workdir, job_id, source_code, label, in_history
):
    prepare_entry(workdir, monkeypatch, job_id, source_code, in_history=in_history, status="postulée")
    apply = client.post(f"/api/applications/{job_id}/apply")
    # 422 avant le 409 d'état (déjà postulée) et avant le 428 (aucun consentement, aucun identifiant).
    assert apply.status_code == 422
    assert apply.json()["detail"] == (
        f"La candidature simplifiée n'existe que pour LinkedIn : ouvre l'offre sur {label} et postule toi-même."
    )
    lookup = client.post(f"/api/applications/{job_id}/contacts/lookup")
    assert lookup.status_code == 422
    assert lookup.json()["detail"] == "La recherche de recruteurs n'existe que pour les offres LinkedIn."
    assert client.post("/api/applications/job:absente/apply").status_code == 404  # 404 avant 422


def test_apply_et_lookup_revérifient_au_demarrage_de_la_tache(manager, monkeypatch, workdir):
    prepare_entry(workdir, monkeypatch, "job:ft-77", "france_travail")
    for name, function in (("apply", services.apply_job), ("lookup", services.lookup_contacts)):
        task = manager.submit(name, name, lambda ctx, f=function: f(ctx, "job:ft-77"), browser=True)
        end = time.time() + 30
        while task.status not in ("terminée", "échec", "annulée") and time.time() < end:
            time.sleep(0.05)
        assert task.status == "échec"
        assert "n'existe que pour" in task.error


def test_apply_sur_offre_linkedin_inchange(client, monkeypatch, workdir):
    prepare_entry(workdir, monkeypatch, "job:99", "linkedin")
    response = client.post("/api/applications/job:99/apply")
    assert response.status_code == 428  # LinkedIn : le consentement reste exigé
    assert response.json()["source"] == "linkedin"


def test_message_email_permis_pour_toute_source(client, monkeypatch, workdir):
    prepare_entry(workdir, monkeypatch, "job:ft-77", "france_travail")
    body = {"channel": "email", "to": "rh@acme.com", "subject": "Candidature", "body": "Bonjour", "confirm": True}
    response = client.post("/api/applications/job:ft-77/messages", json=body)
    assert response.status_code == 503 and "SMTP" in response.json()["detail"]  # pas de refus lié à la source


def test_ancien_rapport_sans_source_code_est_lu_comme_linkedin():
    rows = services.jobs_overview({"top_jobs": [{"id": "job:123", "title": "T", "company": "C", "score": 50.0}]})
    assert rows[0]["source_code"] == "linkedin"
    rows = services.jobs_overview(
        {"top_jobs": [{"id": "job:ft-1", "title": "T", "company": "C", "score": 50.0, "source_code": "france_travail"}]}
    )
    assert rows[0]["source_code"] == "france_travail"
