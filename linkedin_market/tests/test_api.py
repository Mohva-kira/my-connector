import copy
import json
import re

import pytest
from fastapi.testclient import TestClient

from linkedin_market import application, config, report, storage
from linkedin_market.models import Item
from linkedin_market.web.app import create_app
from linkedin_market.web.tasks import TaskManager

pytestmark = pytest.mark.db  # réglages, historique et rapports en PostgreSQL (vague 1)

JOB = {
    "id": "job:111",
    "title": "Senior Python Developer",
    "company": "Acme",
    "url": "https://www.linkedin.com/jobs/view/111/",
    "location": "Remote",
    "posted_at": "2026-10-01",
    "score": 82.0,
    "criteria": {},
    "job_skills": ["python", "aws"],
    "missing_skills": ["aws"],
    "matched_skills": ["python"],
    "summary": ["Build APIs."],
    "years_required": 5,
    "seniority_required": "senior",
    "remote": True,
    "freelance": False,
}
REPORT = {
    "generated_at": "2026-10-07T10:00:00",
    "request": {
        "keyword": "python",
        "location": "",
        "remote_only": True,
        "caution": "normal",
        "max_items": 30,
        "profile_source": "auto",
        "extra_skills": [],
        "years_experience": None,
        "seniority": "",
    },
    "profile": {"origin": "x", "title": "Dev", "skills": ["python"], "years_experience": 10, "seniority": "senior"},
    "counts": {"job": 1, "post": 0, "person": 0},
    "top_jobs": [JOB],
    "min_labels": 6,
    "model": {
        "trained": False,
        "n_labels": 0,
        "train_accuracy": None,
        "importance": [],
        "loss_history": [],
        "trained_at": None,
    },
    "gap": {"skills": [], "missing": [{"skill": "aws", "weight": 0.5, "owned": False}], "coverage": 50.0},
    "rising": {"skills": [{"term": "docker", "growth": 1.5}]},
    "skill_demand": [],
    "top_terms": [],
    "weekly": {},
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    (root / "profil.json").write_text(
        json.dumps({"title": "Dev", "skills": ["python"], "text": "Dev Python"}), encoding="utf-8"
    )
    monkeypatch.setattr(config, "PROJECT_ROOT", root)
    # Sans identifiants, aucune action ne lance de navigateur (ajout de compétence compris).
    monkeypatch.delenv("LINKEDIN_EMAIL", raising=False)
    monkeypatch.delenv("LINKEDIN_PASSWORD", raising=False)
    manager = TaskManager(persist=False)
    with TestClient(
        create_app(tasks=manager, token="secret"), base_url="http://127.0.0.1:8765", headers={"X-Local-Token": "secret"}
    ) as c:
        yield c
    manager.shutdown()


def write_report() -> str:
    """Rapport « python » en base (et son HTML), avec l'offre job:111 dans l'historique ; renvoie son nom."""
    storage.append_items(
        [
            Item(
                kind="job",
                id="job:111",
                url=JOB["url"],
                title=JOB["title"],
                text="Build APIs in Python.",
                keyword="python",
                company="Acme",
                posted_at="2026-10-01",
            )
        ]
    )
    return report.write(copy.deepcopy(REPORT))[1].name


def test_guard_rejects_bad_host_and_token(tmp_path):
    manager = TaskManager(persist=False)
    app = create_app(tasks=manager, token="secret")
    ok = TestClient(app, base_url="http://127.0.0.1:8765")
    assert ok.get("/api/status").status_code == 401
    assert ok.get("/api/health").status_code == 200
    assert ok.get("/api/status", headers={"X-Local-Token": "secret"}).status_code == 200
    assert ok.get("/api/status?token=secret").status_code == 200
    evil = TestClient(app, base_url="http://evil.example.com")
    assert evil.get("/api/health").status_code == 403
    cross = ok.put("/api/settings", json={}, headers={"X-Local-Token": "secret", "Origin": "https://evil.example.com"})
    assert cross.status_code == 403
    manager.shutdown()


def test_settings_roundtrip_and_validation(client):
    assert (
        client.put(
            "/api/settings", json={"keyword": "Data Engineer", "caution": "prudent", "apply_daily_max": 5}
        ).json()["keyword"]
        == "Data Engineer"
    )
    assert client.get("/api/settings").json()["apply_daily_max"] == 5
    assert client.put("/api/settings", json={"caution": "fou"}).status_code == 422
    assert client.put("/api/settings", json={"apply_daily_max": -1}).status_code == 422


def test_skills_add_remove_and_suggestions(client):
    write_report()
    profile = client.get("/api/profile").json()
    assert "python" in profile["skills"]
    after = client.post("/api/profile/skills", json={"skill": "Rust"}).json()
    assert "rust" in after["skills"] and "rust" not in after["base_skills"]
    after = client.delete("/api/profile/skills/python").json()
    assert "python" not in after["skills"] and "python" in after["base_skills"]
    assert [s["skill"] for s in after["suggestions"]["missing"]] == ["aws"]
    assert [s["skill"] for s in after["suggestions"]["rising"]] == ["docker"]
    reset = client.post("/api/profile/reset-overrides").json()
    assert "python" in reset["skills"] and "rust" not in reset["skills"]
    assert client.put("/api/profile", json={"seniority": "dieu"}).status_code == 422


def test_jobs_labels_and_filters(client):
    assert client.get("/api/jobs").json()["jobs"] == []
    write_report()
    jobs = client.get("/api/jobs").json()["jobs"]
    assert jobs[0]["id"] == "job:111" and jobs[0]["label"] is None
    assert client.put("/api/jobs/job:111/label", json={"value": 1}).json()["n_labels"] == 1
    assert client.get("/api/jobs?label=pertinente").json()["jobs"][0]["label"] == 1
    assert client.get("/api/jobs?label=non_annotee").json()["jobs"] == []
    assert client.get("/api/jobs?min_score=90").json()["jobs"] == []
    assert client.get("/api/jobs?q=acme").json()["jobs"]
    assert client.put("/api/jobs/job:111/label", json={"value": 3}).status_code == 422
    assert client.get("/api/jobs/job:999").status_code == 404
    assert client.put("/api/jobs/job:111/label", json={"value": None}).json()["n_labels"] == 0
    assert storage.load_labels() == {}


def test_label_d_une_offre_absente_de_l_historique(client):
    """L'annotation d'une offre purgée n'est plus perdue en silence : 404 actionnable, rien d'écrit."""
    write_report()
    for value in (1, 0, None):
        response = client.put("/api/jobs/job:4242/label", json={"value": value})
        assert response.status_code == 404
        assert response.json()["detail"] == "Cette offre n'est plus dans l'historique : annotation impossible."
    assert storage.load_labels() == {}


def test_legacy_application_roundtrip_and_status_rules(client, data_dir):
    folder = config.APPLICATIONS_DIR / "20261006-acme-dev"
    folder.mkdir(parents=True)
    (folder / "cv.md").write_text("# CV", encoding="utf-8")
    (folder / "message.txt").write_text("Bonjour", encoding="utf-8")
    application.save_applications(
        {
            "job:111": {
                "title": "Dev",
                "company": "Acme",
                "url": "https://www.linkedin.com/jobs/view/111/",
                "score": 80,
                "folder": str(folder),
                "generator": "local",
                "status": "postulée",
                "prepared_at": "2026-10-06T10:00:00",
            }
        }
    )
    detail = client.get("/api/applications/job:111").json()
    assert detail["contacts"] == [] and detail["history"] == [] and detail["message"] == "Bonjour"
    patched = client.patch("/api/applications/job:111", json={"message": "Salut", "cv": "# Nouveau"}).json()
    assert patched["message"] == "Salut" and "Nouveau" in (folder / "cv.html").read_text(encoding="utf-8")
    application.advance_status("job:111", "contactée")
    assert application.get_application("job:111")["status"] == "postulée"  # jamais rétrogradé
    application.advance_status("job:111", "entretien")
    assert application.get_application("job:111")["status"] == "entretien"
    assert client.patch("/api/applications/job:111", json={"status": "n'importe quoi"}).status_code == 422
    bad = client.patch("/api/applications/job:111", json={"contacts": [{"email": "pas-un-email"}]})
    assert bad.status_code == 422
    ok = client.patch("/api/applications/job:111", json={"contacts": [{"full_name": "Ana", "email": "ana@acme.com"}]})
    contact = ok.json()["contacts"][0]
    assert contact.pop("collected_at").endswith("Z")  # V4-8 : date de collecte (UTC) et état d'opposition
    assert contact == {
        "full_name": "Ana",
        "profile_url": "",
        "email": "ana@acme.com",
        "headline": "",
        "source": "manuel",
        "opposed": False,
    }
    # C-4 : « name » n'est plus lu (le nom est perdu, pas d'erreur).
    legacy = client.patch("/api/applications/job:111", json={"contacts": [{"name": "Ana", "email": "ana@acme.com"}]})
    assert legacy.json()["contacts"][0]["full_name"] == ""
    assert client.patch("/api/applications/job:111", json={"contacts": [{"source": "linkedin"}]}).status_code == 422
    assert client.get("/api/applications/job:404").status_code == 404


def test_application_folder_outside_data_dir_is_refused(client, tmp_path):
    application.save_applications({"job:1": {"title": "x", "url": "u", "folder": str(tmp_path), "status": "préparée"}})
    assert client.get("/api/applications/job:1").status_code == 500


def test_monitor_and_status_shape(client):
    write_report()
    monitor = client.get("/api/monitor").json()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", monitor["generated_at"])  # C-1
    assert monitor["funnel"][0] == {"step": "Offres analysées", "count": 1}
    assert monitor["quotas"]["apply"]["limit"] == 10 and len(monitor["activity"]) == 14
    status = client.get("/api/status").json()
    assert status["has_analysis"] and status["profile_file"] and "password" not in json.dumps(status).lower()


def test_run_requires_keyword(client):
    assert client.post("/api/runs", json={"scrape": False}).status_code == 422


def test_added_skill_is_synced_to_linkedin(client, monkeypatch, accept_risks):
    accept_risks()
    import time

    from linkedin_market.web import services

    unavailable = client.post("/api/profile/skills", json={"skill": "Go"}).json()
    assert (
        "go" in unavailable["skills"] and unavailable["linkedin_sync"]["status"] == "unavailable"
    )  # ajout local conservé

    calls = []
    monkeypatch.setenv("LINKEDIN_EMAIL", "me@example.com")
    monkeypatch.setenv("LINKEDIN_PASSWORD", "x")
    monkeypatch.setattr(
        services,
        "sync_skill_to_linkedin",
        lambda ctx, skill: calls.append(skill) or {"skill": skill, "linkedin_name": "Kubernetes", "state": "ajoutée"},
    )
    queued = client.post("/api/profile/skills", json={"skill": "Kubernetes"}).json()
    assert queued["linkedin_sync"]["status"] == "queued" and queued["linkedin_sync"]["task"]["kind"] == "profile_skill"
    task_id = queued["linkedin_sync"]["task"]["id"]
    for _ in range(50):
        task = client.get(f"/api/tasks/{task_id}").json()
        if task["status"] == "terminée":
            break
        time.sleep(0.05)
    assert task["status"] == "terminée" and calls == ["kubernetes"]

    assert client.post("/api/profile/skills", json={"skill": "python"}).json()["linkedin_sync"]["status"] == "skipped"
    client.put("/api/settings", json={"linkedin_skill_sync": False})
    assert client.post("/api/profile/skills", json={"skill": "Rust"}).json()["linkedin_sync"]["status"] == "off"
    assert calls == ["kubernetes"]
