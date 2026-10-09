"""Rapports et modèle de matching en base (vague 1, sous-lot 1b)."""

import copy
import re
from datetime import datetime

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, update
from sqlalchemy.exc import DBAPIError

from linkedin_market import config, matching, report, storage
from linkedin_market.db import models, session
from linkedin_market.tests.test_api import REPORT
from linkedin_market.web.app import create_app
from linkedin_market.web.tasks import TaskManager

pytestmark = pytest.mark.db
ISO_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


@pytest.fixture
def client():
    manager = TaskManager(persist=False)
    with TestClient(create_app(tasks=manager, token=None), base_url="http://127.0.0.1:8765") as c:
        yield c
    manager.shutdown()


def write(keyword: str = "Data Engineer") -> str:
    results = copy.deepcopy(REPORT)
    results["request"]["keyword"] = keyword
    results["interpreter"] = "none"
    return report.write(results)[1].name


def test_slug_du_jour_puis_suffixe():
    today = datetime.now().astimezone().date().isoformat()
    assert write() == f"data-engineer-{today}"
    assert write() == f"data-engineer-{today}-2"
    assert write() == f"data-engineer-{today}-3"
    assert write("Python") == f"python-{today}"


def test_html_sur_disque_et_chemin_relatif(data_dir):
    name = write()
    with session.transaction() as s:
        row = s.execute(select(models.Report).where(models.Report.slug == name)).scalar_one()
        assert row.html_path == f"reports/{name}.html" and row.interpreter_used == "local"
    assert (data_dir / "reports" / f"{name}.html").is_file()
    assert storage.report_html_path(name) == (data_dir / "reports" / f"{name}.html").resolve()
    assert storage.load_report(name)["top_jobs"][0]["id"] == "job:111"
    assert storage.load_report("../x") is None and storage.report_html_path("../x") is None


def test_rapport_non_reecrivable():
    name = write()
    first = storage.load_report(name)
    second = write()  # même mot-clé, même jour : nouveau rapport, l'ancien est intact
    assert second != name and storage.load_report(name) == first
    with pytest.raises(DBAPIError, match="immuable"), session.transaction() as s:
        s.execute(update(models.Report).where(models.Report.slug == name).values(summary={}))


def test_c1_sur_get_api_reports(client):
    name = write()
    rows = client.get("/api/reports").json()
    assert rows[0]["name"] == name and ISO_UTC.match(rows[0]["generated_at"])
    assert rows[0]["jobs"] == 1 and rows[0]["html"] is True and rows[0]["keyword"] == "Data Engineer"
    assert ISO_UTC.match(client.get(f"/api/reports/{name}").json()["generated_at"])
    assert client.get(f"/api/reports/{name}/html").status_code == 200
    assert client.get("/api/reports/inconnu/html").json()["detail"] == "Rapport introuvable."


def test_html_absent_404(client):
    name = write()
    (config.REPORTS_DIR / f"{name}.html").unlink()
    response = client.get(f"/api/reports/{name}/html")
    assert response.status_code == 404 and response.json()["detail"] == "Rapport introuvable."
    assert client.get("/api/reports").json()[0]["html"] is False


def _trained(n_pos: int = 4, n_neg: int = 3) -> matching.MatchingModel:
    rng = np.random.default_rng(0)
    X = rng.random((n_pos + n_neg, len(matching.FEATURES)))
    y = np.array([1.0] * n_pos + [0.0] * n_neg)
    model = matching.MatchingModel()
    model.fit(X, y, epochs=50)
    return model


def test_modele_absent_poids_a_priori():
    model = matching.MatchingModel.load()
    assert not model.trained and model.bias == matching.PRIOR_BIAS


def test_un_seul_modele_actif_versions_croissantes():
    first = _trained()
    assert ISO_UTC.match(first.trained_at)  # C-1 dès l'entraînement, sans microsecondes ni +00:00
    first.save()
    second = _trained(5, 2)
    second.save()
    with session.transaction() as s:
        rows = (
            s.execute(select(models.MatchingModelRecord).order_by(models.MatchingModelRecord.version)).scalars().all()
        )
        assert [(r.version, r.is_active) for r in rows] == [(1, False), (2, True)]
        assert (rows[1].n_positive, rows[1].n_negative, rows[1].learned) == (5, 2, True)
    loaded = matching.MatchingModel.load()
    assert loaded.trained and loaded.n_labels == 7
    assert np.allclose(loaded.weights, second.weights) and np.allclose(loaded.std, second.std)
    assert ISO_UTC.match(loaded.trained_at)


def test_modele_aux_criteres_obsoletes_ignore():
    _trained().save()
    with session.transaction() as s:
        s.execute(update(models.MatchingModelRecord).values(feature_names=["ancien"]))
    assert not matching.MatchingModel.load().trained
