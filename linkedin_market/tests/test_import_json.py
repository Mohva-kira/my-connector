"""Import des fichiers historiques (vague 1, sous-lot 1d) sur des données **synthétiques** (``fixtures/data``).

Jetons des fixtures remplacés à la copie : ``@J-n@`` = date-heure locale naïve d'il y a n jours, ``@D-n@`` = date
d'il y a n jours, ``@H-n@`` = date-heure UTC d'il y a n heures, ``@DATA@`` = dossier de données du test.
"""

import re
import shutil
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import func, select

from linkedin_market import activity, application, matching, settings, storage
from linkedin_market import session as lm_session
from linkedin_market.db import import_json, models, session
from linkedin_market.web.tasks import TaskManager

pytestmark = pytest.mark.db
FIXTURES = Path(__file__).parent / "fixtures" / "data"
_TOKEN = re.compile(r"@([JDH])-(\d+)@")


def _render(text: str, data_dir: Path) -> str:
    def value(match: re.Match) -> str:
        kind, n = match.group(1), int(match.group(2))
        if kind == "J":
            return (datetime.now() - timedelta(days=n)).replace(microsecond=0).isoformat()
        if kind == "D":
            return (date.today() - timedelta(days=n)).isoformat()
        return (datetime.now(UTC) - timedelta(hours=n)).isoformat()

    return _TOKEN.sub(value, text).replace("@DATA@", data_dir.as_posix())


@pytest.fixture
def legacy(data_dir: Path) -> Path:
    """Copie des fixtures synthétiques dans le dossier de données du test (jamais le vrai)."""
    for source in FIXTURES.rglob("*"):
        if source.is_file():
            target = data_dir / source.relative_to(FIXTURES)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(_render(source.read_text(encoding="utf-8"), data_dir), encoding="utf-8")
    for folder in ("20261001-fictive-data-architect", "20261002-absente-offre"):
        (data_dir / "applications" / folder).mkdir(parents=True, exist_ok=True)
        (data_dir / "applications" / folder / "cv.md").write_text("# CV fictif", encoding="utf-8")
    return data_dir


def _count(model) -> int:
    with session.transaction() as s:
        return s.execute(select(func.count()).select_from(model)).scalar_one()


def test_import_complet_puis_sans_effet(legacy):
    counts = import_json.import_legacy_files(legacy)
    assert counts["collected_items"] == 6  # 1001, 1003, 1004, post récent, person, 2001 minimal
    assert counts["labels"] == 2 and counts["ignored_labels"] == 1
    assert counts["reports"] == 1 and counts["applications"] == 2 and counts["company_contacts"] == 2
    assert counts["application_events"] == 3 and counts["platform_sessions"] == 1
    assert counts["activity_events"] == 2 and counts["ignored_activity_events"] == 2
    assert counts["tasks"] == 99 and counts["ignored_tasks"] == 1  # 100 dernières, kind « t » ignoré
    assert counts["matching_models"] == 1 and counts["memorized_answers"] == 2
    # Fichiers renommés après le COMMIT, dossiers conservés.
    assert (legacy / "history.jsonl.migrated").is_file() and not (legacy / "history.jsonl").exists()
    assert (legacy / "reports" / "data-engineer-20261001-1200.json.migrated").is_file()
    assert (legacy / "reports" / "data-engineer-20261001-1200.html").is_file()
    assert (legacy / "applications" / "20261001-fictive-data-architect" / "cv.md").is_file()
    before = {m: _count(m) for m in (models.CollectedItem, models.Task, models.ActivityEvent, models.Report)}
    assert import_json.import_legacy_files(legacy) == {}
    assert {m: _count(m) for m in before} == before


def test_erreur_simulee_rien_insere_rien_renomme(legacy, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("panne simulée")

    monkeypatch.setattr(import_json, "_tasks", boom)
    with pytest.raises(RuntimeError, match="panne simulée"):
        import_json.import_legacy_files(legacy)
    assert _count(models.CollectedItem) == 0 and _count(models.Report) == 0 and _count(models.Label) == 0
    assert not list(legacy.rglob("*.migrated")) and (legacy / "history.jsonl").is_file()


def test_renommage_interrompu_apres_commit_sans_doublon(legacy, monkeypatch):
    """Crash simulé après le COMMIT et avant la fin des renommages : la relance renomme sans réimporter."""
    real_rename = Path.rename
    calls = {"n": 0}

    def crash_after_two(self, target):
        calls["n"] += 1
        if calls["n"] > 2:
            raise OSError("arrêt simulé pendant le renommage")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", crash_after_two)
    with pytest.raises(OSError, match="arrêt simulé"):
        import_json.import_legacy_files(legacy)
    monkeypatch.setattr(Path, "rename", real_rename)
    journals = (models.ActivityEvent, models.ApplicationEvent, models.CompanyContact)
    before = {m: _count(m) for m in journals}
    assert all(before.values()) and len(list(legacy.rglob("*.migrated"))) == 2
    with session.transaction() as s:
        known = s.execute(select(func.count()).select_from(models.LegacyImport)).scalar_one()
    assert known == 11  # 10 fichiers + 1 rapport, inscrits dans la transaction de l'import

    counts = import_json.import_legacy_files(legacy)
    assert counts == {"already_imported_files": 9}
    assert {m: _count(m) for m in journals} == before
    assert not [p for p in legacy.rglob("*.json*") if not p.name.endswith(".migrated")]
    assert import_json.import_legacy_files(legacy) == {}


def test_rapports_identiques_sous_deux_noms_importes_tous_deux(legacy):
    """Même contenu, nom différent : deux rapports distincts ; la relance n'ajoute aucun doublon."""
    reports = legacy / "reports"
    shutil.copyfile(reports / "data-engineer-20261001-1200.json", reports / "data-engineer-copie-20261002-0900.json")
    counts = import_json.import_legacy_files(legacy)
    assert counts["reports"] == 2 and "already_imported_files" not in counts
    with session.transaction() as s:
        slugs = s.execute(select(models.Report.slug).order_by(models.Report.slug)).scalars().all()
        names = s.execute(select(models.LegacyImport.file_name).where(models.LegacyImport.file_name.like("reports/%")))
        assert sorted(names.scalars()) == [
            "reports/data-engineer-20261001-1200.json",
            "reports/data-engineer-copie-20261002-0900.json",
        ]
    assert slugs == ["data-engineer-20261001-1200", "data-engineer-copie-20261002-0900"]
    assert import_json.import_legacy_files(legacy) == {}
    assert _count(models.Report) == 2

    # Fichiers restés sous leur nom (renommage interrompu) : simple renommage, aucun doublon.
    for migrated in reports.glob("*.json.migrated"):
        migrated.rename(migrated.with_name(migrated.name.removesuffix(".migrated")))
    assert import_json.import_legacy_files(legacy) == {"already_imported_files": 2}
    assert _count(models.Report) == 2


def test_conversions_reglages_e3_e7(legacy):
    import_json.import_legacy_files(legacy)
    values = settings.load_settings()
    assert values["interpreter"] == "local"  # none -> local (E-7)
    assert values["max_items"] == 40  # 40 <= 100 en prudent : conservé (E-3)
    assert (values["caution"], values["apply_daily_max"], values["linkedin_skill_sync"]) == ("prudent", 8, False)
    assert settings.find_answer("Authorized to work?") == "Yes"
    overrides = settings.load_profile_settings()
    assert set(overrides["added_skills"]) == {"rust", "go"} and overrides["removed_skills"] == ["java"]
    assert overrides["seniority"] == "senior" and overrides["years_experience"] == 9
    assert storage.load_linkedin_profile_text().startswith("Alex Exemple")


def test_historique_retention_et_minimisation(legacy):
    import_json.import_legacy_files(legacy)
    with session.transaction() as s:
        rows = {r.external_id: r for r in s.execute(select(models.CollectedItem)).scalars()}
    assert "1002" not in rows and "1005" not in rows  # offres de plus de 90 jours sans annotation ni candidature
    assert "urn:li:activity:2" not in rows  # post de plus de 30 jours
    assert rows["1003"].purge_after is None and rows["1004"].purge_after is None  # annotée / candidatée conservées
    assert rows["1001"].title == "Senior Data Engineer"  # dernière version gagnante
    assert rows["1001"].keywords == ["data engineer", "python"]  # mots-clés de toutes les lignes, normalisés
    assert rows["1003"].keywords == ["data engineer"]
    assert "job:1001" in {i.id for i in storage.load_items("Python")}
    assert "job:1001" in {i.id for i in storage.load_items("Data Engineer")}
    assert rows["1001"].kind == "job" and rows["1001"].source_code == "linkedin"
    person = rows["personne-fictive"]
    assert person.text is None and {"python", "aws"} <= set(person.skills)  # I-09
    assert person.collected_at.tzinfo is not None
    assert abs(person.collected_at - (datetime.now(UTC) - timedelta(days=4))) < timedelta(minutes=5)
    assert storage.load_labels() == {"job:1001": 1, "job:1003": 0}


def test_candidatures_e15_et_offre_minimale(legacy):
    import_json.import_legacy_files(legacy)
    entry = application.get_application("job:1004")
    assert entry["status"] == "postulée" and entry["folder"].endswith("applications/20261001-fictive-data-architect")
    assert application.read_files(entry)["cv"] == "# CV fictif"
    assert [h["kind"] for h in entry["history"]] == ["status", "apply", "message"]  # C-6
    assert entry["contacts"][0]["full_name"] == "Recruteuse Fictive"  # name -> full_name (E-15)
    assert (
        entry["contacts"][0]["source"] == "linkedin_equipe_recrutement" and entry["contacts"][1]["source"] == "manuel"
    )
    minimal = application.get_application("job:2001")
    assert minimal["title"] == "Offre hors historique" and minimal["generator"] == "claude"
    with session.transaction() as s:
        legal = s.execute(select(models.CompanyContact.legal_basis).distinct()).scalars().all()
    assert legal == ["interet_legitime"]


def test_rapport_slug_et_modele(legacy):
    import_json.import_legacy_files(legacy)
    rows = storage.list_reports()
    assert rows[0]["name"] == "data-engineer-20261001-1200" and rows[0]["html"] is True
    assert rows[0]["generated_at"].endswith("Z")
    with session.transaction() as s:
        report = s.execute(select(models.Report)).scalar_one()
        assert report.interpreter_used == "local" and report.html_path == "reports/data-engineer-20261001-1200.html"
    assert storage.load_report("data-engineer-20261001-1200")["interpreter"] == "none"  # contenu conservé (I-02)
    model = matching.MatchingModel.load()
    assert model.trained and model.bias == -2.0
    with session.transaction() as s:
        record = s.execute(select(models.MatchingModelRecord)).scalar_one()
        # labels.json : deux 1 et un 0 (n_labels du fichier = 6, ignoré)
        assert (record.n_positive, record.n_negative) == (2, 1)


def test_session_recente_chiffree_et_dates_naives(legacy):
    import_json.import_legacy_files(legacy)
    assert lm_session._load_session()[0]["value"] == "cookie-synthetique-secret"
    with session.transaction() as s:
        token = s.execute(select(models.PlatformSession.cookies_encrypted)).scalar_one()
    assert b"cookie-synthetique-secret" not in token
    events = activity.load()
    assert [e["kind"] for e in events] == ["run", "apply"] and all(e["at"].endswith("Z") for e in events)
    assert activity.count_today("apply", today=date.today() - timedelta(days=1)) == 1


def test_session_de_plus_de_72h_ignoree(legacy):
    path = legacy / "session.json"
    old = (datetime.now(UTC) - timedelta(hours=73)).isoformat()
    path.write_text(re.sub(r'"logged_in_at": "[^"]+"', f'"logged_in_at": "{old}"', path.read_text()), encoding="utf-8")
    counts = import_json.import_legacy_files(legacy)
    assert counts["ignored_sessions"] == 1 and lm_session._load_session() is None


def test_taches_relues(legacy):
    import_json.import_legacy_files(legacy)
    manager = TaskManager()
    titles = {t.title for t in manager.list(200)}
    assert "Tâche 104" in titles and "Tâche 4" not in titles and "Tâche 101" not in titles
    manager.shutdown(wait=True)


def test_dossier_vide_sans_effet(data_dir):
    shutil.rmtree(data_dir / "reports")
    assert import_json.import_legacy_files(data_dir) == {}
