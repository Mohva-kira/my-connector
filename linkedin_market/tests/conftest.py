"""Isole les fichiers de données de chaque test dans un dossier temporaire."""

import pytest

from linkedin_market import config


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
