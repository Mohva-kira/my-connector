"""Pilotes Selenium réels sur des pages HTML locales (pas de LinkedIn)."""

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def url_of(name: str) -> str:
    return (FIXTURES / name).as_uri()


@pytest.fixture(scope="module")
def driver():
    import os

    os.environ["LINKEDIN_HEADLESS"] = "1"
    os.environ.setdefault("LINKEDIN_BROWSER", "firefox")
    from linkedin_market import browser

    try:
        drv = browser.create_driver()
    except Exception as exc:  # noqa: BLE001 - pas de navigateur : on saute ces tests
        pytest.skip(f"Aucun navigateur utilisable : {exc}")
    yield drv
    drv.quit()


@pytest.fixture
def workdir():
    """Dossier de travail sous le dépôt : un Firefox installé en snap ne lit pas /tmp (téléversement de fichier)."""
    import shutil

    path = FIXTURES / "_work"
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


class FakeHooks:
    def __init__(self, replies: dict[str, str] | None = None, confirm: str = "Soumettre"):
        self.replies, self.confirm = replies or {}, confirm
        self.asked: list[dict] = []
        self.logs: list[str] = []

    def log(self, message: str) -> None:
        self.logs.append(message)

    def check(self) -> None:
        pass

    def ask(self, question, field_id="", options=None, *, suggestion="", review=""):
        self.asked.append({"question": question, "field": field_id, "options": options, "suggestion": suggestion, "review": review})
        if field_id == "confirm_submit":
            return self.confirm
        for needle, reply in self.replies.items():
            if needle.lower() in question.lower():
                return reply
        raise AssertionError(f"Question inattendue : {question}")
