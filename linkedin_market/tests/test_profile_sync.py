import pytest

from linkedin_market import config, profile_sync
from linkedin_market.tests.browser_fixtures import FIXTURES, url_of

CAUTION = config.Caution("test", 1, 0.0, 0.0)


def _resolver(form_suffix: str = ""):
    base = (FIXTURES / "in" / "jane-doe.html").as_uri()
    local = {
        profile_sync.ME_URL: base,
        f"{base}/details/skills/": url_of("skills_details.html"),
        f"{base}/edit/forms/skills/new/": url_of("skills_form.html") + form_suffix,
    }
    return lambda u: local[u]


def test_same_skill():
    assert profile_sync.same_skill("REST APIs", "rest api")  # nom LinkedIn, compétence reconnue
    assert profile_sync.same_skill("JavaScript", "javascript")
    assert profile_sync.same_skill("Kubernetes", "kubernetes")
    assert not profile_sync.same_skill("Kubernetes Operators", "terraform")
    assert not profile_sync.same_skill(
        "Conception et documentation d'une API publique", "rest api"
    )  # phrase, pas un nom
    assert not profile_sync.same_skill("", "python")


def test_adds_skill_with_linkedin_name(driver):
    logs = []
    result = profile_sync.add_skill(driver, "Kubernetes", CAUTION, log_fn=logs.append, resolve=_resolver())
    assert result == {"skill": "kubernetes", "linkedin_name": "Kubernetes", "state": "ajoutée"}
    assert (
        driver.execute_script("return window.__saved") == "Kubernetes"
    )  # suggestion exacte, pas « Kubernetes Operators »
    assert any("ajoutée au profil" in line for line in logs)


def test_free_text_when_no_suggestion(driver):
    result = profile_sync.add_skill(driver, "fintech", CAUTION, resolve=_resolver())
    assert result["state"] == "ajoutée" and driver.execute_script("return window.__saved") == "fintech"


def test_existing_skill_is_not_added_again(driver):
    result = profile_sync.add_skill(driver, "javascript", CAUTION, resolve=_resolver())
    assert result == {"skill": "javascript", "linkedin_name": "JavaScript", "state": "déjà_présente"}


def test_linkedin_error_is_reported(driver):
    with pytest.raises(profile_sync.ProfileSyncError, match="nombre maximum"):
        profile_sync.add_skill(driver, "terraform", CAUTION, resolve=_resolver("#full"), timeout=3)


def test_navigation_timeout_is_readable():
    from selenium.common.exceptions import TimeoutException

    class SlowDriver:
        def get(self, url):
            raise TimeoutException("Navigation timed out after 60000 ms")

    with pytest.raises(profile_sync.ProfileSyncError, match="trop de temps"):
        profile_sync.add_skill(SlowDriver(), "kubernetes", CAUTION)
