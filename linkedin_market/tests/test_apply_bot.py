import pytest

from linkedin_market import apply_bot, config, settings
from linkedin_market.tests.browser_fixtures import FakeHooks, url_of

pytestmark = pytest.mark.db  # réglages, historique et rapports en PostgreSQL (vague 1)

CAUTION = config.Caution("test", 1, 0.0, 0.0)
JOB = {"id": "job:1", "title": "Senior Python Developer", "url": ""}


def run(driver, page, hooks, workdir, cfg=None):
    pdf = workdir / "cv.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    bot = apply_bot.EasyApply(
        driver,
        hooks,
        cv_pdf=pdf,
        caution=CAUTION,
        cfg=cfg or {**settings.DEFAULT_SETTINGS, "contact_email": "me@example.com", "contact_phone": "+2250102030405"},
    )
    bot.step_timeout = 2.0
    return bot.run({**JOB, "url": url_of(page)})


def test_full_flow_asks_unknown_questions_and_submits(driver, workdir):
    hooks = FakeHooks({"years of experience": "10", "legally authorized": "Yes"})
    result = run(driver, "easy_apply.html", hooks, workdir)
    assert result.state == "postulée"
    questions = [a["question"] for a in hooks.asked]
    # email et téléphone viennent des paramètres : jamais demandés ; les autres sont posées à l'utilisateur
    assert not any("email" in q.lower() or "phone" in q.lower() for q in questions)
    assert any("years of experience" in q for q in questions) and any("authorized" in q for q in questions)
    # la relecture montre ce qui va partir, et le CV a été téléversé
    review = next(a for a in hooks.asked if a["field"] == "confirm_submit")["review"]
    assert "me@example.com" in review and "Years: 10" in review and "Authorized: Yes" in review
    assert any("CV téléversé" in line for line in hooks.logs)
    assert driver.execute_script("return window.__submitted") is True
    assert driver.execute_script("return window.__follow") is False  # « suivre l'entreprise » décoché


def test_decline_at_confirmation_sends_nothing(driver, workdir):
    hooks = FakeHooks({"years of experience": "10", "legally authorized": "Yes"}, confirm="Annuler")
    result = run(driver, "easy_apply.html", hooks, workdir)
    assert result.state == "annulée"
    assert driver.execute_script("return window.__submitted") is False


def test_remembered_answers_are_used_without_asking(driver, workdir, monkeypatch):
    settings.remember_answer("How many years of experience do you have with Python?", "8")
    settings.remember_answer("Are you legally authorized to work remotely? *", "Yes")
    hooks = FakeHooks()
    result = run(driver, "easy_apply.html", hooks, workdir)
    assert result.state == "postulée"
    assert [a["field"] for a in hooks.asked] == ["confirm_submit"]
    assert "Years: 8" in hooks.asked[0]["review"]


def test_missing_contact_settings_are_asked_not_guessed(driver, workdir):
    cfg = {**settings.DEFAULT_SETTINGS}  # ni email ni téléphone configurés
    hooks = FakeHooks(
        {"email": "other@example.com", "phone": "0707070707", "years of experience": "5", "legally authorized": "No"}
    )
    result = run(driver, "easy_apply.html", hooks, workdir, cfg)
    assert result.state == "postulée"
    assert any("Email" in a["question"] for a in hooks.asked)
    assert "other@example.com" in next(a for a in hooks.asked if a["field"] == "confirm_submit")["review"]


def test_external_and_already_applied_pages(driver, workdir):
    assert run(driver, "external.html", FakeHooks(), workdir).state == "externe"
    assert run(driver, "already.html", FakeHooks(), workdir).state == "déjà_postulée"


def test_option_not_in_choices_fails_cleanly(driver, workdir):
    hooks = FakeHooks({"years of experience": "10", "legally authorized": "Peut-être"})
    with pytest.raises(apply_bot.ApplyError, match="absente des choix"):
        run(driver, "easy_apply.html", hooks, workdir)
    assert driver.execute_script("return window.__submitted") is False
