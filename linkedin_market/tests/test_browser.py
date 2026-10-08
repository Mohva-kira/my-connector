"""Vérifie que le navigateur Chrome démarre sans option de dissimulation (hors ligne)."""

from linkedin_market import browser, config


class _FakeChrome:
    """Doublure de ``webdriver.Chrome`` : capture les options sans lancer de navigateur."""

    def __init__(self, service=None, options=None):
        self.service = service
        self.options = options


def test_chrome_sans_option_de_dissimulation(monkeypatch):
    monkeypatch.setattr(browser.webdriver, "Chrome", _FakeChrome)
    monkeypatch.setattr(config, "browser_binary", lambda: None)
    monkeypatch.setattr(config, "headless", lambda: True)
    monkeypatch.setattr(browser.shutil, "which", lambda _name: None)

    driver = browser._chrome()

    options = driver.options
    assert not any("AutomationControlled" in arg for arg in options.arguments)
    assert "excludeSwitches" not in options.experimental_options
    assert "--window-size=1366,900" in options.arguments
