"""Création du navigateur Selenium, portable Linux / Windows.

Aucun chemin de driver codé en dur : Selenium Manager (Selenium >= 4.6)
télécharge le bon driver, sauf si un ``geckodriver``/``chromedriver`` est
déjà dans le PATH (cas de Firefox installé en snap sous Ubuntu).
"""

from __future__ import annotations

import logging
import random
import shutil
import time

from selenium import webdriver
from selenium.common.exceptions import WebDriverException

from . import config

log = logging.getLogger(__name__)


def human_pause(caution: config.Caution, factor: float = 1.0) -> None:
    """Pause aléatoire pour ne pas avoir un rythme de robot."""
    time.sleep(random.uniform(caution.min_delay, caution.max_delay) * factor)


def _chrome() -> webdriver.Chrome:
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service

    options = Options()
    options.page_load_strategy = "eager"  # LinkedIn ne finit jamais d'émettre « load » (flux, pubs) : DOM prêt suffit
    if config.browser_binary():
        options.binary_location = config.browser_binary()
    if config.headless():
        options.add_argument("--headless=new")
    options.add_argument("--window-size=1366,900")
    driver_path = shutil.which("chromedriver")
    service = Service(driver_path) if driver_path else Service()
    return webdriver.Chrome(service=service, options=options)


def _firefox() -> webdriver.Firefox:
    from selenium.webdriver.firefox.options import Options
    from selenium.webdriver.firefox.service import Service

    options = Options()
    options.page_load_strategy = "eager"  # LinkedIn ne finit jamais d'émettre « load » (flux, pubs) : DOM prêt suffit
    if config.browser_binary():
        options.binary_location = config.browser_binary()
    if config.headless():
        options.add_argument("-headless")
    driver_path = shutil.which("geckodriver")
    service = Service(driver_path) if driver_path else Service()
    return webdriver.Firefox(service=service, options=options)


def create_driver() -> webdriver.Remote:
    name = config.browser_name()
    builders = {"chrome": [_chrome], "firefox": [_firefox]}.get(name, [_chrome, _firefox])
    last_error: Exception | None = None
    for build in builders:
        try:
            driver = build()
            driver.set_page_load_timeout(60)
            log.info("Navigateur démarré : %s", build.__name__.strip("_"))
            return driver
        except WebDriverException as exc:
            last_error = exc
            log.warning("Impossible de démarrer %s : %s", build.__name__.strip("_"), exc.msg)
    raise RuntimeError(f"Aucun navigateur utilisable (LINKEDIN_BROWSER={name}) : {last_error}")
