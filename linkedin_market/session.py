"""Connexion LinkedIn avec session persistée 72 h.

Les cookies sont sauvegardés après chaque connexion par identifiants.
Ils sont réutilisés tant que la dernière connexion date de moins de
``SESSION_TTL_HOURS`` ; au-delà, le fichier est supprimé et une nouvelle
connexion l'écrase. La réutilisation ne prolonge pas la session.
"""

from __future__ import annotations

import getpass
import json
import logging
import time
from datetime import datetime, timedelta, timezone

from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from . import config

log = logging.getLogger(__name__)

FEED_URL = "https://www.linkedin.com/feed/"
LOGIN_URL = "https://www.linkedin.com/login"
EMAIL_SELECTOR = ("#username, input[name='session_key'], input[type='email'], "
                  "input[autocomplete~='username']")
PASSWORD_SELECTOR = "#password, input[name='session_password'], input[type='password']"
# Délai laissé à l'utilisateur pour résoudre un captcha / une validation 2FA.
MANUAL_CHECK_TIMEOUT = 180


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _load_session() -> list[dict] | None:
    if not config.SESSION_FILE.exists():
        return None
    try:
        data = json.loads(config.SESSION_FILE.read_text(encoding="utf-8"))
        logged_in_at = datetime.fromisoformat(data["logged_in_at"])
    except (ValueError, KeyError, json.JSONDecodeError):
        config.SESSION_FILE.unlink(missing_ok=True)
        return None
    age = _now() - logged_in_at
    if age > timedelta(hours=config.SESSION_TTL_HOURS):
        log.info("Session expirée (%s h) : elle sera écrasée.", int(age.total_seconds() // 3600))
        config.SESSION_FILE.unlink(missing_ok=True)
        return None
    return data.get("cookies") or None


def _save_session(driver) -> None:
    config.ensure_data_dirs()
    payload = {"logged_in_at": _now().isoformat(), "cookies": driver.get_cookies()}
    config.SESSION_FILE.write_text(json.dumps(payload), encoding="utf-8")
    try:
        config.SESSION_FILE.chmod(0o600)
    except OSError:  # Windows : chmod partiel, sans conséquence
        pass


def _is_logged_in(driver) -> bool:
    url = driver.current_url
    return "/feed" in url or "/in/" in url or "/mynetwork" in url


def _restore(driver, cookies: list[dict]) -> bool:
    driver.get("https://www.linkedin.com/")
    for cookie in cookies:
        cookie = {k: v for k, v in cookie.items() if k in ("name", "value", "domain", "path", "expiry", "secure", "httpOnly")}
        try:
            driver.add_cookie(cookie)
        except WebDriverException:
            continue
    driver.get(FEED_URL)
    time.sleep(4)
    return _is_logged_in(driver)


class MissingCredentialsError(RuntimeError):
    """Identifiants LinkedIn absents en mode non interactif (interface web)."""


def _credentials(interactive: bool = True) -> tuple[str, str]:
    email, password = config.linkedin_credentials()
    if not interactive and not (email and password):
        raise MissingCredentialsError(
            "Identifiants LinkedIn absents : renseigne LINKEDIN_EMAIL et LINKEDIN_PASSWORD dans le fichier .env.")
    if not email:
        email = input("Email LinkedIn : ").strip()
    if not password:
        password = getpass.getpass("Mot de passe LinkedIn (masqué) : ")
    return email, password


def _visible(driver, selector: str):
    """Premier élément visible : la page de connexion contient deux formulaires
    (dont un masqué) et des id générés dynamiquement."""
    for element in driver.find_elements(By.CSS_SELECTOR, selector):
        if element.is_displayed():
            return element
    return False


def _login_with_password(driver, interactive: bool = True, on_manual_check=None) -> None:
    email, password = _credentials(interactive)
    driver.get(LOGIN_URL)
    email_field = WebDriverWait(driver, 30).until(lambda d: _visible(d, EMAIL_SELECTOR))
    email_field.clear()
    email_field.send_keys(email)
    field = WebDriverWait(driver, 10).until(lambda d: _visible(d, PASSWORD_SELECTOR))
    field.send_keys(password)
    field.send_keys(Keys.RETURN)
    try:
        WebDriverWait(driver, 20).until(EC.url_contains("/feed"))
    except TimeoutException:
        message = (f"LinkedIn demande une vérification (captcha, code 2FA...). "
                   f"Termine-la dans le navigateur : j'attends jusqu'à {MANUAL_CHECK_TIMEOUT} s.")
        if on_manual_check:
            on_manual_check(message)
        else:
            print("\n" + message)
        WebDriverWait(driver, MANUAL_CHECK_TIMEOUT).until(EC.url_contains("/feed"))


def session_age_hours() -> float | None:
    """Âge de la session enregistrée (None si absente/invalide), sans la supprimer."""
    try:
        data = json.loads(config.SESSION_FILE.read_text(encoding="utf-8"))
        return (_now() - datetime.fromisoformat(data["logged_in_at"])).total_seconds() / 3600
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def login(driver, interactive: bool = True, on_manual_check=None) -> None:
    """Réutilise la session si elle a moins de 72 h, sinon se reconnecte.

    ``interactive=False`` (interface web) : pas de ``input()``, identifiants depuis ``.env`` ;
    ``on_manual_check(message)`` est appelé quand LinkedIn réclame une vérification manuelle.
    """
    cookies = _load_session()
    if cookies and _restore(driver, cookies):
        log.info("Session LinkedIn réutilisée.")
        return
    if cookies:
        log.info("Cookies refusés par LinkedIn : nouvelle connexion.")
        config.SESSION_FILE.unlink(missing_ok=True)
    _login_with_password(driver, interactive, on_manual_check)
    _save_session(driver)
    log.info("Connecté à LinkedIn, session sauvegardée pour %s h.", config.SESSION_TTL_HOURS)
