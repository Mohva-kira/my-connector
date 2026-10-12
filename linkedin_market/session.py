"""Connexion LinkedIn avec session persistée 72 h.

Les cookies sont sauvegardés après chaque connexion par identifiants, chiffrés (Fernet, ``LM_SECRET_KEY``) dans
``platform_sessions`` (I-15). Ils sont réutilisés jusqu'à ``expires_at`` = dernière connexion + ``SESSION_TTL_HOURS``
; au-delà, la ligne est supprimée et une nouvelle connexion l'écrase. La réutilisation ne prolonge pas la session.
Aucun cookie n'est jamais journalisé.
"""

from __future__ import annotations

import getpass
import json
import logging
import time
from datetime import UTC, datetime, timedelta

from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from . import activity, config
from .db import crypto
from .db.repositories import platform_sessions as sessions_repo
from .db.repositories import user_transaction

log = logging.getLogger(__name__)

FEED_URL = "https://www.linkedin.com/feed/"
LOGIN_URL = "https://www.linkedin.com/login"
EMAIL_SELECTOR = "#username, input[name='session_key'], input[type='email'], input[autocomplete~='username']"
PASSWORD_SELECTOR = "#password, input[name='session_password'], input[type='password']"
# Délai laissé à l'utilisateur pour résoudre un captcha / une validation 2FA.
MANUAL_CHECK_TIMEOUT = 180
# Attente du fil d'actualité après l'envoi des identifiants, puis après la reprise par cookies.
FEED_TIMEOUT = 20
RESTORE_PAUSE = 4.0
VERIFICATION_NOT_DONE = (
    "LinkedIn demande une vérification que tu n'as pas terminée à temps : les actions navigateur sont suspendues "
    "6 h. Termine-la sur LinkedIn puis lève le blocage depuis le tableau de bord."
)


SOURCE_CODE = "linkedin"


def _now() -> datetime:
    return datetime.now(UTC)


def _forget_session() -> None:
    with user_transaction() as (s, uid):
        sessions_repo.remove(s, uid, SOURCE_CODE)


def _load_session() -> list[dict] | None:
    """Cookies de la dernière connexion s'ils ont moins de 72 h ; sinon la session est supprimée."""
    with user_transaction() as (s, uid):
        row = sessions_repo.get(s, uid, SOURCE_CODE)
        if row is None:
            return None
        token, last_login_at, expires_at = row.cookies_encrypted, row.last_login_at, row.expires_at
    if _now() >= expires_at:
        age = _now() - last_login_at
        log.info("Session expirée (%s h) : elle sera écrasée.", int(age.total_seconds() // 3600))
        _forget_session()
        return None
    try:
        cookies = json.loads(crypto.decrypt(token))
    except crypto.SecretKeyError, ValueError:
        log.warning("Session enregistrée illisible (clé LM_SECRET_KEY changée ?) : nouvelle connexion.")
        _forget_session()
        return None
    return cookies or None


def _save_session(driver) -> None:
    now = _now()
    token = crypto.encrypt(json.dumps(driver.get_cookies()).encode("utf-8"))
    with user_transaction() as (s, uid):
        sessions_repo.save(s, uid, SOURCE_CODE, token, now, now + timedelta(hours=config.SESSION_TTL_HOURS))


def _is_logged_in(driver) -> bool:
    url = driver.current_url
    return "/feed" in url or "/in/" in url or "/mynetwork" in url


def _restore(driver, cookies: list[dict]) -> bool:
    driver.get("https://www.linkedin.com/")
    for cookie in cookies:
        cookie = {
            k: v for k, v in cookie.items() if k in ("name", "value", "domain", "path", "expiry", "secure", "httpOnly")
        }
        try:
            driver.add_cookie(cookie)
        except WebDriverException:
            continue
    driver.get(FEED_URL)
    time.sleep(RESTORE_PAUSE)
    if "/checkpoint" in driver.current_url:
        _blocked_at_login()
    return _is_logged_in(driver)


class MissingCredentialsError(RuntimeError):
    """Identifiants LinkedIn absents en mode non interactif (interface web)."""


class VerificationNotDoneError(RuntimeError):
    """Vérification LinkedIn (captcha, 2FA, checkpoint) non terminée à la connexion : repos de 6 h (I-06, E-5)."""


def _blocked_at_login() -> None:
    """Événement ``blocked`` (``where="connexion"``) puis arrêt : aucune nouvelle tentative (I-06)."""
    activity.record("blocked", where="connexion", source="linkedin")
    log.error("Vérification LinkedIn non terminée : actions navigateur suspendues 6 h.")
    raise VerificationNotDoneError(VERIFICATION_NOT_DONE)


def _credentials(interactive: bool = True) -> tuple[str, str]:
    email, password = config.linkedin_credentials()
    if not interactive and not (email and password):
        raise MissingCredentialsError(
            "Identifiants LinkedIn absents : renseigne LINKEDIN_EMAIL et LINKEDIN_PASSWORD dans le fichier .env."
        )
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
        WebDriverWait(driver, FEED_TIMEOUT).until(EC.url_contains("/feed"))
    except TimeoutException:
        message = (
            f"LinkedIn demande une vérification (captcha, code 2FA...). "
            f"Termine-la dans le navigateur : j'attends jusqu'à {MANUAL_CHECK_TIMEOUT} s."
        )
        if on_manual_check:
            on_manual_check(message)
        else:
            print("\n" + message)
        try:
            WebDriverWait(driver, MANUAL_CHECK_TIMEOUT).until(EC.url_contains("/feed"))
        except TimeoutException:
            _blocked_at_login()


def session_age_hours() -> float | None:
    """Âge de la session enregistrée (None si absente), sans la supprimer."""
    with user_transaction() as (s, uid):
        row = sessions_repo.get(s, uid, SOURCE_CODE)
        last_login_at = row.last_login_at if row else None
    return (_now() - last_login_at).total_seconds() / 3600 if last_login_at else None


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
        _forget_session()
    _login_with_password(driver, interactive, on_manual_check)
    _save_session(driver)
    log.info("Connecté à LinkedIn, session sauvegardée pour %s h.", config.SESSION_TTL_HOURS)
