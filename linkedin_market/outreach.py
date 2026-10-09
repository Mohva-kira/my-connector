"""Messages à l'entreprise : email (SMTP) ou message LinkedIn (Selenium).

L'appelant exige déjà une confirmation explicite de l'utilisateur ; ce module ne fait qu'envoyer, et
lève ``OutreachError`` avec un message lisible quand quelque chose manque ou échoue.
"""

from __future__ import annotations

import logging
import re
import smtplib
import ssl
import time
from collections.abc import Callable
from email.message import EmailMessage
from email.utils import make_msgid
from pathlib import Path

from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from . import config, logmask
from .browser import human_pause

log = logging.getLogger(__name__)

LINKEDIN_NOTE_MAX = 300
SMTP_TIMEOUT = 25
_EMAIL = logmask.EMAIL_ADDRESS  # sans « , ; < > » : ni destinataires multiples ni injection d'en-tête
_PROFILE_URL = re.compile(r"^https://([a-z]{2,3}\.)?linkedin\.com/in/[^/?#\s]+")


class OutreachError(RuntimeError):
    """Envoi impossible ou échoué (message affichable tel quel)."""


class UncertainDelivery(OutreachError):
    """L'envoi a peut-être eu lieu sans confirmation visible : à vérifier par l'utilisateur, jamais renvoyé."""


# --------------------------------------------------------------------- email
def validate_recipient(address: str) -> str:
    address = (address or "").strip()
    if not _EMAIL.match(address):
        raise OutreachError(f"Adresse email invalide : « {address} ».")
    return address


def build_email(sender: str, to: str, subject: str, body: str, attachments: list[Path] | None = None) -> EmailMessage:
    if "\n" in subject or "\r" in subject:
        raise OutreachError("L'objet ne peut pas contenir de saut de ligne.")
    if not body.strip():
        raise OutreachError("Le message est vide.")
    message = EmailMessage()
    message["From"] = sender
    message["To"] = validate_recipient(to)
    message["Subject"] = subject.strip()
    message["Message-ID"] = make_msgid(domain=sender.rsplit("@", 1)[-1].strip("> ") or None)
    message.set_content(body)
    for path in attachments or []:
        message.add_attachment(
            path.read_bytes(),
            maintype="application",
            subtype="pdf" if path.suffix == ".pdf" else "octet-stream",
            filename=path.name,
        )
    return message


def _smtp_failure(exc: Exception, to: str) -> OutreachError:
    """Erreur SMTP survenue **avant** l'acceptation du message par le serveur : rien n'est parti."""
    if isinstance(exc, smtplib.SMTPAuthenticationError):
        return OutreachError("Le serveur SMTP refuse l'authentification : vérifie SMTP_USER et SMTP_PASSWORD.")
    if isinstance(exc, smtplib.SMTPRecipientsRefused):
        return OutreachError(f"Le destinataire a été refusé par le serveur : {logmask.mask_email(to)}.")
    return OutreachError(f"Envoi impossible ({exc.__class__.__name__} : {exc}).")


def send_email(
    to: str, subject: str, body: str, attachments: list[Path] | None = None, smtp: dict | None = None
) -> dict:
    """Envoie un email ; résultat ``{channel, to, subject, attachments}``."""
    return deliver_email(to, subject, body, attachments, smtp)[0]


def deliver_email(
    to: str, subject: str, body: str, attachments: list[Path] | None = None, smtp: dict | None = None
) -> tuple[dict, str]:
    """Envoie un email ; renvoie le résultat et le ``Message-ID`` (conservé dans ``outreach_messages``).

    Aucune nouvelle tentative : un échec revient à l'utilisateur (risque de doublon chez le destinataire).
    """
    smtp = smtp or config.smtp_config()
    if smtp is None:
        raise OutreachError(
            "SMTP non configuré : renseigne SMTP_HOST et SMTP_FROM (et SMTP_USER / SMTP_PASSWORD) dans le fichier .env."
        )
    insecure = config.smtp_security_error(smtp)
    if insecure:
        raise OutreachError(insecure)
    message = build_email(smtp["from"], to, subject, body, attachments)
    try:
        if smtp["security"] == "ssl":
            client = smtplib.SMTP_SSL(
                smtp["host"], smtp["port"], timeout=SMTP_TIMEOUT, context=ssl.create_default_context()
            )
        else:
            client = smtplib.SMTP(smtp["host"], smtp["port"], timeout=SMTP_TIMEOUT)
    except (smtplib.SMTPException, OSError) as exc:
        raise _smtp_failure(exc, to) from exc
    try:
        if smtp["security"] == "starttls":
            client.starttls(context=ssl.create_default_context())
        if smtp["user"]:
            client.login(smtp["user"], smtp["password"])
        client.send_message(message)
    except (smtplib.SMTPException, OSError) as exc:
        client.close()
        raise _smtp_failure(exc, to) from exc
    # Le serveur a accepté le message : une erreur à la fermeture (QUIT) ne change rien à l'envoi.
    try:
        client.quit()
    except smtplib.SMTPException, OSError:
        client.close()
    result = {
        "channel": "email",
        "to": to,
        "subject": subject.strip(),
        "attachments": [p.name for p in attachments or []],
    }
    return result, str(message["Message-ID"])


# ------------------------------------------------------------------ LinkedIn
MESSAGE_WORDS = ("message", "envoyer un message")
CONNECT_WORDS = ("connect", "se connecter", "inviter", "invite")
ADD_NOTE_WORDS = ("add a note", "ajouter une note")
SEND_WORDS = ("send", "envoyer")
_JS_BUTTONS = """
const words = arguments[0], root = arguments[1] || document;
const out = [];
for (const el of root.querySelectorAll('button, a[role="button"], a[aria-label]')) {
  if (!el.getClientRects().length || el.disabled || el.getAttribute('aria-disabled') === 'true') continue;
  const text = ((el.getAttribute('aria-label') || '') + ' ' + (el.innerText || '')).toLowerCase().replace(/\\s+/g, ' ').trim();
  if (words.some(w => text.includes(w))) out.push(el);
}
return out;
"""  # noqa: E501
_JS_COMPOSER = """
for (const el of document.querySelectorAll('[contenteditable="true"][role="textbox"], div.msg-form__contenteditable, textarea[name="message"]')) {
  if (el.getClientRects().length) return el;
}
return null;
"""  # noqa: E501
_JS_NOTE_FIELD = (
    "for (const el of document.querySelectorAll('textarea')) { if (el.getClientRects().length) return el; } "
    "return null;"
)


def _wait(predicate, timeout: float, check: Callable[[], None]):
    end = time.time() + timeout
    while time.time() < end:
        check()
        value = predicate()
        if value:
            return value
        time.sleep(0.25)
    return None


def send_linkedin_message(
    driver,
    profile_url: str,
    text: str,
    caution: config.Caution,
    *,
    check: Callable[[], None] = lambda: None,
    timeout: float = 12.0,
) -> dict:
    """Message direct (relation de 1er degré) ou, à défaut, demande de connexion avec note (300 caractères max)."""
    if not _PROFILE_URL.match(profile_url or ""):
        raise OutreachError(
            "Profil LinkedIn invalide : renseigne l'URL du profil du recruteur (https://www.linkedin.com/in/…)."
        )
    if not text.strip():
        raise OutreachError("Le message est vide.")
    driver.get(profile_url)
    human_pause(caution, 0.6)
    if any(p in driver.current_url for p in ("/checkpoint", "/authwall", "/login", "/uas/")):
        raise PermissionError(f"LinkedIn a interrompu la navigation ({driver.current_url}).")

    def click(el) -> None:
        try:
            el.click()
        except WebDriverException:
            driver.execute_script("arguments[0].click();", el)

    main = driver.find_elements(By.CSS_SELECTOR, "main")
    scope = main[0] if main else None
    message_buttons = [
        b
        for b in driver.execute_script(_JS_BUTTONS, list(MESSAGE_WORDS), scope) or []
        if "notification" not in (b.get_attribute("aria-label") or "").lower()
    ]
    if message_buttons:
        click(message_buttons[0])
        composer = _wait(lambda: driver.execute_script(_JS_COMPOSER), timeout, check)
        if composer is None:
            raise OutreachError("La fenêtre de messagerie ne s'est pas ouverte (page inattendue).")
        for i, line in enumerate(text.split("\n")):
            if i:
                composer.send_keys(Keys.SHIFT, Keys.ENTER)
            composer.send_keys(line)
        human_pause(caution, 0.2)
        send = [
            b
            for b in driver.execute_script(_JS_BUTTONS, list(SEND_WORDS)) or []
            if (
                b.get_attribute("type") == "submit"
                or "send" in (b.get_attribute("aria-label") or "").lower()
                or b.text.strip().lower() in SEND_WORDS
            )
        ]
        if not send:
            raise OutreachError("Bouton « Envoyer » introuvable dans la messagerie.")
        click(send[-1])
        # Après le clic, le message a peut-être été envoyé : toute erreur (annulation, page) devient « à vérifier ».
        try:
            emptied = _wait(
                lambda: (
                    not (
                        driver.execute_script(_JS_COMPOSER) and (driver.execute_script(_JS_COMPOSER).text or "").strip()
                    )
                ),
                timeout,
                check,
            )
        except Exception as exc:
            raise UncertainDelivery(
                "L'envoi n'a pas pu être confirmé : vérifie la conversation sur LinkedIn avant de réessayer."
            ) from exc
        if not emptied:
            raise UncertainDelivery(
                "L'envoi n'a pas pu être confirmé : vérifie la conversation sur LinkedIn avant de réessayer."
            )
        return {"channel": "linkedin", "mode": "message", "to": profile_url, "truncated": False}

    connect = driver.execute_script(_JS_BUTTONS, list(CONNECT_WORDS), scope) or []
    connect = [b for b in connect if "message" not in (b.get_attribute("aria-label") or "").lower()]
    if not connect:
        raise OutreachError(
            "Ni bouton « Message » ni « Se connecter » sur ce profil (InMail non géré) : "
            "contacte cette personne autrement."
        )
    click(connect[0])
    add_note = _wait(lambda: (driver.execute_script(_JS_BUTTONS, list(ADD_NOTE_WORDS)) or [None])[0], timeout, check)
    if add_note is None:
        raise OutreachError("Impossible d'ajouter une note à la demande de connexion (page inattendue).")
    click(add_note)
    field = _wait(lambda: driver.execute_script(_JS_NOTE_FIELD), timeout, check)
    if field is None:
        raise OutreachError("Champ de note introuvable.")
    note = text.strip()
    truncated = len(note) > LINKEDIN_NOTE_MAX
    if truncated:
        note = note[: LINKEDIN_NOTE_MAX - 1].rstrip() + "…"
    field.send_keys(note)
    send = [
        b
        for b in driver.execute_script(_JS_BUTTONS, list(SEND_WORDS)) or []
        if "send" in (b.get_attribute("aria-label") or b.text).lower()
        or "envoyer" in (b.get_attribute("aria-label") or b.text).lower()
    ]
    if not send:
        raise OutreachError("Bouton « Envoyer » introuvable pour la demande de connexion.")
    click(send[-1])
    time.sleep(1.0)
    return {"channel": "linkedin", "mode": "connexion", "to": profile_url, "truncated": truncated}
