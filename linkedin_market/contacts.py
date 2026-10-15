"""Recruteurs d'une offre et leur email, tels que LinkedIn les affiche à l'utilisateur connecté.

- Les personnes viennent du bloc « Rencontrez l'équipe de recrutement » de la page de l'offre.
- L'email vient de la fenêtre « Coordonnées » du profil, uniquement quand LinkedIn l'affiche (selon le degré
  de connexion). Rien n'est deviné ni reconstitué à partir d'un format d'adresse.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

from selenium.common.exceptions import TimeoutException, WebDriverException

from . import config, logmask
from .browser import human_pause

log = logging.getLogger(__name__)

MAX_CONTACTS = 3
TEAM_HEADINGS = (
    "meet the hiring team",
    "rencontrez l'équipe de recrutement",
    "rencontrez l’équipe de recrutement",
    "people you can reach out to",
    "personnes que vous pouvez contacter",
)
_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_SLUG = re.compile(r"linkedin\.com/in/([^/?#]+)")
_DEGREE = re.compile(r"\s*[•·]\s*(?:1er|1st|2e|2nd|3e|3rd|\d+\w*)\+?.*$", re.IGNORECASE)

# Liens de profil situés sous le titre du bloc « équipe de recrutement » (ou, à défaut, dans <main>).
_JS_TEAM = """
const headings = arguments[0];
const norm = s => (s || '').toLowerCase().replace(/\\s+/g, ' ').trim();
let scope = null;
for (const el of document.querySelectorAll('h1, h2, h3, h4, span, div, p')) {
  if (el.children.length > 2) continue;
  const t = norm(el.innerText);
  if (t && t.length < 80 && headings.some(h => t.includes(h))) { scope = el.closest('section, div.job-details-people-who-can-help, .jobs-poster, div') ; break; }
}
if (!scope) return [];
// remonte d'un niveau tant que le conteneur ne contient aucun profil
let guard = 0;
while (scope && !scope.querySelector('a[href*="/in/"]') && guard++ < 4) scope = scope.parentElement;
if (!scope) return [];
const out = [], seen = new Set();
for (const a of scope.querySelectorAll('a[href*="/in/"]')) {
  const m = a.href.match(/linkedin\\.com\\/in\\/([^/?#]+)/);
  if (!m || seen.has(m[1])) continue;
  seen.add(m[1]);
  const card = a.closest('li, div.display-flex, div') || a;
  out.push({slug: m[1], text: (card.innerText || a.innerText || '').trim()});
}
return out;
"""  # noqa: E501

_JS_OVERLAY = """
const dialog = document.querySelector('[role="dialog"], .artdeco-modal') || document.querySelector('main') || document.body;
const mails = Array.from(dialog.querySelectorAll('a[href^="mailto:"]')).map(a => a.getAttribute('href').replace(/^mailto:/i, '').split('?')[0]);
return {mails, text: dialog.innerText || ''};
"""  # noqa: E501


def parse_team(raw: list[dict]) -> list[dict]:
    """[{slug, text}] -> contacts {full_name, profile_url, email, headline, source} (C-4, C-5)."""
    contacts = []
    for row in raw[:MAX_CONTACTS]:
        lines = [line.strip() for line in (row.get("text") or "").splitlines() if line.strip()]
        name = _DEGREE.sub("", lines[0]).strip() if lines else ""
        headline = next(
            (
                line
                for line in lines[1:]
                if not re.match(r"^[•·]?\s*(1er|1st|2e|2nd|3e|3rd)\b", line, re.IGNORECASE)
                and "message" not in line.lower()
                and len(line) > 3
            ),
            "",
        )
        if not name or len(name) > 80:
            name = row["slug"].replace("-", " ").title()
        contacts.append(
            {
                "full_name": name,
                "profile_url": f"https://www.linkedin.com/in/{row['slug']}/",
                "email": None,
                "headline": headline[:160],
                "source": "linkedin_equipe_recrutement",
            }
        )
    return contacts


def extract_email(mails: list[str], text: str) -> str | None:
    for mail in mails:
        if _EMAIL.fullmatch(mail.strip()):
            return mail.strip()
    found = _EMAIL.search(text or "")
    return found.group(0) if found else None


def overlay_url(profile_url: str) -> str:
    return profile_url.rstrip("/") + "/overlay/contact-info/"


def find_hiring_team(
    driver,
    job_url: str,
    caution: config.Caution,
    *,
    check: Callable[[], None] = lambda: None,
    log_fn: Callable[[str], None] = log.info,
    to_overlay: Callable[[str], str] = overlay_url,
) -> list[dict]:
    """Recruteurs de l'offre avec leur email quand LinkedIn l'affiche.

    Le journal (``log_fn``) ne reçoit que des initiales et des adresses masquées (E-12) ; les valeurs complètes
    sont renvoyées pour l'historique de la candidature.
    """
    try:
        driver.get(job_url)
    except TimeoutException:
        raise RuntimeError(
            "LinkedIn met trop de temps à répondre (60 s) : vérifie ta connexion "
            "et la fenêtre du navigateur, puis réessaie."
        ) from None
    human_pause(caution, 0.6)
    if any(p in driver.current_url for p in ("/checkpoint", "/authwall", "/login", "/uas/")):
        raise PermissionError(f"LinkedIn a interrompu la navigation ({driver.current_url}).")
    try:
        raw = driver.execute_script(_JS_TEAM, list(TEAM_HEADINGS)) or []
    except WebDriverException:
        raw = []
    contacts = parse_team(raw)
    if not contacts:
        log_fn("Aucune équipe de recrutement affichée sur cette offre.")
        return []
    log_fn(f"{len(contacts)} contact(s) de recrutement trouvé(s).")
    for contact in contacts:
        check()
        try:
            driver.get(to_overlay(contact["profile_url"]))
            human_pause(caution, 0.6)
            if any(p in driver.current_url for p in ("/checkpoint", "/authwall", "/login", "/uas/")):
                raise PermissionError(f"LinkedIn a interrompu la navigation ({driver.current_url}).")
            data = driver.execute_script(_JS_OVERLAY) or {}
            contact["email"] = extract_email(data.get("mails", []), data.get("text", ""))
        except WebDriverException as exc:
            log_fn(f"Coordonnées illisibles pour {logmask.mask_name(contact['full_name'])} : {exc.__class__.__name__}")
        email = logmask.mask_email(contact["email"]) if contact["email"] else "email non affiché par LinkedIn"
        log_fn(f"{logmask.mask_name(contact['full_name'])} : {email}")
    return contacts


def merge_contacts(existing: list[dict], found: list[dict]) -> list[dict]:
    """Ajoute les contacts trouvés sans écraser un email saisi à la main."""
    merged = {c.get("profile_url") or f"manual:{i}": dict(c) for i, c in enumerate(existing)}
    for contact in found:
        key = contact["profile_url"]
        if key in merged:
            old = merged[key]
            # Un email saisi à la main reste prioritaire ; sinon le plus récent affiché par LinkedIn.
            email = (
                old.get("email")
                if old.get("source") == "manuel" and old.get("email")
                else contact["email"] or old.get("email")
            )
            merged[key] = {**old, **{k: v for k, v in contact.items() if v and k != "source"}, "email": email}
        else:
            merged[key] = contact
    return list(merged.values())
