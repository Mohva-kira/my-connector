"""Ajout d'une compétence au profil LinkedIn (Selenium).

Déclenché quand l'utilisateur ajoute une compétence dans l'interface (« J'ai » sur la page Marché,
page Compétences). Ne fait qu'ajouter : rien n'est jamais retiré du profil LinkedIn.
Lève ``ProfileSyncError`` avec un message lisible quand l'ajout échoue.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By

from . import config, nlp
from .browser import human_pause

ME_URL = "https://www.linkedin.com/in/me/"
SAVE_WORDS = ("save", "enregistrer")
_BLOCKED = ("/checkpoint", "/authwall", "/login", "/uas/")
_JS_SKILL_INPUT = """
const sel = 'input[role="combobox"], input[id*="skill" i], input[placeholder*="skill" i], input[placeholder*="compétence" i]';
for (const root of [document.querySelector('[role="dialog"]'), document]) {
  if (!root) continue;
  for (const el of root.querySelectorAll(sel)) { if (el.getClientRects().length && !el.disabled) return el; }
}
return null;
"""  # noqa: E501 - script JavaScript embarqué
_JS_OPTIONS = """
const out = [];
for (const el of document.querySelectorAll('[role="listbox"] [role="option"], .basic-typeahead__selectable')) {
  if (el.getClientRects().length) out.push(el);
}
return out;
"""
_JS_SAVE = """
const words = arguments[0];
const root = document.querySelector('[role="dialog"]') || document;
for (const el of root.querySelectorAll('button')) {
  if (!el.getClientRects().length || el.disabled || el.getAttribute('aria-disabled') === 'true') continue;
  const text = ((el.getAttribute('aria-label') || '') + ' ' + (el.innerText || '')).toLowerCase().trim();
  if (words.some(w => text.startsWith(w) || text.includes(' ' + w))) return el;
}
return null;
"""
_JS_ERROR = """
for (const el of document.querySelectorAll('[role="alert"], .artdeco-inline-feedback--error')) {
  const text = (el.innerText || '').trim();
  if (el.getClientRects().length && text) return text;
}
return '';
"""


class ProfileSyncError(RuntimeError):
    """Ajout impossible ou non confirmé (message affichable tel quel)."""


def same_skill(label: str, skill: str) -> bool:
    """« REST APIs » (LinkedIn) correspond à « rest api » (outil) : même nom ou même compétence reconnue."""
    label = (label or "").strip()
    if not label or len(label) > 40:
        return False
    skill = nlp.normalize(skill)
    return nlp.normalize(label) == skill or nlp.extract_skills(label) == [skill]


def _wait(predicate, timeout: float, check: Callable[[], None]):
    end = time.time() + timeout
    while time.time() < end:
        check()
        value = predicate()
        if value:
            return value
        time.sleep(0.25)
    return None


def _open(driver, url: str, caution: config.Caution) -> None:
    driver.get(url)
    human_pause(caution, 0.6)
    if any(p in driver.current_url for p in _BLOCKED):
        raise PermissionError(f"LinkedIn a interrompu la navigation ({driver.current_url}).")


def _click(driver, el) -> None:
    try:
        el.click()
    except WebDriverException:
        driver.execute_script("arguments[0].click();", el)


def add_skill(
    driver,
    skill: str,
    caution: config.Caution,
    *,
    check: Callable[[], None] = lambda: None,
    log_fn: Callable[[str], None] = lambda m: None,
    timeout: float = 15.0,
    resolve: Callable[[str], str] = lambda u: u,
) -> dict:
    """Ajoute ``skill`` au profil connecté, sauf si une compétence équivalente y figure déjà.

    ``resolve`` traduit les URL LinkedIn (tests : pages locales).
    """
    skill = nlp.normalize(skill).strip()
    if not skill:
        raise ProfileSyncError("Compétence vide.")
    _open(driver, resolve(ME_URL), caution)
    base = driver.current_url.split("?")[0].rstrip("/")
    if "/in/" not in base or base.endswith("/in/me"):
        raise ProfileSyncError("Profil LinkedIn introuvable (page inattendue après /in/me/).")

    _open(driver, resolve(f"{base}/details/skills/"), caution)
    main = driver.find_elements(By.CSS_SELECTOR, "main")
    lines = (main[0].text if main else "").splitlines()
    existing = next((line.strip() for line in lines if same_skill(line, skill)), None)
    if existing:
        log_fn(f"« {existing} » figure déjà sur le profil LinkedIn.")
        return {"skill": skill, "linkedin_name": existing, "state": "déjà_présente"}

    _open(driver, resolve(f"{base}/edit/forms/skills/new/"), caution)
    field = _wait(lambda: driver.execute_script(_JS_SKILL_INPUT), timeout, check)
    if field is None:
        raise ProfileSyncError("Formulaire d'ajout de compétence introuvable (page inattendue).")
    field.send_keys(skill)
    human_pause(caution, 0.3)
    # La liste de suggestions donne le nom officiel (« REST APIs ») ; sans correspondance, le texte saisi est gardé.
    options = _wait(lambda: driver.execute_script(_JS_OPTIONS), 4.0, check) or []
    labels = [((o.text or "").splitlines() or [""])[0].strip() for o in options]
    exact = [i for i, label in enumerate(labels) if nlp.normalize(label) == skill]
    close = [i for i, label in enumerate(labels) if same_skill(label, skill)]
    index = (exact or close or [None])[0]
    chosen = options[index] if index is not None else None
    name = labels[index] if index is not None else skill
    if chosen is not None:
        _click(driver, chosen)
        human_pause(caution, 0.2)
    save = _wait(lambda: driver.execute_script(_JS_SAVE, list(SAVE_WORDS)), timeout, check)
    if save is None:
        raise ProfileSyncError("Bouton « Enregistrer » introuvable dans le formulaire de compétence.")
    _click(driver, save)
    closed = _wait(
        lambda: driver.execute_script(_JS_ERROR) or not driver.execute_script(_JS_SKILL_INPUT), timeout, check
    )
    error = driver.execute_script(_JS_ERROR)
    if error:
        raise ProfileSyncError(f"LinkedIn a refusé la compétence : {error}")
    if not closed:
        raise ProfileSyncError(
            "L'enregistrement n'a pas pu être confirmé : vérifie la section Compétences de ton profil LinkedIn."
        )
    log_fn(f"Compétence « {name} » ajoutée au profil LinkedIn.")
    return {"skill": skill, "linkedin_name": name, "state": "ajoutée"}
