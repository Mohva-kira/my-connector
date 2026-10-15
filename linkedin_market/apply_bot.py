"""Candidature « Candidature simplifiée » (Easy Apply) LinkedIn pilotée par Selenium.

Principes :
- Les sélecteurs CSS de LinkedIn changent souvent : les boutons sont repérés par leur texte / aria-label
  (français et anglais) et les champs par leur étiquette, jamais par des classes générées.
- Honnêteté des réponses : seuls l'email et le téléphone saisis dans les paramètres et les réponses déjà
  mémorisées par l'utilisateur sont remplis automatiquement. Toute autre question (années d'expérience,
  autorisation de travail, prétentions…) est posée à l'utilisateur, jamais devinée.
- Rien n'est envoyé sans clic explicite : l'outil s'arrête à l'écran de relecture et demande confirmation.
- Offre sans Easy Apply (site tiers) : aucun remplissage, l'état « externe » est renvoyé.
- Checkpoint / authwall / connexion perdue : arrêt immédiat (``BlockedError``).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from selenium.common.exceptions import WebDriverException

from . import config, nlp, settings
from .browser import human_pause

log = logging.getLogger(__name__)

MAX_STEPS = 12
STEP_TIMEOUT = 10.0  # secondes ; ajustable par instance (``step_timeout``)
SUBMIT_TIMEOUT = 20.0

EASY_APPLY_WORDS = ("easy apply", "candidature simplifiée", "candidature simplifiee")
SUBMIT_WORDS = ("submit application", "envoyer la candidature", "soumettre la candidature", "submit")
REVIEW_WORDS = (
    "review your application",
    "vérifier votre candidature",
    "verifier votre candidature",
    "review",
    "vérifier",
)
NEXT_WORDS = (
    "continue to next step",
    "passer à l'étape suivante",
    "passer a l'etape suivante",
    "next",
    "suivant",
    "continuer",
)
DISMISS_WORDS = ("dismiss", "ignorer", "fermer", "close")
DISCARD_WORDS = ("discard", "abandonner", "supprimer", "ne pas enregistrer")
SENT_WORDS = (
    "application was sent",
    "your application was sent",
    "candidature a été envoyée",
    "candidature envoyée",
    "votre candidature a bien été envoyée",
    "application submitted",
)
_ALREADY_APPLIED = re.compile(
    r"\bapplied\s+\d+\s*(?:minute|hour|day|week|month)s?\s+ago|candidature\s+envoy[ée]e\s+il\s+y\s+a|vous\s+avez\s+postul[ée]"
)
PLACEHOLDERS = ("select an option", "sélectionner une option", "selectionner une option", "choisir", "select", "choose")


class ApplyError(RuntimeError):
    """Échec lisible de la candidature (message affiché dans l'interface)."""


class BlockedError(ApplyError):
    """LinkedIn a interrompu la navigation (checkpoint, authwall, session perdue)."""


class Hooks(Protocol):
    """Ce que le bot attend de la tâche appelante (``TaskContext`` les fournit)."""

    def log(self, message: str) -> None: ...
    def check(self) -> None: ...
    def ask(
        self,
        question: str,
        field_id: str = "",
        options: list[str] | None = None,
        *,
        suggestion: str = "",
        review: str = "",
    ) -> str: ...


@dataclass
class ApplyResult:
    state: str  # postulée | externe | déjà_postulée | annulée | incertaine
    detail: str = ""
    answers: dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------- JavaScript
# LinkedIn rend la fenêtre de candidature dans un Shadow DOM ouvert (hôte ``#interop-outlet``) :
# ``document.querySelectorAll`` ne la voit pas, d'où cette recherche qui descend dans les shadow roots ouverts.
_JS_DEEP = """
const deepAll = (root, sel) => {
  const out = Array.from(root.querySelectorAll(sel));
  for (const el of root.querySelectorAll('*')) if (el.shadowRoot) out.push(...deepAll(el.shadowRoot, sel));
  return out;
};
"""

_JS_BUTTONS = (
    _JS_DEEP
    + """
const words = arguments[0];
const root = arguments[1] || document;
const out = [];
// Le bouton « Candidature simplifiée » peut être un lien vers le formulaire (…/apply/…).
for (const el of deepAll(root, 'button, [role="button"], a[href*="/apply"]')) {
  if (!el.getClientRects().length || el.disabled || el.getAttribute('aria-disabled') === 'true') continue;
  const text = ((el.getAttribute('aria-label') || '') + ' ' + (el.innerText || '')).toLowerCase().replace(/\\s+/g, ' ').trim();
  if (words.some(w => text.includes(w))) out.push(el);
}
return out;
"""  # noqa: E501 - script JavaScript embarqué
)

_JS_DIALOG = (
    _JS_DEEP
    + """
for (const el of deepAll(document, '[role="dialog"], .artdeco-modal, .jobs-easy-apply-modal')) {
  const cs = getComputedStyle(el);
  if (cs.display !== 'none' && cs.visibility !== 'hidden' && el.getClientRects().length) return el;
}
return null;
"""
)

_JS_FIND = _JS_DEEP + "return deepAll(document, '[data-lm-key=\"' + arguments[0] + '\"]')[0] || null;"

# Décrit les champs visibles d'une étape ; marque chaque élément (data-lm-key) pour le retrouver côté Selenium.
_JS_FIELDS = """
const root = arguments[0];
const clean = s => (s || '').replace(/\\s+/g, ' ').trim();
const visible = el => el.getClientRects().length > 0 || el.type === 'radio' || el.type === 'checkbox';
const labelOf = el => {
  if (el.id) { const l = root.querySelector('label[for="' + CSS.escape(el.id) + '"]'); if (l) return clean(l.innerText); }
  if (el.getAttribute('aria-label')) return clean(el.getAttribute('aria-label'));
  const wrap = el.closest('fieldset');
  if (wrap) { const lg = wrap.querySelector('legend'); if (lg) return clean(lg.innerText); }
  const group = el.closest('.fb-dash-form-element, .jobs-easy-apply-form-element, .jobs-easy-apply-form-section__grouping, div');
  if (group) { const l = group.querySelector('label, legend, span'); if (l) return clean(l.innerText); }
  return clean(el.name || el.placeholder || '');
};
const fields = [];
const radios = {};
let n = 0;
// Préfixe unique par appel : les marques d'une étape précédente restent dans le DOM.
const base = 's' + (window.__lmScan = (window.__lmScan || 0) + 1) + '_';
for (const el of root.querySelectorAll('input, select, textarea')) {
  const type = (el.type || el.tagName).toLowerCase();
  if (['hidden', 'submit', 'button', 'image', 'reset'].includes(type)) continue;
  if (!visible(el) && type !== 'file') continue;
  let required = el.required || el.getAttribute('aria-required') === 'true' || /\\*|obligatoire|required/i.test(labelOf(el));
  if (type === 'radio') {
    const group = clean(el.closest('fieldset') ? (el.closest('fieldset').querySelector('legend') || {innerText: el.name}).innerText : el.name);
    required = required || /\\*|obligatoire|required/i.test(group) || (el.closest('fieldset') || {}).getAttribute?.('aria-required') === 'true';
    const key = base + 'radio:' + (el.name || group);
    if (!radios[key]) { radios[key] = {key, kind: 'radio', label: group, required, options: [], checked: false, els: []}; fields.push(radios[key]); }
    const lab = root.querySelector('label[for="' + CSS.escape(el.id) + '"]');
    const text = clean(lab ? lab.innerText : el.value);
    el.setAttribute('data-lm-key', radios[key].key + '#' + radios[key].options.length);
    radios[key].options.push(text);
    if (el.checked) radios[key].checked = true;
    continue;
  }
  const key = base + 'f' + (n++);
  el.setAttribute('data-lm-key', key);
  const field = {key, kind: type === 'select-one' || el.tagName === 'SELECT' ? 'select' : type === 'file' ? 'file' : type === 'checkbox' ? 'checkbox' : el.tagName === 'TEXTAREA' ? 'textarea' : type,
    label: labelOf(el), required, value: el.value || '', checked: !!el.checked, options: [], selected: ''};
  if (field.kind === 'select') {
    field.options = Array.from(el.options).map(o => clean(o.text));
    field.selected = el.selectedIndex >= 0 ? clean(el.options[el.selectedIndex].text) : '';
  }
  fields.push(field);
}
return fields;
"""  # noqa: E501 - script JavaScript embarqué

# Texte d'un élément, ou de la page entière y compris le contenu des Shadow DOM ouverts.
_JS_TEXT = (
    _JS_DEEP
    + """
if (arguments[0]) return arguments[0].innerText || '';
const parts = [document.body.innerText || ''];
for (const host of deepAll(document, '*')) {
  if (host.shadowRoot) for (const child of host.shadowRoot.children) parts.push(child.innerText || '');
}
return parts.join('\\n');
"""
)


# ------------------------------------------------------------------- outils
def _norm(text: str) -> str:
    return nlp.normalize(text or "")


def _has(text: str, words: tuple[str, ...]) -> bool:
    low = (text or "").lower()
    return any(w in low for w in words)


class EasyApply:
    def __init__(self, driver, hooks: Hooks, *, cv_pdf: Path | None, caution: config.Caution, cfg: dict | None = None):
        self.driver, self.hooks, self.cv_pdf, self.caution = driver, hooks, cv_pdf, caution
        self.cfg = cfg if cfg is not None else settings.load_settings()
        self.answers: dict[str, str] = {}
        self._uploaded = False
        self.step_timeout = STEP_TIMEOUT

    # ---------------------------------------------------------- navigation
    def _blocked(self) -> bool:
        url = self.driver.current_url
        return any(part in url for part in ("/checkpoint", "/authwall", "/login", "/uas/"))

    def _buttons(self, words: tuple[str, ...], root=None) -> list:
        return self.driver.execute_script(_JS_BUTTONS, list(words), root) or []

    def _dialog(self):
        return self.driver.execute_script(_JS_DIALOG)

    def _page_text(self) -> str:
        return (self.driver.execute_script(_JS_TEXT, None) or "").lower()

    def _describe(self, element) -> str:
        """Résumé court de l'élément cliqué (balise, libellé, lien) pour diagnostiquer un échec."""
        try:
            return self.driver.execute_script(
                "const el = arguments[0];"
                "const label = (el.getAttribute('aria-label') || el.innerText || '').replace(/\\s+/g, ' ').trim();"
                "const href = el.getAttribute('href');"
                "return '<' + el.tagName.toLowerCase() + '> « ' + label.slice(0, 60) + ' »'"
                " + (href ? ' → ' + href.split('?')[0] : '');",
                element,
            )
        except WebDriverException:
            return "élément inconnu"

    def _click(self, element) -> None:
        try:
            element.click()
        except WebDriverException:
            self.driver.execute_script("arguments[0].click();", element)

    def _wait(self, predicate, timeout: float):
        end = time.time() + timeout
        while time.time() < end:
            self.hooks.check()
            value = predicate()
            if value:
                return value
            time.sleep(0.25)
        return None

    # --------------------------------------------------------------- champs
    def _auto_answer(self, label: str) -> str | None:
        norm = _norm(label)
        if any(w in norm for w in ("email", "e-mail", "courriel")) and self.cfg.get("contact_email"):
            return self.cfg["contact_email"]
        is_code = any(w in norm for w in ("country", "pays", "indicatif", "code"))
        if (
            not is_code
            and any(w in norm for w in ("phone", "telephone", "mobile", "portable"))
            and self.cfg.get("contact_phone")
        ):
            return self.cfg["contact_phone"]
        return settings.find_answer(label)

    def _answer(self, field: dict, suggestion: str = "") -> str:
        label = field["label"] or "Question sans intitulé"
        found = self._auto_answer(label)
        if found is not None:
            self.hooks.log(f"Réponse mémorisée : {label[:70]}")
            return found
        reply = self.hooks.ask(label, field["key"], field.get("options") or None, suggestion=suggestion)
        self.answers[label] = reply
        return reply

    def _find(self, key: str):
        element = self.driver.execute_script(_JS_FIND, key)
        if element is None:
            raise ApplyError(f"Champ introuvable dans la fenêtre de candidature ({key}).")
        return element

    def _fill_field(self, field: dict) -> None:
        kind = field["kind"]
        if kind == "file":
            if self.cv_pdf and not self._uploaded:
                self._find(field["key"]).send_keys(str(self.cv_pdf))
                self._uploaded = True
                self.hooks.log(f"CV téléversé : {self.cv_pdf.name}")
            return
        if kind == "checkbox":
            return  # jamais coché automatiquement (consentements, newsletters…)
        if kind == "radio":
            if field["checked"]:
                return
            if not field["required"] and self._auto_answer(field["label"]) is None:
                return
            choice = self._answer(field)
            index = self._match_option(field["options"], choice)
            if index is None:
                raise ApplyError(f"Réponse « {choice} » absente des choix de « {field['label']} ».")
            element = self._find(f"{field['key']}#{index}")
            self.driver.execute_script("arguments[0].click();", element)
            return
        if kind == "select":
            chosen = _norm(field.get("selected", ""))
            if chosen and not any(chosen.startswith(p) for p in PLACEHOLDERS):
                return  # déjà renseigné (souvent pré-rempli par LinkedIn)
            if not field["required"] and self._auto_answer(field["label"]) is None:
                return
            choice = self._answer(field)
            index = self._match_option(field["options"], choice)
            if index is None:
                raise ApplyError(f"Réponse « {choice} » absente des choix de « {field['label']} ».")
            element = self._find(field["key"])
            self.driver.execute_script(
                "arguments[0].selectedIndex = arguments[1];"
                "arguments[0].dispatchEvent(new Event('input',{bubbles:true}));"
                "arguments[0].dispatchEvent(new Event('change',{bubbles:true}));",
                element,
                index,
            )
            return
        # champ texte, nombre, email, téléphone, zone de texte
        if (field["value"] or "").strip():
            return
        if not field["required"] and self._auto_answer(field["label"]) is None:
            return
        suggestion = ""
        if kind == "number" and any(w in _norm(field["label"]) for w in ("year", "an", "experience", "annee")):
            suggestion = str(self.cfg.get("years_hint") or "")
        value = self._answer(field, suggestion)
        self._type(self._find(field["key"]), value)

    def _type(self, element, value: str) -> None:
        """Saisie réaliste ; repli sur l'affectation JS (avec événements, compatible React) si l'élément résiste."""
        self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
        try:
            element.clear()
            element.send_keys(value)
        except WebDriverException:
            self.driver.execute_script(
                "const el = arguments[0];"
                "const proto = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;"
                "Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, arguments[1]);"
                "el.dispatchEvent(new Event('input', {bubbles: true}));"
                "el.dispatchEvent(new Event('change', {bubbles: true}));",
                element,
                value,
            )

    @staticmethod
    def _match_option(options: list[str], choice: str) -> int | None:
        wanted = _norm(choice)
        for i, option in enumerate(options):
            if _norm(option) == wanted:
                return i
        for i, option in enumerate(options):
            if wanted and (wanted in _norm(option) or _norm(option) in wanted) and _norm(option):
                return i
        return None

    def _fill_step(self, dialog) -> None:
        fields = self.driver.execute_script(_JS_FIELDS, dialog) or []
        for described in fields:
            self.hooks.check()
            self._fill_field(described)

    def _errors(self, dialog) -> list[str]:
        script = """
        const out = [];
        for (const el of arguments[0].querySelectorAll('.artdeco-inline-feedback--error, [role="alert"]')) {
          const t = (el.innerText || '').replace(/\\s+/g, ' ').trim(); if (t) out.push(t);
        }
        return out;"""
        return list(dict.fromkeys(self.driver.execute_script(script, dialog) or []))

    # ------------------------------------------------------------- dialogue
    def _close_dialog(self) -> None:
        try:
            dialog = self._dialog()
            if dialog is None:
                return
            for button in self._buttons(DISMISS_WORDS, dialog)[:1]:
                self._click(button)
            time.sleep(0.8)
            for button in self._buttons(DISCARD_WORDS)[:1]:
                self._click(button)
        except WebDriverException:
            pass

    def _uncheck_follow(self, dialog) -> None:
        script = """
        for (const box of arguments[0].querySelectorAll('input[type="checkbox"]')) {
          const lab = arguments[0].querySelector('label[for="' + CSS.escape(box.id) + '"]');
          const text = ((lab ? lab.innerText : '') + ' ' + (box.getAttribute('aria-label') || '')).toLowerCase();
          if (box.checked && (text.includes('follow') || text.includes('suivre'))) { (lab || box).click(); return true; }
        }
        return false;"""  # noqa: E501 - script JavaScript embarqué
        if self.driver.execute_script(script, dialog):
            self.hooks.log("« Suivre l'entreprise » décoché.")

    # ----------------------------------------------------------------- flux
    def run(self, job: dict) -> ApplyResult:
        driver = self.driver
        driver.get(job["url"])
        human_pause(self.caution, 0.5)
        if self._blocked():
            raise BlockedError(f"LinkedIn a interrompu la navigation ({driver.current_url}).")

        buttons = self._buttons(EASY_APPLY_WORDS)
        if not buttons:
            if _ALREADY_APPLIED.search(self._page_text()[:3000]):
                return ApplyResult("déjà_postulée", "LinkedIn indique que tu as déjà postulé.")
            return ApplyResult("externe", "Pas de candidature simplifiée : postule sur le site de l'entreprise.")

        self.hooks.log("Ouverture de la candidature simplifiée…")
        clicked = self._describe(buttons[0])
        self._click(buttons[0])
        dialog = self._wait(self._dialog, self.step_timeout)
        if dialog is None:
            if self._blocked():
                raise BlockedError(f"LinkedIn a interrompu la navigation ({driver.current_url}).")
            raise ApplyError(
                "La fenêtre de candidature ne s'est pas ouverte (page inattendue) : "
                f"clic sur {clicked}, page actuelle {driver.current_url.split('?')[0]}."
            )

        try:
            return self._steps(job)
        except BaseException:
            self._close_dialog()
            raise

    def _steps(self, job: dict) -> ApplyResult:
        previous_errors: list[str] = []
        for step in range(1, MAX_STEPS + 1):
            self.hooks.check()
            if self._blocked():
                raise BlockedError("LinkedIn a interrompu la navigation pendant la candidature.")
            dialog = self._dialog()
            if dialog is None:
                raise ApplyError("La fenêtre de candidature s'est fermée de façon inattendue.")
            self._fill_step(dialog)

            submit = self._buttons(SUBMIT_WORDS, dialog)
            if submit:
                self._uncheck_follow(dialog)
                review = (self.driver.execute_script(_JS_TEXT, dialog) or "").strip()
                reply = self.hooks.ask(
                    f"Soumettre la candidature « {job.get('title', '')} » ?",
                    "confirm_submit",
                    ["Soumettre", "Annuler"],
                    review=review[:3500],
                )
                if reply != "Soumettre":
                    self._close_dialog()
                    return ApplyResult("annulée", "Candidature annulée avant l'envoi.", self.answers)
                self._click(submit[0])
                return self._confirm_sent()

            advance = self._buttons(REVIEW_WORDS, dialog) or self._buttons(NEXT_WORDS, dialog)
            if not advance:
                raise ApplyError("Aucun bouton « Suivant / Vérifier / Envoyer » trouvé : la page a peut-être changé.")
            before = self.driver.execute_script(_JS_TEXT, dialog)
            self.hooks.log(f"Étape {step} remplie.")
            self._click(advance[0])
            human_pause(self.caution, 0.15)

            def changed(before=before):  # lié à l'étape courante (la boucle réaffecte ``before``)
                current = self._dialog()
                return current is not None and self.driver.execute_script(_JS_TEXT, current) != before

            if not self._wait(changed, self.step_timeout):
                errors = self._errors(self._dialog() or dialog)
                if errors and errors == previous_errors:
                    raise ApplyError("Champs invalides : " + " | ".join(errors)[:300])
                previous_errors = errors
                # Un champ obligatoire a pu être manqué : on repasse sur l'étape
                # (le remplissage ne touche que les champs vides).
        raise ApplyError(f"Plus de {MAX_STEPS} étapes : candidature interrompue.")

    def _confirm_sent(self) -> ApplyResult:
        sent = self._wait(lambda: _has(self._page_text(), SENT_WORDS), SUBMIT_TIMEOUT)
        for button in self._buttons(("done", "terminé", "termine", "dismiss", "ignorer"))[:1]:
            self._click(button)
        if sent:
            return ApplyResult("postulée", "Candidature envoyée.", self.answers)
        return ApplyResult(
            "incertaine", "L'envoi n'a pas pu être confirmé : vérifie sur LinkedIn avant de réessayer.", self.answers
        )
