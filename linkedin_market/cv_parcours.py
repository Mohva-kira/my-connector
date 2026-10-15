"""Texte brut du profil -> sections ATS en Markdown : une expérience par bloc (titre / période / description),
puis Formation, Certifications, Langues… repérées en fin de texte. Dates au format ``MM/AAAA`` (``cv_format``).

Le texte LinkedIn arrive soit avec ses retours à la ligne, soit « collé » (titres et paragraphes accolés
sans espace). Chaque expérience est repérée par sa période (« oct. 2021 - Présent · 5 ans 1 mois ») ; l'en-tête
(poste, entreprise) est le texte qui précède cette période. Sans période reconnue, le texte est seulement
découpé en paragraphes : rien n'est jamais réécrit ni supprimé dans une expérience.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .cv_format import normalize_period

_MONTH = r"(?:janv|févr|mars|avr|mai|juin|juil|août|sept|oct|nov|déc)\.?"
_END = rf"(?:(?:{_MONTH}\s+)?\d{{4}}|aujourd'hui|présent|present)"
_PERIOD = re.compile(
    rf"(?P<start>(?:{_MONTH}\s+)?\d{{4}})\s*[-–]\s*(?P<end>{_END})"
    r"(?:\s*·\s*(?P<length>(?:\d+\s+(?:ans?|mois)\s*)+))?",
    re.IGNORECASE,
)
# Rubriques LinkedIn de fin de profil -> clé de ``cv_format.HEADINGS`` ; None = non reprise dans le CV
# (« Compétences » est reconstruite depuis le profil, « Recommandations » est un texte de tiers).
_TAIL_KEYS = {
    "Formation": "education",
    "Licences et certifications": "certifications",
    "Langues": "languages",
    "Projets": "projects",
    "Publications": "publications",
    "Bénévolat": "volunteering",
    "Compétences": None,
    "Recommandations": None,
}
_TAIL_NAMES = "|".join(_TAIL_KEYS)
# Rubrique collée au texte précédent (« …Paris.Formation… ») ou seule sur sa ligne.
_TAIL_HEADING = re.compile(rf"(?<=[a-zà-ÿ.)!?:\d])(?:{_TAIL_NAMES})(?=[A-ZÀ-Ý\s]|$)|(?<=\n)(?:{_TAIL_NAMES})(?=\n|$)")
# Forme sans ambiguïté (collée à un mot en majuscule ou seule sur sa ligne) : cherchée dès la première expérience,
# car la Formation porte elle aussi des périodes.
_TAIL_STRICT = re.compile(rf"(?<=[a-zà-ÿ.)!?:\d])(?:{_TAIL_NAMES})(?=[A-ZÀ-Ý])|(?<=\n)(?:{_TAIL_NAMES})(?=\n|$)")
_TAIL_SPLIT = re.compile(rf"^(?:{_TAIL_NAMES})|{_TAIL_HEADING.pattern}")
_GLUE = re.compile(r"(?<=[a-zà-ÿ0-9.)!?:])(?=[A-ZÀ-Ý])")
_SENTENCE_END = re.compile(r"(?<=[.!?)])(?=[A-ZÀ-Ý])")
_HEADER_MAX = 120


def _split_header(header: str) -> tuple[str, str]:
    """(poste, entreprise · type) : sur les retours à la ligne, sinon à la dernière jointure collée."""
    lines = [ln.strip() for ln in header.splitlines() if ln.strip()]
    if len(lines) >= 2:
        return lines[-2], lines[-1]
    one = lines[0] if lines else ""
    cuts = [m.start() for m in _GLUE.finditer(one)]
    if not cuts:
        return one, ""
    return one[: cuts[0]].strip(), one[cuts[0] :].strip()


def _cut_header(before: str) -> tuple[str, str]:
    """Sépare la description de l'expérience précédente et l'en-tête de la suivante."""
    lines = before.rstrip().splitlines()
    if len(lines) >= 3 or (len(lines) >= 2 and before.count("\n") >= 2):
        return "\n".join(lines[:-2]), "\n".join(lines[-2:])
    ends = [m.start() for m in _SENTENCE_END.finditer(before)]
    if ends and len(before) - ends[-1] <= _HEADER_MAX:
        return before[: ends[-1]], before[ends[-1] :]
    if len(before) <= _HEADER_MAX:
        return "", before
    return before[:-_HEADER_MAX], before[-_HEADER_MAX:]


def _paragraphs(text: str) -> str:
    return "\n\n".join(ln.strip() for ln in text.splitlines() if ln.strip())


@dataclass
class Parcours:
    experience: str
    sections: list[tuple[str, str]] = field(default_factory=list)  # (clé de cv_format.HEADINGS, Markdown)


def _period(match: re.Match) -> str:
    return normalize_period(match.group("start"), match.group("end"))


def _tail_sections(tail: str) -> list[tuple[str, str]]:
    marks = list(_TAIL_SPLIT.finditer(tail))
    sections = []
    for i, mark in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(tail)
        key = _TAIL_KEYS[mark.group(0)]
        body = _paragraphs(_PERIOD.sub(_period, tail[mark.end() : end]))
        if key and body:
            sections.append((key, body))
    return sections


def format_parcours(text: str) -> Parcours:
    """Expériences en Markdown (sans titre de section) et rubriques de fin de profil reconnues."""
    text = text.strip()
    periods = list(_PERIOD.finditer(text))
    if not periods:
        return Parcours(_paragraphs(text))

    tail = ""
    heading = _TAIL_STRICT.search(text, periods[0].end()) or _TAIL_HEADING.search(text, periods[-1].end())
    if heading:
        text, tail = text[: heading.start()], text[heading.start() :]
    periods = list(_PERIOD.finditer(text))

    entries: list[tuple[str, str, str, str]] = []  # poste, société, période, description
    header_start = _cut_header(text[: periods[0].start()])[1]
    for i, match in enumerate(periods):
        end = periods[i + 1].start() if i + 1 < len(periods) else len(text)
        body, next_header = _cut_header(text[match.end() : end]) if i + 1 < len(periods) else (text[match.end() :], "")
        role, company = _split_header(header_start)
        entries.append((role, company, _period(match), body))
        header_start = next_header

    blocks = []
    for role, company, period, body in entries:
        title = f"### {role or 'Expérience'}" + (f" — {company}" if company else "")
        desc = re.sub(r"\s*\n\s*", "\n\n", body.strip())
        blocks.append("\n\n".join(part for part in (title, f"*{period}*", desc) if part))
    return Parcours("\n\n".join(blocks), _tail_sections(tail))
