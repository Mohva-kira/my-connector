"""Format ATS du CV : titres de sections canoniques (FR/EN), en-tête de contact, dates, contrôle de conformité.

Les deux générateurs (Claude et local) produisent le même schéma Markdown :

    # Prénom NOM
    **Titre aligné sur le poste**
    Ville · email · téléphone · linkedin.com/in/…

    ## Profil / ## Compétences / ## Expérience professionnelle / ## Formation / ## Certifications /
    ## Langues / ## Disponibilité

L'en-tête de contact est posé par le code (``apply_identity``), jamais par Claude : il ne reçoit aucune
coordonnée de plus et l'en-tête reste exact. Les rendus PDF et DOCX lisent ce Markdown via ``blocks``.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass

HEADINGS: dict[str, dict[str, str]] = {
    "fr": {
        "profile": "Profil",
        "skills": "Compétences",
        "experience": "Expérience professionnelle",
        "education": "Formation",
        "certifications": "Certifications",
        "languages": "Langues",
        "projects": "Projets",
        "publications": "Publications",
        "volunteering": "Bénévolat",
        "availability": "Disponibilité",
    },
    "en": {
        "profile": "Summary",
        "skills": "Skills",
        "experience": "Professional Experience",
        "education": "Education",
        "certifications": "Certifications",
        "languages": "Languages",
        "projects": "Projects",
        "publications": "Publications",
        "volunteering": "Volunteering",
        "availability": "Availability",
    },
}
REQUIRED = ("profile", "skills", "experience")
MAX_WORDS = 1000  # au-delà, le CV dépasse en pratique deux pages

_KNOWN = {name.casefold(): key for lang in HEADINGS.values() for key, name in lang.items()}
_EN_WORDS = re.compile(r"\b(the|and|with|you|our|will|experience|team|skills|we)\b", re.IGNORECASE)
_FR_WORDS = re.compile(r"\b(le|la|les|et|des|vous|nous|avec|une|pour|expérience|équipe)\b", re.IGNORECASE)
_MONTHS = {
    "janv": 1, "jan": 1, "févr": 2, "fév": 2, "feb": 2, "mars": 3, "mar": 3, "avr": 4, "apr": 4, "mai": 5,
    "may": 5, "juin": 6, "jun": 6, "juil": 7, "jul": 7, "août": 8, "aug": 8, "sept": 9, "sep": 9, "oct": 10,
    "nov": 11, "déc": 12, "dec": 12,
}  # fmt: skip
_DATE = re.compile(r"(?:(?P<month>[a-zéû]+)\.?\s+)?(?P<year>\d{4})", re.IGNORECASE)
_PRESENT = {"présent": "Présent", "present": "Present", "aujourd'hui": "Présent"}
_LINKEDIN = re.compile(r"^https://([a-z]{2,3}\.)?linkedin\.com/in/[^/?#\s]+/?$")
_CONTACT = re.compile(r"[^@\s,;<>()\[\]]+@[^@\s,;<>()\[\]]+\.\w+|\+?\d[\d .-]{7,}\d")


@dataclass(frozen=True)
class Identity:
    """Coordonnées de l'utilisateur saisies dans Paramètres (vides = omises)."""

    full_name: str = ""
    email: str = ""
    phone: str = ""
    linkedin_url: str = ""
    location: str = ""


def identity_from(prefs: dict) -> Identity:
    return Identity(
        full_name=prefs.get("full_name", ""),
        email=prefs.get("contact_email", ""),
        phone=prefs.get("contact_phone", ""),
        linkedin_url=prefs.get("linkedin_url", ""),
        location=prefs.get("home_location", ""),
    )


def is_linkedin_url(value: str) -> bool:
    return bool(_LINKEDIN.match(value))


def language_of(text: str) -> str:
    """``en`` si le texte compte nettement plus de mots outils anglais que français, sinon ``fr``."""
    english, french = len(_EN_WORDS.findall(text)), len(_FR_WORDS.findall(text))
    return "en" if english > french * 1.5 else "fr"


def heading(key: str, lang: str) -> str:
    return HEADINGS.get(lang, HEADINGS["fr"])[key]


def normalize_period(start: str, end: str) -> str:
    """« oct. 2021 », « Présent » -> « 10/2021 – Présent » ; une année seule reste une année."""

    def one(value: str) -> str:
        value = value.strip()
        if value.casefold() in _PRESENT:
            return _PRESENT[value.casefold()]
        match = _DATE.fullmatch(value)
        if not match:
            return value
        month = _MONTHS.get((match.group("month") or "").casefold().rstrip("."))
        return f"{month:02d}/{match.group('year')}" if month else match.group("year")

    return f"{one(start)} – {one(end)}"


def contact_line(identity: Identity) -> str:
    """Ligne de contact ; l'URL LinkedIn reste lisible en clair (les ATS lisent le texte, pas le lien)."""
    url = identity.linkedin_url.rstrip("/")
    shown = re.sub(r"^https://(www\.)?", "", url)
    parts = [identity.location, identity.email, identity.phone, f"[{shown}]({url})" if url else ""]
    return " · ".join(p for p in parts if p)


def apply_identity(markdown: str, identity: Identity) -> str:
    """Remplace l'en-tête (tout ce qui précède la première section) par nom, titre et coordonnées.

    Le titre est celui du premier ``#`` produit par le générateur ; les autres lignes d'en-tête sont conservées.
    """
    lines = markdown.strip().splitlines()
    first_section = next((i for i, line in enumerate(lines) if line.startswith("## ")), len(lines))
    head, body = lines[:first_section], lines[first_section:]
    title = next((line[2:].strip() for line in head if line.startswith("# ")), "")
    others = [line for line in head if line.strip() and not line.startswith("# ")]
    if identity.full_name:
        top = [f"# {identity.full_name}", *([f"**{title}**"] if title else [])]
    else:
        top = [f"# {title}"] if title else []
    contact = contact_line(identity)
    return "\n".join([*top, *([contact] if contact else []), *others, "", *body]).strip() + "\n"


def validate_cv(markdown: str) -> list[str]:
    """Écarts au format ATS, en phrases actionnables ; liste vide = conforme."""
    issues: list[str] = []
    lines = markdown.splitlines()
    first_section = next((i for i, line in enumerate(lines) if line.startswith("## ")), len(lines))
    head = "\n".join(lines[:first_section])
    if not any(line.startswith("# ") for line in lines[:first_section]):
        issues.append("Nom absent en tête du CV : renseigne-le dans Paramètres → Coordonnées.")
    if not _CONTACT.search(head):
        issues.append("Coordonnées absentes de l'en-tête : renseigne email ou téléphone dans Paramètres.")
    found = set()
    for line in lines:
        if line.startswith("## "):
            name = plain(line[3:]).strip()
            key = _KNOWN.get(name.casefold())
            if key:
                found.add(key)
            else:
                issues.append(f"Titre de section non standard : « {name} » (les ATS risquent de l'ignorer).")
    for key in REQUIRED:
        if key not in found:
            issues.append(f"Section manquante : « {heading(key, 'fr')} » / « {heading(key, 'en')} ».")
    if any(line.lstrip().startswith("|") for line in lines):
        issues.append("Tableau détecté : les ATS lisent mal les tableaux, préfère des puces.")
    if "![" in markdown:
        issues.append("Image détectée : les ATS ne lisent pas les images.")
    words = len(re.findall(r"\w+", markdown))
    if words > MAX_WORDS:
        issues.append(f"CV long ({words} mots) : vise deux pages au plus (environ {MAX_WORDS} mots).")
    return issues


# ------------------------------------------------------------ lecture du Markdown (rendus PDF et DOCX)
@dataclass(frozen=True)
class Block:
    kind: str  # heading | bullet | text | blank
    text: str = ""
    level: int = 0


def blocks(markdown: str) -> Iterator[Block]:
    for raw in markdown.splitlines():
        line = raw.rstrip()
        if not line.strip():
            yield Block("blank")
        elif match := re.match(r"^(#{1,6})\s+(.*)$", line):
            yield Block("heading", match.group(2).strip(), len(match.group(1)))
        elif match := re.match(r"^\s*[-*+]\s+(.*)$", line):
            yield Block("bullet", match.group(1))
        elif match := re.match(r"^\s*(\d+)[.)]\s+(.*)$", line):
            yield Block("bullet", f"{match.group(1)}. {match.group(2)}")
        else:
            yield Block("text", line.strip())


_SPANS = re.compile(r"\*\*(?P<b>.+?)\*\*|\[(?P<t>[^\]]+)\]\((?P<u>[^)\s]+)\)|(?<![\w*])[*_](?P<i>[^*_]+?)[*_](?![\w*])")


def spans(text: str) -> Iterator[tuple[str, str, str]]:
    """Découpe une ligne en segments (texte, style ``""``/``B``/``I``, url) ; le code en ligne perd ses ``` ` ```."""
    text = text.replace("`", "")
    pos = 0
    for match in _SPANS.finditer(text):
        if match.start() > pos:
            yield text[pos : match.start()], "", ""
        if match.group("b") is not None:
            yield match.group("b"), "B", ""
        elif match.group("t") is not None:
            yield match.group("t"), "", match.group("u")
        else:
            yield match.group("i"), "I", ""
        pos = match.end()
    if pos < len(text):
        yield text[pos:], "", ""


def plain(text: str) -> str:
    return "".join(part for part, _, _ in spans(text))


def file_stem(full_name: str, title: str) -> str:
    """« Prénom-Nom-CV-Titre » en ASCII (nom de fichier vu par le recruteur)."""

    def ascii_words(value: str, limit: int) -> str:
        flat = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
        return "-".join(re.findall(r"[A-Za-z0-9]+", flat))[:limit].strip("-")

    parts = [ascii_words(full_name, 60), "CV", ascii_words(title, 40)]
    return "-".join(p for p in parts if p)
