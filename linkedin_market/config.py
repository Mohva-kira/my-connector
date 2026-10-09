"""Configuration : chemins, identifiants, niveaux de prudence.

Tout est piloté par variables d'environnement (fichier ``.env`` à la racine
du projet) pour fonctionner à l'identique sous Linux et Windows.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv est optionnel
    load_dotenv = None

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")

# Données personnelles (cookies, scraping, labels) : jamais versionnées.
DATA_DIR = Path(os.environ.get("LINKEDIN_MARKET_DATA_DIR", PROJECT_ROOT / "linkedin_market_data"))
SESSION_FILE = DATA_DIR / "session.json"
HISTORY_FILE = DATA_DIR / "history.jsonl"
LABELS_FILE = DATA_DIR / "labels.json"
MODEL_FILE = DATA_DIR / "matching_model.json"
REPORTS_DIR = DATA_DIR / "reports"
APPLICATIONS_DIR = DATA_DIR / "applications"
APPLICATIONS_FILE = DATA_DIR / "applications.json"
# Dernier profil lu sur LinkedIn : réutilisé quand on relance avec --no-scrape.
LINKEDIN_PROFILE_CACHE = DATA_DIR / "linkedin_profile.txt"
# Interface web : réglages, surcharges de compétences, journal d'activité, tâches.
SETTINGS_FILE = DATA_DIR / "settings.json"
PROFILE_SETTINGS_FILE = DATA_DIR / "profile_settings.json"
ACTIVITY_FILE = DATA_DIR / "activity.jsonl"
TASKS_FILE = DATA_DIR / "tasks.jsonl"
CV_PDF_NAME = "cv.pdf"

# Score à partir duquel une offre non annotée est proposée pour candidater.
APPLY_MIN_SCORE = 70.0
# Nombre maximal d'offres affichées à l'étape candidature (les annotées d'abord).
APPLY_MAX_LIST = 10

SESSION_TTL_HOURS = 72
TREND_WINDOW_DAYS = 30

# Fichiers de profil recherchés à la racine du projet, par ordre de priorité.
PROFILE_FILE_CANDIDATES = (
    "profil.json",
    "profile.json",
    "profil.md",
    "profile.md",
    "profil.txt",
    "profile.txt",
    "cv.pdf",
    "CV.pdf",
    "profil.pdf",
)

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"


@dataclass(frozen=True)
class Caution:
    """Volume et rythme du scraping, à ajuster selon les signes de blocage."""

    name: str
    max_items: int  # par source (offres, posts, profils)
    min_delay: float  # pause aléatoire entre deux pages (secondes)
    max_delay: float


CAUTION_LEVELS = {
    "prudent": Caution("prudent", 15, 6.0, 14.0),
    "normal": Caution("normal", 30, 4.0, 9.0),
    "agressif": Caution("agressif", 60, 2.5, 6.0),
}
DEFAULT_CAUTION = "prudent"  # E-3


def linkedin_credentials() -> tuple[str | None, str | None]:
    return os.environ.get("LINKEDIN_EMAIL"), os.environ.get("LINKEDIN_PASSWORD")


def browser_name() -> str:
    """``auto`` (Chrome puis Firefox), ``chrome`` ou ``firefox``."""
    return (os.environ.get("LINKEDIN_BROWSER") or "auto").strip().lower()


def browser_binary() -> str | None:
    return os.environ.get("LINKEDIN_BROWSER_BINARY") or None


def headless() -> bool:
    return os.environ.get("LINKEDIN_HEADLESS", "0").strip() in ("1", "true", "yes")


INTERPRETERS = ("claude", "local")


def interpreter_name() -> str:
    """``claude`` (défaut) ou ``local`` (modèle entraîné).

    ``none`` (ancienne valeur, E-7) est lu comme ``local`` : le mode local produit toujours une interprétation
    (I-11).
    """
    name = (os.environ.get("LINKEDIN_INTERPRETER") or "claude").strip().lower()
    if name == "none":
        log.warning(
            "LINKEDIN_INTERPRETER=none n'est plus accepté : interprète local utilisé. "
            "Remplace-le par « local » ou « claude » dans le fichier .env."
        )
        return "local"
    return name


def anthropic_model() -> str:
    return (os.environ.get("ANTHROPIC_MODEL") or "").strip() or DEFAULT_ANTHROPIC_MODEL


def smtp_config() -> dict | None:
    """Réglages SMTP depuis l'environnement (mêmes noms que email_analyzer) ; None si incomplets."""
    host = (os.environ.get("SMTP_HOST") or "").strip()
    sender = (os.environ.get("SMTP_FROM") or os.environ.get("SMTP_USER") or "").strip()
    if not host or not sender:
        return None
    security = (os.environ.get("SMTP_SECURITY") or "starttls").strip().lower()
    default_port = {"ssl": 465, "starttls": 587}.get(security, 25)
    return {
        "host": host,
        "port": int(os.environ.get("SMTP_PORT") or default_port),
        "security": security,
        "user": os.environ.get("SMTP_USER") or "",
        "password": os.environ.get("SMTP_PASSWORD") or "",
        "from": sender,
    }


SMTP_LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")
SMTP_INSECURE_REMOTE = (
    "SMTP_SECURITY=none n'est permis que vers un serveur SMTP local (127.0.0.1) : "
    "choisis starttls ou ssl dans le fichier .env."
)


def smtp_security_error(smtp: dict) -> str | None:
    """Message d'erreur si la connexion SMTP serait en clair vers un serveur distant (E-20, RGPD art. 32).

    ``none`` (connexion sans chiffrement) n'est admise que vers un relais local (``127.0.0.1``, ``localhost``,
    ``::1``) ; une valeur inconnue est refusée avec un message explicite.
    """
    if smtp["security"] in ("ssl", "starttls"):
        return None
    if smtp["security"] != "none":
        return (
            f"SMTP_SECURITY={smtp['security']} n'est pas reconnu : choisis starttls ou ssl dans le fichier .env "
            "(none seulement pour un serveur SMTP local, 127.0.0.1)."
        )
    host = str(smtp["host"]).strip().lower().strip("[]")
    return None if host in SMTP_LOCAL_HOSTS else SMTP_INSECURE_REMOTE


DATABASE_URL_VAR = "LM_DATABASE_URL"


class DatabaseConfigError(RuntimeError):
    """Connexion PostgreSQL non configurée (message lisible)."""


def database_url() -> str:
    """URL SQLAlchemy de la base PostgreSQL 18 (``LM_DATABASE_URL``, jamais ``DATABASE_URL`` d'email_analyzer).

    ``postgresql://`` et ``postgres://`` sont acceptés et convertis vers le driver psycopg 3.
    """
    url = (os.environ.get(DATABASE_URL_VAR) or "").strip()
    if not url:
        raise DatabaseConfigError(
            f"{DATABASE_URL_VAR} n'est pas définie : ajoute dans le .env à la racine du dépôt une ligne "
            f"{DATABASE_URL_VAR}=postgresql+psycopg://linkedin_market:<mot de passe>@127.0.0.1:5432/linkedin_market "
            "(PostgreSQL 18 : paquet système ou "
            "« docker compose --env-file .env -f linkedin_market/compose.yaml up -d db »)."
        )
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url.removeprefix(prefix)
    return url


def ensure_data_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    APPLICATIONS_DIR.mkdir(parents=True, exist_ok=True)
