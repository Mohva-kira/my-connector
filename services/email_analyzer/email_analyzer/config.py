"""Configuration et constantes partagées."""

import os

# Évite les conflits TensorFlow / Keras si transformers charge TF
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_KERAS", "1")

# Préréglages de période pour filtrage (CLI : --period)
PERIOD_TODAY = "today"
PERIOD_YESTERDAY = "yesterday"
PERIOD_3_DAYS = "3"
PERIOD_7_DAYS = "7"
PERIOD_11_DAYS = "11"
VALID_PERIODS = frozenset(
    {PERIOD_TODAY, PERIOD_YESTERDAY, PERIOD_3_DAYS, PERIOD_7_DAYS, PERIOD_11_DAYS}
)

# Limites par défaut pour l'API OpenAI (entrée / sortie)
DEFAULT_OPENAI_MAX_INPUT_CHARS = 12000
DEFAULT_OPENAI_BODY_CHARS = 2000
DEFAULT_OPENAI_MAX_TOKENS = 700
ECONOMY_OPENAI_MAX_INPUT_CHARS = 4000
ECONOMY_OPENAI_BODY_CHARS = 600
ECONOMY_OPENAI_MAX_TOKENS = 512

# Plafond de sortie pour /api/chat uniquement (Gemini 2.5 peut consommer du budget en raisonnement interne).
CHAT_MAX_OUTPUT_TOKENS = 4096

# Timeout client LLM (secondes). Volontairement < au timeout de passerelle : mieux
# vaut échouer vite que tenir la tâche de fond trop longtemps. Surchargeable via env.
DEFAULT_LLM_TIMEOUT_SECONDS = 60.0


def llm_timeout_seconds() -> float:
    """Timeout du client LLM en secondes (env LLM_TIMEOUT_SECONDS, défaut 60)."""
    raw = (os.environ.get("LLM_TIMEOUT_SECONDS") or "").strip()
    if not raw:
        return DEFAULT_LLM_TIMEOUT_SECONDS
    try:
        val = float(raw)
        return val if val > 0 else DEFAULT_LLM_TIMEOUT_SECONDS
    except ValueError:
        return DEFAULT_LLM_TIMEOUT_SECONDS


# Timeout socket IMAP (secondes). Sans timeout explicite, un serveur mail lent
# ou injoignable bloque le thread du worker bien au-delà du timeout du job
# arq (300s par défaut), qui finit par tuer le job côté arq sans jamais
# libérer le thread orphelin. Surchargeable via env.
DEFAULT_IMAP_TIMEOUT_SECONDS = 30.0


def imap_timeout_seconds() -> float:
    """Timeout socket IMAP en secondes (env IMAP_TIMEOUT_SECONDS, défaut 30)."""
    raw = (os.environ.get("IMAP_TIMEOUT_SECONDS") or "").strip()
    if not raw:
        return DEFAULT_IMAP_TIMEOUT_SECONDS
    try:
        val = float(raw)
        return val if val > 0 else DEFAULT_IMAP_TIMEOUT_SECONDS
    except ValueError:
        return DEFAULT_IMAP_TIMEOUT_SECONDS

# gemini-2.0-flash n'est plus disponible pour les nouveaux comptes (API renvoie 404).
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

ASSISTANT_PROVIDER_OPENAI = "openai"
ASSISTANT_PROVIDER_GEMINI = "gemini"
ASSISTANT_PROVIDER_NONE = "none"
VALID_ASSISTANT_PROVIDERS = frozenset(
    {ASSISTANT_PROVIDER_OPENAI, ASSISTANT_PROVIDER_GEMINI, ASSISTANT_PROVIDER_NONE}
)


def use_local_ml() -> bool:
    """True = charger transformers + spaCy ; False = API / heuristiques uniquement."""
    v = os.environ.get("EMAIL_ANALYZER_USE_LOCAL_ML", "true").strip().lower()
    return v not in ("0", "false", "no", "off")


# Chargement séquentiel par lots : au-delà de ce nombre d'emails candidats, la
# récupération IMAP est découpée en lots pour remonter une progression réelle
# au client au lieu de le faire attendre sans retour. En dessous du seuil,
# comportement inchangé (aucun lot, aucun callback de progression).
DEFAULT_BATCH_THRESHOLD = 15
DEFAULT_BATCH_CHUNK_SIZE = 10


def batch_threshold() -> int:
    """Nb d'emails candidats au-delà duquel le traitement est découpé en lots
    (env EMAIL_ANALYZER_BATCH_THRESHOLD, défaut 15)."""
    raw = (os.environ.get("EMAIL_ANALYZER_BATCH_THRESHOLD") or "").strip()
    if not raw:
        return DEFAULT_BATCH_THRESHOLD
    try:
        val = int(raw)
        return val if val > 0 else DEFAULT_BATCH_THRESHOLD
    except ValueError:
        return DEFAULT_BATCH_THRESHOLD


def batch_chunk_size() -> int:
    """Taille de lot pour le traitement séquentiel (env EMAIL_ANALYZER_BATCH_SIZE,
    défaut 10)."""
    raw = (os.environ.get("EMAIL_ANALYZER_BATCH_SIZE") or "").strip()
    if not raw:
        return DEFAULT_BATCH_CHUNK_SIZE
    try:
        val = int(raw)
        return val if val > 0 else DEFAULT_BATCH_CHUNK_SIZE
    except ValueError:
        return DEFAULT_BATCH_CHUNK_SIZE


# Store de jobs (jobs.py) et queue arq : Redis partagé entre le process API
# (uvicorn) et le(s) process worker (arq), ce qui lève la contrainte historique
# de un-seul-worker-uvicorn du store en mémoire.
DEFAULT_REDIS_URL = "redis://localhost:6379/0"


def redis_url() -> str:
    """URL de connexion Redis (env REDIS_URL, défaut redis://localhost:6379/0)."""
    raw = (os.environ.get("REDIS_URL") or "").strip()
    return raw or DEFAULT_REDIS_URL
