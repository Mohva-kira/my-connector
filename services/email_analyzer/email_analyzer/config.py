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
