"""Package d'analyse d'emails projet (IMAP + IA)."""

from .analyzer import EmailProcessor
from .config import (
    ASSISTANT_PROVIDER_GEMINI,
    ASSISTANT_PROVIDER_NONE,
    ASSISTANT_PROVIDER_OPENAI,
    VALID_ASSISTANT_PROVIDERS,
)
from .project_mail import EmailProjectAnalyzer
from .templates import generate_response_draft

__all__ = [
    "EmailProcessor",
    "EmailProjectAnalyzer",
    "generate_response_draft",
    "ASSISTANT_PROVIDER_OPENAI",
    "ASSISTANT_PROVIDER_GEMINI",
    "ASSISTANT_PROVIDER_NONE",
    "VALID_ASSISTANT_PROVIDERS",
]
