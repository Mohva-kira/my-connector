"""Façade : EmailProcessor, credentials .env, JSON structuré."""

import logging
import os
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv

from .config import (
    ASSISTANT_PROVIDER_OPENAI,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OPENAI_BODY_CHARS,
    DEFAULT_OPENAI_MAX_INPUT_CHARS,
    DEFAULT_OPENAI_MAX_TOKENS,
    VALID_ASSISTANT_PROVIDERS,
)
from .project_mail import EmailProjectAnalyzer, imap_days_for_period_arg
from .templates import generate_response_draft as build_response_draft


class EmailProcessor:
    """
    API interne : connexion IMAP, analyse, templates de réponse.
    Les identifiants sont lus dans l'environnement (fichier .env supporté).
    """

    def __init__(
        self,
        email_address: Optional[str] = None,
        password: Optional[str] = None,
        imap_server: Optional[str] = None,
        port: Optional[int] = None,
        cache_file: Optional[str] = None,
        max_deep_emails: Optional[int] = None,
        load_env: bool = True,
        imap_folder: Optional[str] = None,
        imap_use_ssl: Optional[bool] = None,
    ):
        if load_env:
            load_dotenv()

        self.email_address = (
            email_address
            or os.environ.get("IMAP_USER")
            or os.environ.get("EMAIL")
        )
        self.password = password or os.environ.get("IMAP_PASSWORD")
        self.imap_server = imap_server or os.environ.get("IMAP_HOST", "mail.mediasoftci.net")
        self.port = int(port or os.environ.get("IMAP_PORT", "993"))
        self.cache_file = cache_file or os.environ.get(
            "EMAIL_ANALYZER_CACHE_FILE", "email_analysis_cache.json"
        )
        self.max_deep_emails = int(
            max_deep_emails or os.environ.get("EMAIL_ANALYZER_MAX_DEEP_EMAILS", "20")
        )
        if imap_folder is not None:
            self.imap_folder = str(imap_folder).strip() or "INBOX"
        else:
            raw_folder = os.environ.get("IMAP_FOLDER", "INBOX")
            self.imap_folder = str(raw_folder).strip() or "INBOX"
        self.imap_use_ssl = (
            imap_use_ssl
            if imap_use_ssl is not None
            else os.environ.get("IMAP_USE_SSL", "true").strip().lower()
            not in ("0", "false", "no")
        )

    def process_latest_emails(
        self,
        project_filter: Union[str, List[str]],
        period: Optional[str] = None,
        days: int = 30,
        assistant_provider: str = ASSISTANT_PROVIDER_OPENAI,
        openai_model: str = "gpt-4o-mini",
        gemini_model: str = DEFAULT_GEMINI_MODEL,
        openai_max_tokens: int = DEFAULT_OPENAI_MAX_TOKENS,
        openai_max_input_chars: int = DEFAULT_OPENAI_MAX_INPUT_CHARS,
        openai_body_chars: int = DEFAULT_OPENAI_BODY_CHARS,
    ) -> Dict[str, Any]:
        """
        Récupère les emails correspondant aux projets, applique la période optionnelle,
        retourne le même JSON structuré que `generate_intelligent_summary` (par nom de projet).

        :param project_filter: un nom de projet ou une liste de noms.
        :param period: today | yesterday | 3 | 7 | 11 (optionnel) ; sinon fenêtre `days` pour IMAP.
        """
        if not self.email_address or not self.password:
            return {
                "_error": "Identifiants IMAP manquants (IMAP_USER / EMAIL et IMAP_PASSWORD).",
            }

        if assistant_provider not in VALID_ASSISTANT_PROVIDERS:
            return {"_error": f"assistant_provider invalide: {assistant_provider}"}

        projects: List[str] = (
            [project_filter] if isinstance(project_filter, str) else list(project_filter)
        )
        if not projects:
            return {"_error": "project_filter vide."}

        analyzer = EmailProjectAnalyzer(
            self.email_address,
            self.password,
            self.imap_server,
            self.port,
            max_deep_emails=self.max_deep_emails,
            cache_file=self.cache_file,
            imap_folder=self.imap_folder,
            prefer_ssl=self.imap_use_ssl,
        )

        try:
            if not analyzer.connect():
                return {"_error": "Échec de la connexion IMAP."}

            imap_days = imap_days_for_period_arg(period, days)
            project_data = analyzer.search_project_emails(projects, imap_days)

            if not project_data:
                filter_label = projects[0] if len(projects) == 1 else ", ".join(projects)
                return {
                    "_empty": True,
                    "_message": (
                        "Aucun email ne correspond au filtre dans le dossier IMAP sur la période "
                        "analysée (recherche dans le sujet et le corps)."
                    ),
                    "_filter": filter_label,
                    "_imap_folder": self.imap_folder,
                    "_days_back": imap_days,
                }

            if period:
                project_data = analyzer.apply_period_filter(project_data, period)
                total_in_period = sum(len(d.get("emails", [])) for d in project_data.values())
                if total_in_period == 0:
                    return {"_error": f"Aucun email dans la période {period}."}

            return analyzer.generate_intelligent_summary(
                project_data,
                assistant_provider=assistant_provider,
                openai_model=openai_model,
                gemini_model=gemini_model,
                openai_max_tokens=openai_max_tokens,
                openai_max_input_chars=openai_max_input_chars,
                openai_body_chars=openai_body_chars,
            )
        finally:
            analyzer.disconnect()

    def fetch_last_n_emails_for_chat(
        self,
        project_filter: str,
        period: Optional[str] = None,
        days: int = 30,
        n: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Récupère les n derniers emails correspondant au filtre (IMAP), sans lecture/écriture du cache JSON.
        Utilisé pour l’assistant conversationnel : contexte basé sur le contenu des messages, pas sur le JSON d’analyse.
        """
        if not self.email_address or not self.password:
            return []

        pf = project_filter.strip()
        if not pf:
            return []

        analyzer = EmailProjectAnalyzer(
            self.email_address,
            self.password,
            self.imap_server,
            self.port,
            max_deep_emails=self.max_deep_emails,
            cache_file=self.cache_file,
            imap_folder=self.imap_folder,
            use_email_cache=False,
            prefer_ssl=self.imap_use_ssl,
        )

        try:
            if not analyzer.connect():
                return []

            imap_days = imap_days_for_period_arg(period, days)
            project_data = analyzer.search_project_emails([pf], imap_days)
            if not project_data:
                return []

            if period:
                project_data = analyzer.apply_period_filter(project_data, period)
                if not project_data:
                    return []

            keys = [k for k in project_data.keys()]
            key = next((k for k in keys if k.lower() == pf.lower()), None) or keys[0]
            block = project_data.get(key) or {}
            emails_all = block.get("emails") or []
            if not emails_all:
                return []
            return emails_all[-n:]
        finally:
            analyzer.disconnect()

    def generate_response_draft(
        self,
        analysis_result: Dict[str, Any],
        project_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Choisit Routine / Alerte / Kickoff et remplit les placeholders.
        Si plusieurs projets, passez le nom explicitement ; sinon le premier champ utile est utilisé.
        """
        if project_name and project_name in analysis_result:
            inner = analysis_result[project_name]
            if isinstance(inner, dict):
                return build_response_draft(project_name, inner)

        if project_name and ("évaluation_risque" in analysis_result or "nb_emails" in analysis_result):
            return build_response_draft(project_name, analysis_result)

        keys = [k for k in analysis_result if not str(k).startswith("_")]
        if len(keys) == 1:
            only = keys[0]
            inner = analysis_result[only]
            if isinstance(inner, dict) and (
                "évaluation_risque" in inner or "nb_emails" in inner
            ):
                return build_response_draft(only, inner)

        if "évaluation_risque" in analysis_result or "nb_emails" in analysis_result:
            return build_response_draft(project_name or "Projet", analysis_result)

        return build_response_draft(project_name or "Projet", analysis_result)
