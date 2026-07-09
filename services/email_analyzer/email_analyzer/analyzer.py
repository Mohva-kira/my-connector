"""Façade : EmailProcessor, credentials .env, JSON structuré."""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Union

from dotenv import load_dotenv

from .classification import ProjectRules
from .config import (
    ASSISTANT_PROVIDER_OPENAI,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OPENAI_BODY_CHARS,
    DEFAULT_OPENAI_MAX_INPUT_CHARS,
    DEFAULT_OPENAI_MAX_TOKENS,
    VALID_ASSISTANT_PROVIDERS,
)
from .period import parse_email_datetime
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
        gmail_connection: Optional[Any] = None,
        outlook_connection: Optional[Any] = None,
        use_env_fallback: bool = True,
    ):
        if load_env:
            load_dotenv()

        # use_env_fallback=False isole les credentials d'un tenant SaaS de
        # tout IMAP_USER/IMAP_PASSWORD global du process (utilisé par le
        # mode legacy) — sans ça, un tenant sans IMAP configuré analyserait
        # silencieusement la boîte définie par le .env du serveur au lieu de
        # renvoyer une erreur ou de basculer sur Gmail (voir saas_logic.
        # processor_from_tenant).
        self.email_address = email_address or (
            (os.environ.get("IMAP_USER") or os.environ.get("EMAIL")) if use_env_fallback else None
        )
        self.password = password or (
            os.environ.get("IMAP_PASSWORD") if use_env_fallback else None
        )
        # Connexions OAuth (objets OAuthConnection), utilisées en repli quand
        # aucun IMAP n'est configuré — voir process_latest_emails. Gmail est
        # prioritaire sur Outlook si les deux sont connectés (choix arbitraire,
        # pas de signal pour départager ; à revisiter si ça devient un vrai cas).
        self.gmail_connection = gmail_connection
        self.outlook_connection = outlook_connection
        self.last_gmail_token_refresh: Optional[Dict[str, Any]] = None
        self.last_outlook_token_refresh: Optional[Dict[str, Any]] = None
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
        on_batch: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Récupère les emails correspondant aux projets, applique la période optionnelle,
        retourne le même JSON structuré que `generate_intelligent_summary` (par nom de projet).

        :param project_filter: un nom de projet ou une liste de noms.
        :param period: today | yesterday | 3 | 7 | 11 (optionnel) ; sinon fenêtre `days` pour IMAP.
        :param on_batch: callback optionnel(processed, total, partial) appelé après chaque
            lot lorsque le nombre d'emails candidats dépasse le seuil de chunking. Ignoré
            sur le chemin Gmail (pas de notion de lot, un seul appel API).

        Si aucun identifiant IMAP n'est configuré mais qu'une connexion Gmail ou
        Outlook OAuth l'est (``self.gmail_connection``/``self.outlook_connection``),
        bascule automatiquement sur l'API correspondante (Gmail prioritaire si les
        deux sont connectés) — voir Open Question #1 de progress-tracker.md /
        ``saas_logic.processor_from_tenant``.
        """
        has_imap = bool(self.email_address and self.password)
        use_gmail = not has_imap and self.gmail_connection is not None
        use_outlook = not has_imap and not use_gmail and self.outlook_connection is not None
        if not has_imap and not use_gmail and not use_outlook:
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
            self.email_address or "",
            self.password or "",
            self.imap_server,
            self.port,
            max_deep_emails=self.max_deep_emails,
            cache_file=self.cache_file,
            imap_folder=self.imap_folder,
            prefer_ssl=self.imap_use_ssl,
        )

        imap_days = imap_days_for_period_arg(period, days)
        source_label = "IMAP"

        if use_gmail:
            source_label = "Gmail"
            try:
                project_data = self._fetch_gmail_project_data(analyzer, projects, imap_days)
            except Exception as exc:
                logging.exception("Échec de la récupération Gmail")
                return {"_error": f"Échec de la récupération Gmail : {exc}"}
        elif use_outlook:
            source_label = "Outlook"
            try:
                project_data = self._fetch_outlook_project_data(analyzer, projects, imap_days)
            except Exception as exc:
                logging.exception("Échec de la récupération Outlook")
                return {"_error": f"Échec de la récupération Outlook : {exc}"}
        else:
            if not analyzer.connect():
                return {"_error": "Échec de la connexion IMAP."}
            try:
                project_data = analyzer.search_project_emails(projects, imap_days, on_batch=on_batch)
            finally:
                analyzer.disconnect()

        if not project_data:
            filter_label = projects[0] if len(projects) == 1 else ", ".join(projects)
            return {
                "_empty": True,
                "_message": (
                    f"Aucun email ne correspond au filtre dans {source_label} sur la période "
                    "analysée (recherche dans le sujet et le corps)."
                ),
                "_filter": filter_label,
                "_imap_folder": self.imap_folder if not (use_gmail or use_outlook) else None,
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

    def _fetch_gmail_project_data(
        self,
        analyzer: EmailProjectAnalyzer,
        projects: List[str],
        days_back: int,
    ) -> Dict[str, Any]:
        """Récupère les emails récents via Gmail OAuth (aucune connexion IMAP) et
        les fait matcher par ``analyzer`` exactement comme le chemin IMAP —
        réutilise ``check_project_relevance``/``extract_participants`` via
        ``search_project_emails_from_list``, pas de logique de matching dupliquée.
        Limite à 200 messages (une seule page Gmail API, pas de pagination) —
        suffisant pour un scan périodique mais pas un historique exhaustif ;
        limitation connue, à lever si besoin réel constaté.
        """
        from . import gmail_oauth

        query = gmail_oauth.build_gmail_query("", days_back=days_back)
        emails, token_refresh = gmail_oauth.fetch_emails(
            self.gmail_connection.access_token_encrypted,
            self.gmail_connection.refresh_token_encrypted,
            self.gmail_connection.token_expiry,
            query=query,
            max_results=200,
        )
        if token_refresh:
            self.last_gmail_token_refresh = token_refresh
        return analyzer.search_project_emails_from_list(emails, projects)

    def _fetch_outlook_project_data(
        self,
        analyzer: EmailProjectAnalyzer,
        projects: List[str],
        days_back: int,
    ) -> Dict[str, Any]:
        """Miroir de `_fetch_gmail_project_data` pour Outlook (Microsoft Graph) —
        même limitation connue (200 messages, une seule page, pas de pagination)."""
        from . import outlook_oauth

        query = outlook_oauth.build_outlook_filter(days_back=days_back)
        emails, token_refresh = outlook_oauth.fetch_emails(
            self.outlook_connection.access_token_encrypted,
            self.outlook_connection.refresh_token_encrypted,
            self.outlook_connection.token_expiry,
            query=query,
            max_results=200,
        )
        if token_refresh:
            self.last_outlook_token_refresh = token_refresh
        return analyzer.search_project_emails_from_list(emails, projects)

    def process_delta(
        self,
        project_name: str,
        since: Optional[datetime],
        assistant_provider: str = ASSISTANT_PROVIDER_OPENAI,
        openai_model: str = "gpt-4o-mini",
        gemini_model: str = DEFAULT_GEMINI_MODEL,
        rules_matrix: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """Fast-Track : récupère uniquement les emails reçus après ``since`` et
        régénère le résumé à partir de ce seul delta (architecture.md, Process 2).

        Si ``since`` est ``None`` (premier rafraîchissement d'un projet), retombe
        sur une fenêtre par défaut de 30 jours, comme ``process_latest_emails``.

        Contrairement à ``process_latest_emails``, expose aussi la liste brute
        des emails du delta (clé ``"emails"``) pour permettre leur persistance
        par l'appelant (voir ``email_analyzer/analysis_tasks.py``).

        ``rules_matrix`` (JSONB brut de ``Project.rules_matrix``) alimente la
        classification multi-critères (``email_analyzer/classification.py``)
        en plus du simple nom de projet ; ``None`` reproduit exactement
        l'ancien comportement (recherche du nom de projet en sous-chaîne).
        """
        if not self.email_address or not self.password:
            return {"_error": "Identifiants IMAP manquants (IMAP_USER / EMAIL et IMAP_PASSWORD)."}

        name = project_name.strip()
        if not name:
            return {"_error": "project_name vide."}

        if since is not None:
            since_utc = since if since.tzinfo else since.replace(tzinfo=timezone.utc)
            # +2 jours de marge : IMAP SINCE ne connaît que la granularité jour
            # (fuseau du serveur) — le filtrage exact se fait ensuite en Python.
            days_back = max(1, (datetime.now(timezone.utc) - since_utc).days + 2)
        else:
            since_utc = None
            days_back = 30

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

            rules_map = (
                {name: ProjectRules.from_dict(rules_matrix)} if rules_matrix else None
            )
            project_data = analyzer.search_project_emails([name], days_back, rules_map=rules_map)
            if not project_data:
                return {
                    "_empty": True,
                    "emails": [],
                    "_message": "Aucun email ne correspond au filtre dans la fenêtre analysée.",
                }

            keys = list(project_data.keys())
            key = next((k for k in keys if k.lower() == name.lower()), None) or keys[0]
            block = project_data.get(key) or {}
            emails: List[Dict[str, Any]] = list(block.get("emails") or [])

            if since_utc is not None:
                filtered: List[Dict[str, Any]] = []
                for em in emails:
                    parsed = parse_email_datetime(em.get("date"))
                    if parsed is None:
                        continue
                    parsed_utc = parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
                    if parsed_utc > since_utc:
                        filtered.append(em)
                emails = filtered

            if not emails:
                return {
                    "_empty": True,
                    "emails": [],
                    "_message": "Aucun nouvel email depuis le dernier rafraîchissement.",
                }

            participants: set = set()
            for em in emails:
                analyzer.extract_participants(em, participants)

            filtered_data = {
                key: {
                    "emails": emails,
                    "participants": participants,
                    "keywords": {},
                    "dates": [em.get("date") for em in emails],
                }
            }
            summary = analyzer.generate_intelligent_summary(
                filtered_data,
                assistant_provider=assistant_provider,
                openai_model=openai_model,
                gemini_model=gemini_model,
            )
            return {"emails": emails, "summary": summary.get(key, {})}
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
            project_data = analyzer.search_project_emails([pf], imap_days, max_matches=n)
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
