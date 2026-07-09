"""Connexion IMAP, recherche par projet et orchestration de l'analyse."""

import email
import imaplib
import json
import logging
import os
import re
from html import unescape
from collections import defaultdict
from datetime import datetime, timedelta
from email.header import decode_header
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional

from .ai_intelligent import get_shared_analyzer
from .classification import ProjectRules, score_project_relevance
from .config import (
    ASSISTANT_PROVIDER_GEMINI,
    ASSISTANT_PROVIDER_NONE,
    ASSISTANT_PROVIDER_OPENAI,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OPENAI_BODY_CHARS,
    DEFAULT_OPENAI_MAX_INPUT_CHARS,
    DEFAULT_OPENAI_MAX_TOKENS,
    VALID_PERIODS,
    batch_chunk_size,
    batch_threshold,
    imap_timeout_seconds,
)
from .llm import (
    build_résumé_assistant_unifié,
    generate_gemini_assistant_summary,
    generate_openai_assistant_summary,
    get_gemini_api_key,
)
from .period import filter_emails_by_period, imap_days_back_for_period


class EmailProjectAnalyzer:
    def __init__(
        self,
        email_address: str,
        password: str,
        imap_server: str = "mail.mediasoftci.net",
        port: int = 993,
        max_deep_emails: int = 20,
        cache_file: str = "email_analysis_cache.json",
        imap_folder: Optional[str] = None,
        use_email_cache: bool = True,
        prefer_ssl: bool = True,
    ):
        self.email_address = email_address
        self.password = password
        self.imap_server = imap_server
        self.port = port
        self.prefer_ssl = prefer_ssl
        raw_folder = (imap_folder if imap_folder is not None else os.environ.get("IMAP_FOLDER")) or "INBOX"
        self.imap_folder = str(raw_folder).strip() or "INBOX"
        self.mail = None
        self.project_emails = defaultdict(list)
        self.max_deep_emails = max(5, max_deep_emails)
        self.cache_file = cache_file
        self.use_email_cache = use_email_cache
        self.email_cache = self.load_cache() if use_email_cache else {}
        self.step_timings = defaultdict(float)
        self.step_counts = defaultdict(int)

        self.ai_analyzer = get_shared_analyzer()

    def record_timing(self, step_name: str, duration_seconds: float):
        """Enregistre le temps passé sur une étape."""
        self.step_timings[step_name] += duration_seconds
        self.step_counts[step_name] += 1

    def get_top_timing_steps(self, top_n: int = 3) -> List[Dict]:
        """Retourne les étapes les plus coûteuses."""
        sorted_steps = sorted(self.step_timings.items(), key=lambda item: item[1], reverse=True)
        top_steps = []
        for step_name, total_seconds in sorted_steps[:top_n]:
            runs = self.step_counts.get(step_name, 1)
            top_steps.append(
                {
                    "étape": step_name,
                    "total_s": round(total_seconds, 3),
                    "runs": runs,
                    "moyenne_s": round(total_seconds / max(1, runs), 3),
                }
            )
        return top_steps

    def load_cache(self) -> Dict:
        """Charge le cache local des emails déjà traités."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    raw_cache = json.load(f)
                if isinstance(raw_cache, dict):
                    return raw_cache
        except Exception as e:
            logging.warning("Impossible de charger le cache: %s", e)
        return {}

    def save_cache(self):
        """Sauvegarde un cache borné pour éviter les retraitements."""
        try:
            max_entries = 2000
            if len(self.email_cache) > max_entries:
                keys = list(self.email_cache.keys())[-max_entries:]
                self.email_cache = {k: self.email_cache[k] for k in keys}
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.email_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.warning("Impossible de sauvegarder le cache: %s", e)

    def truncate_text(self, text: str, max_len: int = 160) -> str:
        """Tronque proprement le texte pour l'affichage console."""
        if not text:
            return ""
        clean = " ".join(text.split())
        if len(clean) <= max_len:
            return clean
        return clean[: max_len - 3].rstrip() + "..."

    def decode_best_effort(self, payload: bytes, charset_hint: str = None) -> str:
        """Décode au mieux les corps d'email pour limiter la perte d'accents."""
        if not payload:
            return ""
        candidates = [charset_hint, "utf-8", "latin-1", "cp1252"]
        for encoding in candidates:
            if not encoding:
                continue
            try:
                return payload.decode(encoding, errors="ignore")
            except Exception:
                continue
        return payload.decode("utf-8", errors="ignore")

    def normalize_email_text(self, email_content: Dict) -> str:
        """Construit une version normalisée réutilisable par les analyseurs."""
        subject = email_content.get("subject", "")
        body = email_content.get("body", "")
        return f"{subject} {body}".lower()

    def strip_html_to_text(self, raw: str) -> str:
        """Extrait du texte lisible depuis du HTML (emails sans partie text/plain)."""
        if not raw:
            return ""
        without_blocks = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw)
        with_breaks = re.sub(r"(?i)<br\s*/?>", "\n", without_blocks)
        without_tags = re.sub(r"(?i)<[^>]+>", " ", with_breaks)
        return " ".join(unescape(without_tags).split())

    def _cached_content_usable(self, content: Optional[Dict]) -> bool:
        """True si le cache contient assez de données pour réappliquer le filtre projet."""
        if not content or not isinstance(content, dict):
            return False
        if (content.get("normalized_text") or "").strip():
            return True
        subj = (content.get("subject") or "").strip()
        body = (content.get("body") or "").strip()
        return bool(subj or body)

    def _escape_imap_quoted(self, s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"')

    def _or_subj_text_clause(self, project_filter: str) -> str:
        q = self._escape_imap_quoted(project_filter.strip())
        if not q:
            return ""
        return f'(OR SUBJECT "{q}" TEXT "{q}")'

    def _uid_search_narrow_ids(self, project_filters: List[str], start_date: str) -> Optional[List[bytes]]:
        """
        Réduit les UID via SEARCH OR SUBJECT/TEXT + SINCE (côté serveur).
        Retourne [] si aucun candidat, None si la recherche n'est pas utilisable (fallback SINCE large).
        Désactiver : EMAIL_ANALYZER_IMAP_FAST_SEARCH=0
        """
        flag = os.environ.get("EMAIL_ANALYZER_IMAP_FAST_SEARCH", "1").strip().lower()
        if flag in ("0", "false", "no", "off"):
            return None
        clauses = [
            self._or_subj_text_clause(f) for f in project_filters if f and str(f).strip()
        ]
        clauses = [c for c in clauses if c]
        if not clauses:
            return None
        nested = clauses[0]
        for c in clauses[1:]:
            nested = f"(OR {nested} {c})"
        criteria = f'{nested} SINCE "{start_date}"'
        charset: Optional[str] = None
        for f in project_filters:
            if not f:
                continue
            try:
                str(f).encode("ascii")
            except UnicodeEncodeError:
                charset = "UTF-8"
                break
        try:
            if charset:
                typ, data = self.mail.uid("search", charset, criteria)
            else:
                typ, data = self.mail.uid("search", None, criteria)
            if typ != "OK" or not data:
                logging.info(
                    "UID SEARCH étroit indisponible (typ=%s), fallback SINCE large", typ
                )
                return None
            raw = data[0]
            if raw is None:
                return None
            if isinstance(raw, bytes):
                ids = raw.split()
            else:
                ids = str(raw).encode("ascii", errors="ignore").split()
            logging.info(
                "UID SEARCH étroit : %s message(s) candidat(s) (période SINCE)",
                len(ids),
            )
            return ids
        except Exception as e:
            logging.info("UID SEARCH étroit échoué (%s), fallback SINCE large", e)
            return None

    def format_flags(self, flags: List[str]) -> str:
        """Rend les flags de criticité plus lisibles pour l'utilisateur."""
        if not flags:
            return "aucun indicateur explicite"
        readable = []
        for flag in flags[:3]:
            if flag.startswith("urgent_"):
                readable.append(f"urgence ({flag.replace('urgent_', '')})")
            else:
                readable.append(flag)
        return ", ".join(readable)

    def build_project_diagnostic(self, analysis: Dict) -> str:
        """Crée une synthèse orientée décision pour le rapport."""
        risk = analysis.get("évaluation_risque", {})
        sentiment = analysis.get("analyse_sentiment", {})
        priority = analysis.get("priorité_attention", "NORMALE")
        risk_level = risk.get("niveau_risque", "INDETERMINÉ")
        trend = sentiment.get("tendance", "Neutre")
        email_count = analysis.get("nb_emails", 0)

        if risk_level == "CRITIQUE":
            return (
                f"Situation tendue ({email_count} emails, tendance {trend.lower()}). "
                f"Priorité {priority.lower()}: escalade immédiate et plan d'action court terme."
            )
        if risk_level == "MODÉRÉ":
            return (
                f"Situation sous surveillance ({email_count} emails, tendance {trend.lower()}). "
                f"Prévoir un point de cadrage et un suivi hebdomadaire."
            )
        return (
            f"Situation globalement stable ({email_count} emails, tendance {trend.lower()}). "
            f"Maintenir le rythme de suivi actuel."
        )

    def format_project_report(self, project_name: str, analysis: Dict) -> List[str]:
        """Formate un bloc de rapport projet plus narratif et actionnable."""
        lines = []
        lines.append(f"\n🚀 PROJET: {project_name}")
        lines.append(f"   📊 Priorité: {analysis.get('priorité_attention', 'N/A')}")
        lines.append(f"   📧 Emails: {analysis.get('nb_emails', 0)}")
        lines.append(f"   👥 Participants: {analysis.get('nb_participants', 0)}")
        lines.append(f"   🧭 Diagnostic: {self.build_project_diagnostic(analysis)}")

        sentiment = analysis.get("analyse_sentiment", {})
        if sentiment.get("tendance"):
            method = sentiment.get("méthode", "N/A")
            confidence = sentiment.get("confiance_moyenne")
            confidence_txt = f", confiance moyenne {confidence}" if confidence is not None else ""
            lines.append(
                f"   😊 Sentiment: {sentiment['tendance']} (méthode: {method}{confidence_txt})"
            )

        risk = analysis.get("évaluation_risque", {})
        risk_score = risk.get("score_risque", 0)
        lines.append(f"   ⚠️ Risque: {risk.get('niveau_risque', 'N/A')} (score: {risk_score}/100)")
        risk_factors = risk.get("facteurs_risque", [])
        if risk_factors:
            lines.append(f"   🔎 Facteurs: {self.truncate_text('; '.join(risk_factors), 180)}")
        lines.append(f"   💡 Action: {risk.get('recommandation', 'Suivi standard recommandé')}")

        entities = analysis.get("entités_extraites", {})
        techs = entities.get("technologies", [])
        montants = entities.get("montants", [])
        if techs:
            lines.append(f"   💻 Technologies mentionnées: {', '.join(techs[:5])}")
        clean_montants = [m for m in montants if str(m).strip() and str(m).strip() != "000"]
        if clean_montants:
            lines.append(f"   💰 Montants détectés: {', '.join(clean_montants[:3])}")

        auto_summary = analysis.get("résumé_automatique", {})
        if isinstance(auto_summary, dict) and auto_summary.get("résumé_automatique"):
            method = auto_summary.get("méthode", "N/A")
            summary_text = self.truncate_text(auto_summary["résumé_automatique"], 260)
            lines.append(f"   📝 Synthèse ({method}): {summary_text}")

        critical = analysis.get("emails_critiques", [])
        if critical:
            lines.append(f"   🚨 Emails critiques: {len(critical)} détectés")
            for critical_email in critical[:2]:
                subject = self.truncate_text(critical_email.get("subject", ""), 70)
                reasons = self.format_flags(critical_email.get("flags", []))
                score = critical_email.get("criticality_score", 0)
                lines.append(f"      - {subject} (score: {score}, causes: {reasons})")

        assistant = analysis.get("résumé_assistant")
        if not isinstance(assistant, dict):
            assistant = analysis.get("résumé_assistant_openai")
        if isinstance(assistant, dict):
            fournisseur = assistant.get("fournisseur") or "openai"
            if fournisseur == ASSISTANT_PROVIDER_GEMINI:
                label = "Gemini"
            elif fournisseur == ASSISTANT_PROVIDER_NONE:
                label = "désactivé"
            else:
                label = "OpenAI"
            if assistant.get("texte"):
                model = assistant.get("modèle", "")
                preview = self.truncate_text(assistant["texte"], 400)
                lines.append(f"   🤖 Assistant ({label}) ({model}): {preview}")
            elif assistant.get("erreur"):
                lines.append(f"   🤖 Assistant ({label}): ({assistant['erreur']})")
        return lines

    def connect(self) -> bool:
        """Connexion au serveur IMAP."""
        try:
            logging.info(
                "Tentative de connexion à %s:%s ssl=%s",
                self.imap_server,
                self.port,
                self.prefer_ssl,
            )
            timeout = imap_timeout_seconds()
            if self.prefer_ssl:
                try:
                    self.mail = imaplib.IMAP4_SSL(self.imap_server, self.port, timeout=timeout)
                    logging.info("Connexion SSL établie")
                except Exception as ssl_error:
                    logging.warning("Échec SSL: %s", ssl_error)
                    logging.info("Tentative de connexion sans SSL...")
                    self.mail = imaplib.IMAP4(self.imap_server, 143, timeout=timeout)
                    logging.info("Connexion non-SSL établie")
            else:
                self.mail = imaplib.IMAP4(self.imap_server, self.port, timeout=timeout)
                logging.info("Connexion non-SSL (explicit) établie")

            logging.info("Tentative d'authentification pour %s", self.email_address)
            self.mail.login(self.email_address, self.password)
            logging.info("Connexion réussie")
            return True
        except Exception as e:
            logging.error("Erreur de connexion: %s", e)
            return False

    def disconnect(self):
        """Déconnexion du serveur IMAP."""
        if self.mail:
            try:
                self.mail.close()
                self.mail.logout()
                logging.info("Déconnexion réussie")
            except Exception:
                pass

    def decode_header_value(self, value: str) -> str:
        """Décode les en-têtes d'email."""
        if not value:
            return ""
        decoded_parts = decode_header(value)
        decoded_string = ""
        for part, encoding in decoded_parts:
            try:
                if isinstance(part, bytes):
                    decoded_string += part.decode(encoding or "utf-8", errors="ignore")
                else:
                    decoded_string += part
            except Exception:
                pass
        return decoded_string

    def extract_email_content(self, msg) -> Dict:
        """Extrait le contenu d'un email."""
        content = {
            "subject": self.decode_header_value(msg["Subject"]),
            "from": self.decode_header_value(msg["From"]),
            "to": self.decode_header_value(msg["To"]),
            "cc": self.decode_header_value(msg["Cc"]),
            "date": msg["Date"],
            # Identifiant stable de l'email, utilisé pour la déduplication lors
            # de la persistance (Fast-Track — voir email_analyzer/analysis_tasks.py).
            "message_id": self.decode_header_value(msg["Message-ID"]),
            "body": "",
        }

        try:
            if msg.is_multipart():
                plain_part = None
                html_part = None
                for part in msg.walk():
                    ctype = part.get_content_type()
                    if ctype == "text/plain" and plain_part is None:
                        plain_part = part
                    elif ctype == "text/html" and html_part is None:
                        html_part = part
                if plain_part:
                    body = plain_part.get_payload(decode=True)
                    if body:
                        content["body"] = self.decode_best_effort(body, plain_part.get_content_charset())
                elif html_part:
                    raw = html_part.get_payload(decode=True)
                    if raw:
                        html_str = self.decode_best_effort(raw, html_part.get_content_charset())
                        content["body"] = self.strip_html_to_text(html_str)
            else:
                body = msg.get_payload(decode=True)
                if body:
                    decoded = self.decode_best_effort(body, msg.get_content_charset())
                    if msg.get_content_type() == "text/html":
                        content["body"] = self.strip_html_to_text(decoded)
                    else:
                        content["body"] = decoded
        except Exception:
            pass

        content["normalized_text"] = self.normalize_email_text(content)
        return content

    def _process_one_email_id(
        self,
        email_id,
        use_uid_fetch: bool,
        project_filters: List[str],
        project_data: Dict,
        rules_map: Optional[Dict[str, ProjectRules]] = None,
    ) -> None:
        """Fetch (cache-aware) + parse un email et l'ajoute à `project_data` s'il
        correspond à un des `project_filters`. Bloc de travail partagé par le
        balayage avant (toutes les correspondances) et le balayage arrière borné
        (dernières correspondances seulement, cf. `max_matches`)."""
        try:
            cache_key = email_id.decode(errors="ignore") if isinstance(email_id, bytes) else str(email_id)
            cache_entry = self.email_cache.get(cache_key) if self.use_email_cache else None
            email_content = None
            if cache_entry and isinstance(cache_entry.get("content"), dict):
                cand = cache_entry["content"]
                if self._cached_content_usable(cand):
                    email_content = cand

            if email_content is None:
                fetch_start = perf_counter()
                if use_uid_fetch:
                    status, msg_data = self.mail.uid("fetch", email_id, "(RFC822)")
                else:
                    status, msg_data = self.mail.fetch(email_id, "(RFC822)")
                self.record_timing("imap_fetch_full_message", perf_counter() - fetch_start)
                if status != "OK" or not msg_data or not msg_data[0]:
                    return

                parse_start = perf_counter()
                raw_payload = msg_data[0][1]
                if not isinstance(raw_payload, (bytes, bytearray)):
                    return
                msg = email.message_from_bytes(raw_payload)
                email_content = self.extract_email_content(msg)
                self.record_timing("email_extract_content", perf_counter() - parse_start)
                if self.use_email_cache:
                    self.email_cache[cache_key] = {"content": email_content}

            # Construit via .items() (jamais un accès par crochet) pour ne
            # jamais déclencher la factory du defaultdict pour un projet pas
            # encore matché — sinon on changerait silencieusement le jeu de
            # clés retourné par search_project_emails.
            participants_map = {name: data["participants"] for name, data in project_data.items()}
            matching_projects = self.check_project_relevance(
                email_content, project_filters, rules_map=rules_map, participants_map=participants_map
            )
            if not matching_projects or not email_content:
                return

            for project in matching_projects:
                project_data[project]["emails"].append(email_content)
                project_data[project]["dates"].append(email_content.get("date"))
                self.extract_participants(email_content, project_data[project]["participants"])
        except Exception:
            return

    def search_project_emails(
        self,
        project_filters: List[str],
        days_back: int = 30,
        on_batch: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
        max_matches: Optional[int] = None,
        rules_map: Optional[Dict[str, ProjectRules]] = None,
    ) -> Dict:
        """Recherche les emails concernant les projets spécifiés.

        Si le nombre d'emails candidats dépasse `batch_threshold()`, le fetch/parse
        est découpé en lots de `batch_chunk_size()` et `on_batch(processed, total,
        partial)` est appelé après chaque lot (progression + aperçu rapide basé sur
        `identify_critical_emails`, rule-based donc sans coût ML). En dessous du
        seuil, `on_batch` n'est jamais invoqué — comportement inchangé.

        Si `max_matches` est fourni, le balayage se fait depuis les emails les plus
        récents (ordre IMAP inverse) et s'arrête dès que chaque filtre a atteint
        `max_matches` correspondances — évite de scanner tout le mailbox quand seuls
        les derniers emails d'un projet sont utiles (contexte de `/api/chat`).
        `on_batch` est ignoré si `max_matches` est fourni (balayage non séquentiel,
        pas de notion de "lot" à rapporter).
        """
        if not self.mail:
            return {}

        try:
            status, _count = self.mail.select(self.imap_folder)
            if status != "OK":
                return {}

            start_date = (datetime.now() - timedelta(days=days_back)).strftime("%d-%b-%Y")
            search_criteria = f'(SINCE "{start_date}")'

            narrow_ids = self._uid_search_narrow_ids(project_filters, start_date)
            use_uid_fetch: bool
            if narrow_ids is not None:
                email_ids = narrow_ids
                use_uid_fetch = True
                if not email_ids:
                    logging.info("Aucun candidat après UID SEARCH étroit, rien à analyser")
                    return {}
            else:
                status, messages = self.mail.search(None, search_criteria)
                if status != "OK":
                    return {}
                email_ids = messages[0].split()
                use_uid_fetch = False

            total = len(email_ids)
            logging.info("Trouvé %s emails à analyser", total)

            project_data = defaultdict(
                lambda: {
                    "emails": [],
                    "participants": set(),
                    "keywords": defaultdict(int),
                    "dates": [],
                }
            )

            if max_matches is not None:
                # Balayage arrière borné : s'arrête dès que chaque filtre a assez
                # de correspondances, sans lire tout le mailbox (contexte chat).
                remaining = set(project_filters)
                for email_id in reversed(email_ids):
                    self._process_one_email_id(
                        email_id, use_uid_fetch, project_filters, project_data, rules_map=rules_map
                    )
                    remaining = {
                        pf
                        for pf in remaining
                        if len(project_data.get(pf, {}).get("emails", [])) < max_matches
                    }
                    if not remaining:
                        break
                # Le balayage arrière ajoute du plus récent au plus ancien :
                # remettre chaque projet en ordre chronologique.
                for data in project_data.values():
                    data["emails"].reverse()
                    data["dates"].reverse()
                if self.use_email_cache:
                    self.save_cache()
                return dict(project_data)

            chunk_size = batch_chunk_size()
            chunked = on_batch is not None and total > batch_threshold()

            for i, email_id in enumerate(email_ids):
                if i % 20 == 0:
                    logging.info("Traitement: %s/%s emails", i + 1, total)
                self._process_one_email_id(
                    email_id, use_uid_fetch, project_filters, project_data, rules_map=rules_map
                )
                if chunked and (i + 1) % chunk_size == 0:
                    self._emit_batch_progress(on_batch, i + 1, total, project_data)

            if chunked and total % chunk_size != 0:
                self._emit_batch_progress(on_batch, total, total, project_data)

            if self.use_email_cache:
                self.save_cache()
            return dict(project_data)

        except Exception as e:
            logging.error("Erreur recherche: %s", e)
            return {}

    def search_project_emails_from_list(
        self,
        emails: List[Dict],
        project_filters: List[str],
        rules_map: Optional[Dict[str, ProjectRules]] = None,
    ) -> Dict:
        """Variante de `search_project_emails` pour une source déjà normalisée
        et déjà récupérée (pas d'IMAP) — ex. emails Gmail
        (`gmail_oauth.fetch_emails`, même format de dict : `subject/from/to/
        date/body/normalized_text`). Réutilise le même matching par projet
        (`check_project_relevance`/`extract_participants`) que le chemin IMAP,
        sans connexion ni cache IMAP."""
        project_data: Dict = defaultdict(
            lambda: {"emails": [], "participants": set(), "keywords": defaultdict(int), "dates": []}
        )
        for email_content in emails:
            participants_map = {name: data["participants"] for name, data in project_data.items()}
            matching_projects = self.check_project_relevance(
                email_content, project_filters, rules_map=rules_map, participants_map=participants_map
            )
            for project in matching_projects:
                project_data[project]["emails"].append(email_content)
                project_data[project]["dates"].append(email_content.get("date"))
                self.extract_participants(email_content, project_data[project]["participants"])
        return dict(project_data)

    def _emit_batch_progress(
        self,
        on_batch: Callable[[int, int, Dict[str, Any]], None],
        processed: int,
        total: int,
        project_data: Dict,
    ) -> None:
        """Construit un aperçu partiel (nb_emails + emails critiques, rule-based)
        et le remonte via `on_batch`. Une erreur ici ne doit jamais interrompre
        l'analyse en cours."""
        try:
            partial = {
                name: {
                    "nb_emails": len(data["emails"]),
                    "emails_critiques": self.ai_analyzer.identify_critical_emails(data["emails"]),
                }
                for name, data in project_data.items()
            }
            on_batch(processed, total, partial)
        except Exception:
            logging.exception("Callback de progression en échec (ignoré, n'affecte pas l'analyse)")

    def check_project_relevance(
        self,
        email_content: Dict,
        project_filters: List[str],
        rules_map: Optional[Dict[str, ProjectRules]] = None,
        participants_map: Optional[Dict[str, set]] = None,
    ) -> List[str]:
        """Vérifie si un email concerne un projet spécifique.

        Sans `rules_map` ni `participants_map`, délègue à
        `classification.score_project_relevance` avec seulement le signal
        `project_name` — reproduit exactement l'ancienne recherche en
        sous-chaîne. Avec ces paramètres, combine mots-clés/adresses/
        participants connus/références internes en un score de confiance
        (voir MATCH_THRESHOLD dans classification.py)."""
        matching_projects = []
        for project_filter in project_filters:
            rules = rules_map.get(project_filter) if rules_map else None
            known_participants = participants_map.get(project_filter) if participants_map else None
            result = score_project_relevance(email_content, project_filter, rules, known_participants)
            if result.matched:
                matching_projects.append(project_filter)
        return matching_projects

    def extract_participants(self, email_content: Dict, participants: set):
        """Extrait les participants du projet."""
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        for field in ["from", "to"]:
            if email_content[field]:
                emails = re.findall(email_pattern, email_content[field])
                participants.update(emails)

    def apply_period_filter(self, project_data: Dict, period: str) -> Dict:
        """Restreint les emails et les participants à la fenêtre de période."""
        if period not in VALID_PERIODS:
            return project_data
        filtered_projects: Dict = {}
        for name, data in project_data.items():
            emails = filter_emails_by_period(data.get("emails", []), period)
            participants = set()
            for em in emails:
                self.extract_participants(em, participants)
            filtered_projects[name] = {
                "emails": emails,
                "participants": participants,
                "keywords": data.get("keywords", defaultdict(int)),
                "dates": [em.get("date") for em in emails],
            }
        return filtered_projects

    def generate_intelligent_summary(
        self,
        project_data: Dict,
        assistant_provider: str = ASSISTANT_PROVIDER_OPENAI,
        openai_model: str = "gpt-4o-mini",
        gemini_model: str = DEFAULT_GEMINI_MODEL,
        openai_max_tokens: int = DEFAULT_OPENAI_MAX_TOKENS,
        openai_max_input_chars: int = DEFAULT_OPENAI_MAX_INPUT_CHARS,
        openai_body_chars: int = DEFAULT_OPENAI_BODY_CHARS,
    ) -> Dict:
        """Génère un résumé intelligent avec IA."""
        intelligent_summary = {}

        for project_name, data in project_data.items():
            logging.info("Analyse IA pour le projet: %s", project_name)
            emails_all = data["emails"]
            deep_emails = emails_all[-self.max_deep_emails :]

            sentiment_start = perf_counter()
            sentiment_analysis = self.ai_analyzer.analyze_sentiment(deep_emails)
            self.record_timing("ia_sentiment", perf_counter() - sentiment_start)

            entities_start = perf_counter()
            entities = self.ai_analyzer.extract_entities(deep_emails)
            self.record_timing("ia_entities", perf_counter() - entities_start)

            summary_start = perf_counter()
            auto_summary = self.ai_analyzer.generate_auto_summary(deep_emails)
            self.record_timing("ia_summary", perf_counter() - summary_start)

            risk_start = perf_counter()
            risk_assessment = self.ai_analyzer.calculate_risk_score(
                emails_all, sentiment_analysis, entities
            )
            self.record_timing("ia_risk", perf_counter() - risk_start)

            critical_start = perf_counter()
            critical_emails = self.ai_analyzer.identify_critical_emails(emails_all)
            self.record_timing("ia_critical", perf_counter() - critical_start)

            participants_list = list(data["participants"])
            email_count = len(emails_all)

            intelligent_summary[project_name] = {
                "nb_emails": email_count,
                "nb_participants": len(participants_list),
                "participants": participants_list[:10],
                "emails_analyzes_en_profondeur": len(deep_emails),
                "analyse_sentiment": sentiment_analysis,
                "entités_extraites": entities,
                "résumé_automatique": auto_summary,
                "évaluation_risque": risk_assessment,
                "emails_critiques": critical_emails,
                "score_activité": email_count + len(entities.get("technologies", [])),
                "priorité_attention": (
                    "HAUTE" if risk_assessment.get("niveau_risque") == "CRITIQUE" else "NORMALE"
                ),
            }

            if assistant_provider == ASSISTANT_PROVIDER_NONE:
                raw_disabled = {
                    "texte": None,
                    "modèle": None,
                    "erreur": "désactivé (--assistant-provider none ou --no-openai)",
                    "max_tokens": None,
                    "max_input_chars": None,
                    "max_body_chars": None,
                }
                intelligent_summary[project_name]["résumé_assistant"] = build_résumé_assistant_unifié(
                    ASSISTANT_PROVIDER_NONE, raw_disabled
                )
                intelligent_summary[project_name]["résumé_assistant_openai"] = {
                    "texte": None,
                    "modèle": openai_model,
                    "erreur": raw_disabled["erreur"],
                }
            elif assistant_provider == ASSISTANT_PROVIDER_OPENAI:
                openai_start = perf_counter()
                raw_oa = generate_openai_assistant_summary(
                    emails_all,
                    project_name,
                    openai_model,
                    os.environ.get("OPENAI_API_KEY"),
                    max_tokens=openai_max_tokens,
                    max_input_chars=openai_max_input_chars,
                    max_body_chars=openai_body_chars,
                )
                self.record_timing("openai_assistant", perf_counter() - openai_start)
                intelligent_summary[project_name]["résumé_assistant"] = build_résumé_assistant_unifié(
                    ASSISTANT_PROVIDER_OPENAI, raw_oa
                )
                intelligent_summary[project_name]["résumé_assistant_openai"] = raw_oa
            elif assistant_provider == ASSISTANT_PROVIDER_GEMINI:
                gemini_start = perf_counter()
                raw_g = generate_gemini_assistant_summary(
                    emails_all,
                    project_name,
                    gemini_model,
                    get_gemini_api_key(),
                    max_tokens=openai_max_tokens,
                    max_input_chars=openai_max_input_chars,
                    max_body_chars=openai_body_chars,
                )
                self.record_timing("gemini_assistant", perf_counter() - gemini_start)
                intelligent_summary[project_name]["résumé_assistant"] = build_résumé_assistant_unifié(
                    ASSISTANT_PROVIDER_GEMINI, raw_g
                )
            else:
                logging.warning("assistant_provider inconnu: %s", assistant_provider)
                raw_unk = {
                    "texte": None,
                    "modèle": None,
                    "erreur": f"Fournisseur inconnu: {assistant_provider}",
                }
                intelligent_summary[project_name]["résumé_assistant"] = build_résumé_assistant_unifié(
                    assistant_provider, raw_unk
                )

        return intelligent_summary


def imap_days_for_period_arg(period: Optional[str], days: int) -> int:
    """Jours IMAP : période CLI ou days."""
    if period is not None:
        return imap_days_back_for_period(period)
    return days
