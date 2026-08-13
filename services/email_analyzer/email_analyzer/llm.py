"""Appels LLM (OpenAI, Gemini) pour résumés assistant."""

import json
import logging
import os
import uuid
from typing import Any, Dict, Iterator, List, Optional

from pydantic import BaseModel

from .config import (
    CHAT_INTERACTIVE_MAX_OUTPUT_TOKENS,
    CHAT_MAX_OUTPUT_TOKENS,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OPENAI_BODY_CHARS,
    DEFAULT_OPENAI_MAX_INPUT_CHARS,
    DEFAULT_OPENAI_MAX_TOKENS,
    llm_timeout_seconds,
)

_CHAT_TRUNCATION_NOTICE = (
    "\n\n*(Réponse tronquée par la limite de longueur — réessayez ou demandez un point plus court.)*"
)


def _gemini_candidate_hits_max_tokens(candidate: Any) -> bool:
    fr = getattr(candidate, "finish_reason", None)
    if fr is None:
        return False
    name = getattr(fr, "name", None)
    if name == "MAX_TOKENS":
        return True
    return "MAX_TOKEN" in str(fr).upper()


def get_gemini_api_key() -> Optional[str]:
    """Clé API Gemini : GEMINI_API_KEY ou GOOGLE_API_KEY."""
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def get_openai_api_key() -> Optional[str]:
    """Clé API OpenAI."""
    return os.environ.get("OPENAI_API_KEY")


def select_assistant_provider(preferred: Optional[str]) -> Optional[str]:
    """Choisit "openai" ou "gemini" selon la préférence du client si sa clé
    est configurée, sinon le premier fournisseur disponible. Retourne
    ``None`` si aucune clé n'est configurée — à l'appelant de lever l'erreur
    HTTP adaptée à son contexte (garde `llm.py` indépendant de FastAPI).
    Utilisé par ConversationService et MemoryService pour ne pas dupliquer
    cette logique de sélection."""
    if preferred == "openai" and get_openai_api_key():
        return "openai"
    if preferred == "gemini" and get_gemini_api_key():
        return "gemini"
    if get_gemini_api_key():
        return "gemini"
    if get_openai_api_key():
        return "openai"
    return None


def build_llm_email_corpus(
    emails: List[Dict],
    max_total_chars: int = DEFAULT_OPENAI_MAX_INPUT_CHARS,
    max_body_chars: int = DEFAULT_OPENAI_BODY_CHARS,
) -> str:
    """Agrège sujets et extraits de corps pour les assistants LLM (OpenAI, Gemini)."""
    parts: List[str] = []
    total = 0
    for e in emails:
        subj = (e.get("subject") or "").strip()
        body = (e.get("body") or "").strip()
        excerpt = body[:max_body_chars] if body else ""
        block = f"Sujet: {subj}\nDe: {e.get('from', '')}\nDate: {e.get('date', '')}\n\n{excerpt}"
        if total + len(block) > max_total_chars:
            remain = max_total_chars - total
            if remain > 200:
                parts.append(block[:remain] + "\n[...tronqué]")
            break
        parts.append(block)
        total += len(block) + 4
    return "\n\n---\n\n".join(parts)


# Alias rétrocompatibilité
build_openai_email_corpus = build_llm_email_corpus


def build_llm_thread_corpus(
    threads: List[Dict],
    max_total_chars: int = DEFAULT_OPENAI_MAX_INPUT_CHARS,
    max_body_chars: int = DEFAULT_OPENAI_BODY_CHARS,
) -> str:
    """Même rôle que ``build_llm_email_corpus``, mais un bloc par fil de
    discussion (``project_mail.py::group_emails_by_subject``) plutôt que par
    email brut : réduit le bruit des relances/réponses répétées sur un même
    sujet avant l'appel LLM. Un seul représentant par fil (le message le plus
    récent, ``latest_email``) porte le corps, précédé du nombre de messages et
    de la période couverte."""
    parts: List[str] = []
    total = 0
    for t in threads:
        latest = t.get("latest_email") or {}
        subj = (t.get("subject") or latest.get("subject") or "").strip()
        body = (latest.get("body") or "").strip()
        excerpt = body[:max_body_chars] if body else ""
        count = t.get("count", 1)
        first_date = t.get("first_date")
        last_date = t.get("last_date")
        if first_date and last_date and first_date != last_date:
            span = f", du {first_date} au {last_date}"
        elif last_date:
            span = f", le {last_date}"
        else:
            span = ""
        block = (
            f"Sujet: {subj} ({count} message{'s' if count > 1 else ''}{span})\n"
            f"De: {latest.get('from', '')}\n\n{excerpt}"
        )
        if total + len(block) > max_total_chars:
            remain = max_total_chars - total
            if remain > 200:
                parts.append(block[:remain] + "\n[...tronqué]")
            break
        parts.append(block)
        total += len(block) + 4
    return "\n\n---\n\n".join(parts)


def format_tags_reference_context(tags_reference: Optional[List[Dict]]) -> str:
    """Formate les tags de référence agrégés (``project_mail.py::
    generate_intelligent_summary``) en une ligne de contexte à injecter dans
    le prompt LLM, pour orienter le poids relatif donné à chaque thème."""
    if not tags_reference:
        return ""
    parts = [f"{t.get('tag')} ({t.get('count')})" for t in tags_reference if t.get("tag")]
    if not parts:
        return ""
    return "Tags dominants observés dans ces emails : " + ", ".join(parts) + "."


_EXECUTIVE_SYSTEM_PROMPT = (
    "Tu es le/la chef(fe) de cabinet d'un dirigeant : tu prépares une synthèse "
    "exécutive du suivi de projets par email, pensée pour être lue en moins de "
    "deux minutes mais qui donne une vraie compréhension de la situation (pas "
    "une liste télégraphique). Réponds en français, dans un style dense, "
    "factuel, orienté décision, mais rédigé en phrases complètes qui relient "
    "les faits entre eux (qui a dit quoi, pourquoi ça compte, ce que ça change) "
    "plutôt qu'une simple énumération ; n'invente jamais d'information absente "
    "des emails. Structure impérativement ta réponse en 6 sections numérotées : "
    "1) Où en est ce projet aujourd'hui, en 2-3 phrases racontées (pas une "
    "liste) qui donnent le fil de l'histoire récente. "
    "2) Ce qui a concrètement bougé récemment et pourquoi c'est important. "
    "3) Décisions à prendre ou risques identifiés, avec pour chacun le contexte "
    "qui explique pourquoi il existe. "
    "4) Échéances repérées dans les emails, avec qui est concerné. "
    "5) Ce qu'il manque pour avoir une vision complète (angles morts, "
    "informations attendues mais absentes des échanges). "
    "6) Recommandation d'action priorisée, avec la justification en une phrase."
)


def generate_openai_assistant_summary(
    threads: List[Dict],
    project_name: str,
    model: str,
    api_key: Optional[str],
    max_tokens: int = DEFAULT_OPENAI_MAX_TOKENS,
    max_input_chars: int = DEFAULT_OPENAI_MAX_INPUT_CHARS,
    max_body_chars: int = DEFAULT_OPENAI_BODY_CHARS,
    tags_reference: Optional[List[Dict]] = None,
) -> Dict:
    """
    Résumé exécutif structuré via l'API Chat Completions OpenAI, à partir des
    fils de discussion regroupés par sujet (``threads``, voir
    ``project_mail.py::group_emails_by_subject``) plutôt que des emails bruts.
    """
    result: Dict = {
        "texte": None,
        "modèle": model,
        "erreur": None,
        "max_tokens": max_tokens,
        "max_input_chars": max_input_chars,
        "max_body_chars": max_body_chars,
    }
    if not api_key:
        result["erreur"] = "OPENAI_API_KEY non définie dans l'environnement"
        return result
    if not threads:
        result["erreur"] = "Aucun email à résumer pour cette période"
        return result

    corpus = build_llm_thread_corpus(
        threads,
        max_total_chars=max_input_chars,
        max_body_chars=max_body_chars,
    )
    if not corpus.strip():
        result["erreur"] = "Contenu vide après agrégation"
        return result

    system_prompt = _EXECUTIVE_SYSTEM_PROMPT
    tags_context = format_tags_reference_context(tags_reference)
    user_prompt = (
        f"Projet : {project_name}\n\n"
        + (f"{tags_context}\n\n" if tags_context else "")
        + "Voici les fils de discussion (regroupés par sujet, extraits) à synthétiser :\n\n"
        + corpus
    )

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, timeout=llm_timeout_seconds())
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=max_tokens,
        )
        text = completion.choices[0].message.content
        result["texte"] = text.strip() if text else None
    except Exception as ex:
        logging.warning("Échec appel OpenAI: %s", ex)
        raw = str(ex)
        if "insufficient_quota" in raw:
            result["erreur"] = (
                "Quota ou crédits API OpenAI insuffisants — vérifiez la facturation et les "
                "crédits sur https://platform.openai.com. Vous pouvez réduire la consommation "
                "avec --openai-economy ou désactiver l'appel avec --no-openai. "
                f"Détail technique : {raw}"
            )
        else:
            result["erreur"] = raw

    return result


def generate_gemini_assistant_summary(
    threads: List[Dict],
    project_name: str,
    model: str,
    api_key: Optional[str],
    max_tokens: int = DEFAULT_OPENAI_MAX_TOKENS,
    max_input_chars: int = DEFAULT_OPENAI_MAX_INPUT_CHARS,
    max_body_chars: int = DEFAULT_OPENAI_BODY_CHARS,
    tags_reference: Optional[List[Dict]] = None,
) -> Dict:
    """
    Résumé exécutif structuré via l'API Gemini (google-generativeai), à partir
    des fils de discussion regroupés par sujet (``threads``, voir
    ``project_mail.py::group_emails_by_subject``) plutôt que des emails bruts.
    """
    result: Dict = {
        "texte": None,
        "modèle": model,
        "erreur": None,
        "max_tokens": max_tokens,
        "max_input_chars": max_input_chars,
        "max_body_chars": max_body_chars,
    }
    if not api_key:
        result["erreur"] = (
            "GEMINI_API_KEY ou GOOGLE_API_KEY non définie dans l'environnement "
            "(clé depuis https://aistudio.google.com)"
        )
        return result
    if not threads:
        result["erreur"] = "Aucun email à résumer pour cette période"
        return result

    corpus = build_llm_thread_corpus(
        threads,
        max_total_chars=max_input_chars,
        max_body_chars=max_body_chars,
    )
    if not corpus.strip():
        result["erreur"] = "Contenu vide après agrégation"
        return result

    system_instruction = _EXECUTIVE_SYSTEM_PROMPT
    tags_context = format_tags_reference_context(tags_reference)
    user_prompt = (
        f"Projet : {project_name}\n\n"
        + (f"{tags_context}\n\n" if tags_context else "")
        + "Voici les fils de discussion (regroupés par sujet, extraits) à synthétiser :\n\n"
        + corpus
    )

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        try:
            gen_model = genai.GenerativeModel(
                model_name=model,
                system_instruction=system_instruction,
            )
        except TypeError:
            gen_model = genai.GenerativeModel(model_name=model)
            user_prompt = f"{system_instruction}\n\n{user_prompt}"
        generation_config = genai.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=0.4,
        )
        response = gen_model.generate_content(
            user_prompt,
            generation_config=generation_config,
            request_options={"timeout": llm_timeout_seconds()},
        )
        text_out = None
        if response.candidates:
            try:
                text_out = response.text
            except (ValueError, AttributeError):
                parts = []
                for cand in response.candidates:
                    if cand.content and cand.content.parts:
                        for p in cand.content.parts:
                            if hasattr(p, "text") and p.text:
                                parts.append(p.text)
                text_out = "\n".join(parts) if parts else None
        if text_out:
            result["texte"] = text_out.strip()
        else:
            block_reason = getattr(response.prompt_feedback, "block_reason", None)
            result["erreur"] = (
                "Réponse Gemini vide ou bloquée"
                + (f" (raison: {block_reason})" if block_reason else "")
            )
    except Exception as ex:
        logging.warning("Échec appel Gemini: %s", ex)
        raw = str(ex)
        type_name = type(ex).__name__
        if (
            "ResourceExhausted" in type_name
            or "quota" in raw.lower()
            or "429" in raw
            or ("rate" in raw.lower() and "limit" in raw.lower())
        ):
            result["erreur"] = (
                "Quota ou limite API Gemini atteinte — vérifiez les plafonds sur "
                "https://ai.google.dev/gemini-api/docs/pricing et votre clé sur "
                "https://aistudio.google.com. Réduisez le volume avec --openai-economy "
                "(limites partagées avec le corpus) ou utilisez --assistant-provider none. "
                f"Détail technique : {raw}"
            )
        else:
            result["erreur"] = raw

    return result


_STRUCTURED_EXECUTIVE_SYSTEM_PROMPT = (
    "Tu es le/la chef(fe) de cabinet d'un dirigeant : tu extrais, à partir de fils "
    "de discussion email d'un projet, une synthèse structurée exploitable par une "
    "application (pas un texte à lire). Réponds en français, reste factuel, "
    "n'invente jamais un fait, une date ou un nom absent des emails. Si une "
    "information demandée est absente, laisse la liste correspondante vide plutôt "
    "que d'inventer une valeur plausible. `decisions` = décisions déjà prises "
    "d'après les emails (pas des recommandations). `risques` = risques ou blocages "
    "identifiés. `next_steps` = prochaines actions concrètes à mener, chacune "
    "courte et actionnable. `deadlines` = échéances explicitement mentionnées. "
    "Pour chaque `next_steps`/`deadlines` : `responsable` = la ou les "
    "personnes/organisations concernées si les emails l'indiquent ; `raison` "
    "= pourquoi cette action ou échéance existe, en une phrase, d'après le "
    "contexte des emails (jamais une pure reformulation de `description`) ; "
    "`conseil_prevention` = un conseil concret et actionnable pour éviter que "
    "ce type de situation (retard, oubli, malentendu) ne se reproduise sur ce "
    "projet — laisse `null` si les emails ne donnent pas assez de contexte "
    "pour en proposer un d'utile plutôt que d'inventer un conseil générique. "
    "`llm_risk_level` doit valoir 'FAIBLE', 'MODÉRÉ' ou 'CRITIQUE'. "
    "`appointments` = rendez-vous/réunions explicitement proposés ou confirmés "
    "dans les emails (call, point d'avancement, réunion client) — une entrée "
    "par rendez-vous distinct, liste vide si aucun n'est mentionné. "
    "`probable_next_contact` = UNIQUEMENT si les emails donnent un signal réel "
    "d'une prochaine prise de contact (ex. « je reviens vers vous la semaine "
    "prochaine », un délai de réponse habituel observable dans l'historique du "
    "fil) : une estimation raisonnée de la prochaine date de retour, avec "
    "`reason` qui cite ce signal. Laisse `null` si rien dans les emails ne "
    "permet une estimation ancrée — ne jamais deviner une date sans base "
    "réelle dans le contenu fourni. "
    "`activite_recente` résume spécifiquement les 7 derniers jours calendaires "
    "(section « Activité des 7 derniers jours » du message utilisateur, si "
    "présente) : `evenements` = les faits marquants de cette fenêtre (un email "
    "important, une relance, un changement de statut), chacun avec `resume` en "
    "une phrase et `provenance` (expéditeur ou objet du message). "
    "`synthese_courte` = 2-3 phrases condensant cette activité récente. Si "
    "aucune section « Activité des 7 derniers jours » n'est fournie ou si elle "
    "est vide, retourne `evenements` vide et `synthese_courte` = « Aucune "
    "activité sur cette période. », `periode_couverte` = « 7 derniers jours ». "
    "`recurring_issues` = thèmes RÉCURRENTS observés dans les tickets/réclamations "
    "client de ce fil (sujets portant un motif [Banque_Plateforme_Numéro], ex. « "
    "[BDM TOGO_BDM MOBILE_12002] »), quand au moins 2 tickets distincts de ce fil "
    "pointent vers la même cause racine (ex. « indisponibilité récurrente de la "
    "plateforme », « lenteur des paiements marchand ») — PAS une réénumération "
    "brute de chaque ticket (l'énumération/comptage exact est déjà disponible par "
    "ailleurs, côté application). `occurrences` = nombre de tickets de ce fil "
    "illustrant ce thème, `exemple` = un extrait de sujet représentatif. Liste "
    "vide si aucun thème ne se répète au moins 2 fois."
)


# Pas de valeur par défaut sur aucun champ ci-dessous (y compris `Optional`) :
# le schéma JSON envoyé aux deux fournisseurs doit lister tous les champs comme
# "required" — Gemini (google-generativeai 0.8) rejette explicitement toute
# clé "default" dans un schéma ("Unknown field for Schema: default"), et
# OpenAI Structured Outputs impose la même contrainte en mode strict. Un champ
# `Optional[str]` sans défaut reste correctement "requis mais nullable" pour
# les deux — le LLM doit toujours renvoyer la clé, avec `null` si absent.
class DecisionItem(BaseModel):
    description: str
    date_iso: Optional[str]
    confidence: Optional[str]


class RiskItem(BaseModel):
    description: str
    niveau: Optional[str]
    source: Optional[str]


class NextStepItem(BaseModel):
    description: str
    responsable: Optional[str]
    priorite: Optional[str]
    # Détail exploité par le frontend (fiche détail d'une action, ActionsPage) :
    # pourquoi cette étape est nécessaire, et comment éviter de se retrouver
    # dans la même situation à l'avenir. Optionnels comme le reste du schéma —
    # laissés `null` plutôt qu'inventés si les emails ne donnent pas matière.
    raison: Optional[str]
    conseil_prevention: Optional[str]


class DeadlineItem(BaseModel):
    description: str
    date_iso: Optional[str]
    confiance: Optional[str]
    responsable: Optional[str]
    raison: Optional[str]
    conseil_prevention: Optional[str]


class AppointmentItem(BaseModel):
    description: str
    date_iso: Optional[str]
    participants: List[str]
    confidence: Optional[str]


class ProbableContactItem(BaseModel):
    date_iso: Optional[str]
    confidence: Optional[str]
    reason: str


class RecentActivityItem(BaseModel):
    date_iso: Optional[str]
    resume: str
    provenance: Optional[str]
    importance: Optional[str]


class RecentActivityBlock(BaseModel):
    periode_couverte: str
    evenements: List[RecentActivityItem]
    synthese_courte: str


class TicketPatternItem(BaseModel):
    """Thème récurrent observé à travers plusieurs tickets/réclamations client
    (voir subject_tickets.py) — commentaire qualitatif du LLM, complémentaire
    (pas redondant) à l'énumération/comptage déterministe déjà disponible côté
    application (Email.ticket_label/ticket_number, tools/list_tickets_tool.py)."""

    theme: str
    occurrences: int
    exemple: Optional[str]


class ProjectSummaryLLM(BaseModel):
    """Contrat JSON structuré du résumé projet (rfc-email-pipeline-v2.md §11).

    Additif à `résumé_assistant` (texte libre, inchangé) : ce schéma alimente
    `ProjectSummary.structured_content`/`llm_risk_level` (db/models.py), pas le
    champ `content` existant. `llm_risk_level` ne remplace jamais
    `ai_intelligent.calculate_risk_score` — les deux sont stockés séparément
    (voir analysis_tasks.py::_run_fasttrack_sync). `schema_version` n'est pas
    demandé au LLM : c'est une constante de code (voir `structured_content`
    persisté par `_run_fasttrack_sync`), pas une valeur qu'il doit déduire."""

    synthese: str
    points_cles: List[str]
    decisions: List[DecisionItem]
    risques: List[RiskItem]
    next_steps: List[NextStepItem]
    deadlines: List[DeadlineItem]
    participants: List[str]
    companies: List[str]
    products: List[str]
    sentiment_trend: Optional[str]
    llm_risk_level: Optional[str]
    missing_information: List[str]
    project_updates: List[str]
    appointments: List[AppointmentItem]
    probable_next_contact: Optional[ProbableContactItem]
    activite_recente: RecentActivityBlock
    # Thèmes récurrents à travers les tickets/réclamations client du fil (voir
    # _STRUCTURED_EXECUTIVE_SYSTEM_PROMPT) — qualitatif uniquement, l'énumération/
    # comptage exact reste déterministe côté application (pas re-dérivé ici,
    # _run_fasttrack_sync ne fait que stocker ce champ tel quel dans
    # structured_content, voir migration 015).
    recurring_issues: List[TicketPatternItem]


def _structured_user_prompt(
    threads: List[Dict],
    project_name: str,
    tags_reference: Optional[List[Dict]],
    recent_activity_corpus: Optional[str] = None,
) -> Optional[str]:
    corpus = build_llm_thread_corpus(threads)
    if not corpus.strip():
        return None
    tags_context = format_tags_reference_context(tags_reference)
    return (
        f"Projet : {project_name}\n\n"
        + (f"{tags_context}\n\n" if tags_context else "")
        + "Voici les fils de discussion (regroupés par sujet, extraits) à analyser :\n\n"
        + corpus
        + (
            "\n\nActivité des 7 derniers jours (pour la section activite_recente "
            "uniquement) :\n\n" + recent_activity_corpus
            if recent_activity_corpus and recent_activity_corpus.strip()
            else ""
        )
    )


def extract_structured_project_summary_openai(
    threads: List[Dict],
    project_name: str,
    model: str,
    api_key: Optional[str],
    tags_reference: Optional[List[Dict]] = None,
    recent_activity_corpus: Optional[str] = None,
) -> Dict[str, Any]:
    """Extraction structurée via OpenAI Structured Outputs (schéma
    `ProjectSummaryLLM`). Retourne ``{"data": ProjectSummaryLLM, "erreur": None}``
    ou ``{"data": None, "erreur": "..."}`` — jamais d'exception qui remonte au
    worker arq (même discipline que `generate_openai_assistant_summary`).

    ``recent_activity_corpus`` : corpus pré-filtré aux 7 derniers jours
    calendaires (voir project_mail.py::_generate_project_summary), injecté
    comme section additionnelle du prompt utilisateur pour alimenter
    ``ProjectSummaryLLM.activite_recente`` sans dépendre de l'arithmétique de
    dates du LLM sur tout le corpus."""
    if not api_key:
        return {"data": None, "erreur": "OPENAI_API_KEY non définie dans l'environnement"}
    if not threads:
        return {"data": None, "erreur": "Aucun email à analyser pour cette période"}
    user_prompt = _structured_user_prompt(threads, project_name, tags_reference, recent_activity_corpus)
    if not user_prompt:
        return {"data": None, "erreur": "Contenu vide après agrégation"}

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, timeout=llm_timeout_seconds())
        completion = client.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": _STRUCTURED_EXECUTIVE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            response_format=ProjectSummaryLLM,
        )
        message = completion.choices[0].message
        if message.refusal:
            return {"data": None, "erreur": f"Réponse refusée par le modèle : {message.refusal}"}
        if message.parsed is None:
            return {"data": None, "erreur": "Extraction structurée vide"}
        return {"data": message.parsed, "erreur": None}
    except Exception as ex:
        logging.warning("Échec extraction structurée OpenAI: %s", ex)
        return {"data": None, "erreur": str(ex)}


def extract_structured_project_summary_gemini(
    threads: List[Dict],
    project_name: str,
    model: str,
    api_key: Optional[str],
    tags_reference: Optional[List[Dict]] = None,
    recent_activity_corpus: Optional[str] = None,
) -> Dict[str, Any]:
    """Équivalent Gemini de ``extract_structured_project_summary_openai`` —
    ``response_schema=ProjectSummaryLLM`` (google-generativeai >= 0.8, déjà la
    version installée), une tentative puis retry unique sur échec de
    validation Pydantic (rfc-email-pipeline-v2.md §11). ``recent_activity_corpus``
    : voir docstring de la version OpenAI."""
    if not api_key:
        return {
            "data": None,
            "erreur": "GEMINI_API_KEY ou GOOGLE_API_KEY non définie dans l'environnement",
        }
    if not threads:
        return {"data": None, "erreur": "Aucun email à analyser pour cette période"}
    user_prompt = _structured_user_prompt(threads, project_name, tags_reference, recent_activity_corpus)
    if not user_prompt:
        return {"data": None, "erreur": "Contenu vide après agrégation"}

    try:
        import google.generativeai as genai
        from pydantic import ValidationError

        genai.configure(api_key=api_key)
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=ProjectSummaryLLM,
            temperature=0.2,
        )
        try:
            gen_model = genai.GenerativeModel(
                model_name=model,
                system_instruction=_STRUCTURED_EXECUTIVE_SYSTEM_PROMPT,
            )
            prompt = user_prompt
        except TypeError:
            gen_model = genai.GenerativeModel(model_name=model)
            prompt = f"{_STRUCTURED_EXECUTIVE_SYSTEM_PROMPT}\n\n{user_prompt}"

        last_error: Optional[str] = None
        for attempt in range(2):
            response = gen_model.generate_content(
                prompt,
                generation_config=generation_config,
                request_options={"timeout": llm_timeout_seconds()},
            )
            if not response.candidates:
                block_reason = getattr(response.prompt_feedback, "block_reason", None)
                last_error = "Réponse Gemini vide ou bloquée" + (
                    f" (raison: {block_reason})" if block_reason else ""
                )
                continue
            try:
                data = ProjectSummaryLLM.model_validate_json(response.text)
                return {"data": data, "erreur": None}
            except (ValidationError, ValueError) as ve:
                last_error = f"Sortie JSON non conforme au schéma : {ve}"
                continue
        return {"data": None, "erreur": last_error or "Échec d'extraction structurée"}
    except Exception as ex:
        logging.warning("Échec extraction structurée Gemini: %s", ex)
        return {"data": None, "erreur": str(ex)}


def serialize_analysis_for_chat(analysis: Dict[str, Any], max_chars: int) -> str:
    """Sérialise le bloc d'analyse JSON avec troncature si nécessaire."""
    try:
        s = json.dumps(analysis, ensure_ascii=False)
    except (TypeError, ValueError):
        s = str(analysis)
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 30] + "\n...[contexte tronqué]"


def build_project_chat_system_prompt(project_name: str, emails_corpus: str) -> str:
    """
    Prompt système pour l'assistant conversationnel : efficacité et priorisation
    (inspiré des principes de productivité associés à « La semaine de 4 heures »),
    sans citation marketing ni reproduction de texte protégé.
    """
    return (
        "Tu es un coach de synthèse pour le suivi de projets professionnels par email. "
        "Tu t'inspires d'une approche orientée impact : réduire le bruit, prioriser ce qui "
        "change vraiment la situation, décider vite avec peu d'informations fiables, et proposer "
        "déléguer ou automatiser lorsque c'est pertinent — sans surcharger l'utilisateur.\n"
        "Réponds toujours en français, ton professionnel et direct.\n\n"
        "Contexte : extraits des derniers emails du filtre projet (boîte IMAP). "
        "Ne confonds pas avec une analyse structurée : base-toi uniquement sur ce texte ; "
        "s'il est incomplet ou ambigu, dis-le.\n"
        f"Projet (filtre) : {project_name}\n\n"
        f"{emails_corpus}\n\n"
        "Règles :\n"
        "- Résume et conseille à partir de ces messages ; n'invente pas de faits absents des extraits.\n"
        "- Quand il manque des éléments pour adapter le résumé au lecteur, pose des questions "
        "ciblées (rôle, temps disponible, priorité métier, niveau de détail souhaité).\n"
        "- Quand tu as assez d'éléments, fournis un résumé actionnable : faits clés, risques, "
        "prochaines étapes courtes et ordonnées.\n"
        "- Reste clair et structuré, sans redites ; utilise des listes à puces si utile."
    )


def build_portfolio_context(projects: List[Dict[str, Any]], total_count: Optional[int] = None) -> str:
    """Formate la liste des projets d'un tenant (nom/sentiment/actions en
    attente/résumé) en texte de contexte pour l'assistant portefeuille — voir
    ``build_portfolio_chat_system_prompt``. Chaque entrée de ``projects`` est
    un dict simple (pas un modèle ORM) construit par l'appelant
    (``api/routers/assistant.py``), pour garder ce module sans dépendance DB.

    ``total_count`` : nombre réel de projets du tenant (avant troncature à
    ``_MAX_PROJECTS_IN_CONTEXT`` côté ``ContextService``). Si plus grand que
    ``len(projects)``, on le signale explicitement dans le texte — sans ça le
    LLM traite cette liste comme le portefeuille complet et répond "je n'ai
    pas d'information sur ce projet" pour un projet réel simplement absent
    des N plus récemment mis à jour (voir search_projects dans les règles du
    system prompt pour le filet de sécurité correspondant)."""
    if not projects:
        return "Aucun projet actif pour le moment."
    blocks = []
    for p in projects:
        lines = [f"- {p.get('name')}"]
        if p.get("sentiment"):
            lines.append(f"  Statut : {p['sentiment']}")
        pending_descriptions = p.get("pending_action_descriptions") or []
        if pending_descriptions:
            lines.append("  Actions en attente :")
            lines.extend(f"    - {d}" for d in pending_descriptions)
        if p.get("open_ticket_count"):
            lines.append(f"  Tickets/réclamations détectés : {p['open_ticket_count']}")
        if p.get("document_count"):
            lines.append(f"  Documents disponibles : {p['document_count']}")
        if p.get("summary_content"):
            lines.append(f"  Résumé : {p['summary_content']}")
        blocks.append("\n".join(lines))
    text = "\n\n".join(blocks)
    if total_count is not None and total_count > len(projects):
        text += (
            f"\n\n(Cette liste ne montre que les {len(projects)} projets les plus "
            f"récemment mis à jour sur {total_count} au total pour ce compte. Si "
            "l'utilisateur mentionne un projet absent d'ici, appelle l'outil "
            "search_projects avec son nom avant de conclure que tu n'as pas "
            "d'information à son sujet.)"
        )
    return text


def build_memory_context(memories: List[Dict[str, Any]]) -> str:
    """Formate les souvenirs (Memory, voir memory_service.py) en bloc de
    contexte. ``memories`` : dicts simples ``{"category", "content"}``
    (même convention que ``build_portfolio_context`` : pas de modèle ORM
    ici pour garder ce module indépendant de la DB). Chaîne vide si aucun
    souvenir — à l'appelant de ne pas insérer de section vide dans le
    prompt."""
    if not memories:
        return ""
    lines = ["Souvenirs retenus sur l'utilisateur (préférences, faits durables) :"]
    lines.extend(f"- ({m.get('category')}) {m.get('content')}" for m in memories)
    return "\n".join(lines)


def build_knowledge_context(contacts: List[Dict[str, Any]]) -> str:
    """Formate les contacts connus (Contact/Organization, voir
    knowledge_service.py) en bloc de contexte. ``contacts`` : dicts simples
    ``{"name", "organization"}`` (``organization`` optionnel)."""
    if not contacts:
        return ""
    lines = ["Contacts connus (extraits des projets suivis) :"]
    for c in contacts:
        org = f" — {c['organization']}" if c.get("organization") else ""
        lines.append(f"- {c.get('name')}{org}")
    return "\n".join(lines)


def build_profile_context(profile: Optional[Dict[str, Any]]) -> str:
    """Formate le profil personnel et les habitudes de l'utilisateur
    (Profile, voir profile_service.py — domaine "Contexte Personnel") en
    bloc de contexte. ``profile`` : dict simple (mêmes conventions que
    ``build_memory_context``/``build_knowledge_context``) ou ``None``/vide
    si l'utilisateur n'a encore rien renseigné — chaîne vide dans ce cas,
    à l'appelant de ne pas insérer de section vide dans le prompt."""
    if not profile:
        return ""
    lines: List[str] = []
    if profile.get("display_name"):
        lines.append(f"- Nom : {profile['display_name']}")
    if profile.get("languages"):
        lines.append(f"- Langues : {', '.join(profile['languages'])}")
    if profile.get("timezone"):
        lines.append(f"- Fuseau horaire : {profile['timezone']}")
    if profile.get("location"):
        lines.append(f"- Localisation : {profile['location']}")
    if profile.get("work_start_time") and profile.get("work_end_time"):
        lines.append(
            f"- Horaires de travail habituels : {profile['work_start_time']}-"
            f"{profile['work_end_time']}. Ne propose pas de réunion en dehors de "
            "cette plage sauf demande explicite."
        )
    if profile.get("wake_time"):
        lines.append(f"- Heure de réveil habituelle : {profile['wake_time']}")
    if profile.get("sleep_time"):
        lines.append(f"- Heure de coucher habituelle : {profile['sleep_time']}")
    if profile.get("sport_days"):
        lines.append(f"- Jours de sport : {', '.join(profile['sport_days'])}")
    if profile.get("focus_hours"):
        lines.append(f"- Plage de concentration préférée : {profile['focus_hours']}")
    if profile.get("family_members"):
        family = ", ".join(
            f"{m.get('name')} ({m['relation']})" if m.get("relation") else str(m.get("name"))
            for m in profile["family_members"]
        )
        lines.append(f"- Famille : {family}")
    if profile.get("preferences"):
        prefs = ", ".join(f"{k} : {v}" for k, v in profile["preferences"].items())
        lines.append(f"- Préférences : {prefs}")
    if not lines:
        return ""
    return "Contexte personnel de l'utilisateur :\n" + "\n".join(lines)


def build_objectives_context(objectives: List[Dict[str, Any]]) -> str:
    """Formate les objectifs personnels/professionnels (Objective/Milestone,
    voir objective_service.py — domaine "Contexte Objectifs") en bloc de
    contexte. ``objectives`` : dicts simples ``{"title", "status",
    "blocking_reason", "target_date", "percent", "next_milestone"}`` (mêmes
    conventions que ``build_memory_context``/``build_knowledge_context`` —
    pas de modèle ORM ici). Chaîne vide si aucun objectif, à l'appelant de
    ne pas insérer de section vide dans le prompt."""
    if not objectives:
        return ""
    lines = ["Objectifs suivis par l'utilisateur :"]
    for o in objectives:
        status = o.get("status")
        marker = " (BLOQUÉ)" if status == "blocked" else ""
        lines.append(f"- {o.get('title')}{marker} — progression {o.get('percent', 0)}%")
        if status == "blocked" and o.get("blocking_reason"):
            lines.append(f"  Raison du blocage : {o['blocking_reason']}")
        if o.get("target_date"):
            lines.append(f"  Échéance visée : {o['target_date']}")
        if o.get("next_milestone"):
            lines.append(f"  Prochain jalon : {o['next_milestone']}")
    return "\n".join(lines)


def build_portfolio_chat_system_prompt(
    projects_context: str, extra_context_sections: Optional[List[str]] = None
) -> str:
    """Prompt système de l'assistant permanent (portefeuille) : contrairement à
    ``build_project_chat_system_prompt``, le contexte n'est pas le corpus brut
    d'UN projet mais l'état agrégé de TOUS les projets du tenant
    (``build_portfolio_context``) — l'assistant répond sur l'ensemble du
    portefeuille, pas sur un seul filtre projet.

    ``extra_context_sections`` (Context Engine, Phase 7) : blocs de contexte
    additionnels déjà formatés (``build_memory_context``,
    ``build_knowledge_context``, ...), assemblés par
    ``email_analyzer.services.context_service.ContextService`` — ce module
    reste le seul à savoir formater du texte de prompt, la logique
    d'agrégation (quelles données lire) vit dans le service."""
    extra_blocks = "\n\n".join(s for s in (extra_context_sections or []) if s)
    extra_section = f"\n\n{extra_blocks}" if extra_blocks else ""
    return (
        "Tu es l'assistante personnelle d'un chef de projet : tu l'aides à piloter "
        "l'ensemble de son portefeuille de projets suivis par email, pas un seul "
        "projet à la fois. Réponds toujours en français, ton direct et professionnel, "
        "comme une secrétaire de direction qui connaît le dossier sur le bout des "
        "doigts et anticipe ses besoins.\n\n"
        "Contexte : état actuel de chaque projet du portefeuille (nom, statut, "
        "actions en attente, résumé). Ne confonds pas avec une analyse email brute ; "
        "base-toi uniquement sur ce texte, et dis-le si une information manque plutôt "
        "que de l'inventer.\n\n"
        f"{projects_context}"
        f"{extra_section}\n\n"
        "Règles :\n"
        "- Réponds aux questions sur l'état des projets, les risques, les échéances et "
        "les actions en attente à partir de ce contexte.\n"
        "- Le contexte projets ci-dessus ne liste que les projets les plus récemment "
        "mis à jour (voir note de troncature éventuelle) : si l'utilisateur nomme un "
        "projet précis absent de cette liste, appelle d'abord search_projects avec ce "
        "nom (ou le mot-clé le plus proche, ex. un domaine email) avant de conclure "
        "quoi que ce soit — ne dis jamais que tu n'as pas d'information sur un projet "
        "sans avoir vérifié via cet outil. L'utilisateur peut désigner un projet par "
        "son nom affiché OU par un domaine expéditeur/client/société associé — les "
        "deux fonctionnent, essaie donc aussi ce mot-clé même s'il ne ressemble pas "
        "au nom du projet.\n"
        "- Si, après vérification via les outils, la question porte sur un projet ou "
        "un détail réellement absent, dis-le clairement plutôt que de deviner : "
        "n'invente jamais une date, un montant, un nom de personne ou un contenu "
        "d'email qui ne figure pas explicitement dans ce contexte ou dans un résultat "
        "d'outil.\n"
        "- Reste concis et actionnable ; utilise des listes à puces si utile.\n"
        "- Sois proactive : quand c'est pertinent pour la question posée, signale "
        "de toi-même une échéance proche, un objectif bloqué ou des tickets ouverts "
        "présents dans le contexte, même si l'utilisateur ne les mentionne pas "
        "explicitement — comme le ferait une assistante de direction qui prévient "
        "avant qu'on ait à demander.\n\n"
        "Tu disposes déjà, ci-dessus, d'un contexte résumé (projets, mémoire, "
        "contacts, objectifs). Réponds directement à partir de ce contexte quand il "
        "suffit. N'appelle un outil que lorsque ce contexte résumé est insuffisant "
        "pour répondre précisément : détail ou contenu d'un email récent, ou "
        "recherche par mot-clé dans le sujet/corps des emails d'un projet "
        "(search_emails), liste de rendez-vous (list_meetings), résumé projet à jour "
        "(summarize_project), où en était un projet à une date passée ou son "
        "évolution dans le temps plutôt que de te fier au seul résumé courant "
        "(search_project_summaries), détail des jalons d'un objectif "
        "(list_objectives), tickets/réclamations (list_tickets), documents "
        "disponibles et leur contenu (search_documents), génération d'un brouillon "
        "de réponse (draft_reply)."
    )


def _normalize_chat_messages(messages: List[Dict[str, str]]) -> Optional[List[Dict[str, str]]]:
    """Filtre rôles user/assistant ; retourne None si invalide."""
    out: List[Dict[str, str]] = []
    for m in messages:
        role = (m.get("role") or "").strip()
        content = (m.get("content") or "").strip()
        if role not in ("user", "assistant"):
            continue
        if not content:
            return None
        out.append({"role": role, "content": content})
    if not out or out[-1]["role"] != "user":
        return None
    return out


def _gemini_history_from_messages(prior: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Convertit les tours précédents au format attendu par start_chat Gemini."""
    history: List[Dict[str, Any]] = []
    for m in prior:
        if m["role"] == "user":
            history.append({"role": "user", "parts": [m["content"]]})
        else:
            history.append({"role": "model", "parts": [m["content"]]})
    return history


def _gemini_history_with_embedded_system(
    system_instruction: str,
    prior: List[Dict[str, str]],
    last_user: str,
) -> tuple[List[Dict[str, Any]], str]:
    """Sans system_instruction sur le modèle : injecte le système dans le premier message utilisateur."""
    history: List[Dict[str, Any]] = []
    first_user_done = False
    for m in prior:
        if m["role"] == "user":
            c = m["content"]
            if not first_user_done:
                c = f"{system_instruction}\n\n{c}"
                first_user_done = True
            history.append({"role": "user", "parts": [c]})
        else:
            history.append({"role": "model", "parts": [m["content"]]})
    if not first_user_done:
        last_user = f"{system_instruction}\n\n{last_user}"
    return history, last_user


def project_assistant_chat_openai(
    project_name: str,
    recent_emails: List[Dict[str, Any]],
    messages: List[Dict[str, str]],
    model: str,
    api_key: Optional[str],
    max_context_chars: int = DEFAULT_OPENAI_MAX_INPUT_CHARS,
    max_tokens: int = CHAT_MAX_OUTPUT_TOKENS,
) -> Dict[str, Any]:
    """Un tour de conversation via Chat Completions (OpenAI)."""
    result: Dict[str, Any] = {"message": None, "erreur": None, "modèle": model}
    norm = _normalize_chat_messages(messages)
    if not norm:
        result["erreur"] = (
            "Historique invalide : au moins un message utilisateur non vide, "
            "le dernier message doit être du rôle utilisateur."
        )
        return result
    if not api_key:
        result["erreur"] = "OPENAI_API_KEY non définie dans l'environnement"
        return result

    corpus_budget = max(4000, max_context_chars - 6000)
    emails_corpus = build_llm_email_corpus(
        recent_emails,
        max_total_chars=corpus_budget,
        max_body_chars=DEFAULT_OPENAI_BODY_CHARS,
    )
    if not emails_corpus.strip():
        result["erreur"] = "Aucun contenu d'email exploitable pour le contexte assistant."
        return result
    system_prompt = build_project_chat_system_prompt(project_name, emails_corpus)
    api_messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    api_messages.extend(norm)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, timeout=llm_timeout_seconds())
        completion = client.chat.completions.create(
            model=model,
            messages=api_messages,
            temperature=0.45,
            max_tokens=max_tokens,
        )
        choice = completion.choices[0]
        text = choice.message.content
        msg = text.strip() if text else None
        if msg and getattr(choice, "finish_reason", None) == "length":
            msg = msg + _CHAT_TRUNCATION_NOTICE
        result["message"] = msg
        if not result["message"]:
            result["erreur"] = "Réponse OpenAI vide"
    except Exception as ex:
        logging.warning("Échec chat OpenAI: %s", ex)
        raw = str(ex)
        if "insufficient_quota" in raw:
            result["erreur"] = (
                "Quota ou crédits API OpenAI insuffisants — vérifiez la facturation sur "
                "https://platform.openai.com. "
                f"Détail : {raw}"
            )
        else:
            result["erreur"] = raw

    return result


def project_assistant_chat_gemini(
    project_name: str,
    recent_emails: List[Dict[str, Any]],
    messages: List[Dict[str, str]],
    model: str,
    api_key: Optional[str],
    max_context_chars: int = DEFAULT_OPENAI_MAX_INPUT_CHARS,
    max_tokens: int = CHAT_MAX_OUTPUT_TOKENS,
) -> Dict[str, Any]:
    """Un tour de conversation via chat Gemini."""
    result: Dict[str, Any] = {"message": None, "erreur": None, "modèle": model}
    norm = _normalize_chat_messages(messages)
    if not norm:
        result["erreur"] = (
            "Historique invalide : au moins un message utilisateur non vide, "
            "le dernier message doit être du rôle utilisateur."
        )
        return result
    if not api_key:
        result["erreur"] = (
            "GEMINI_API_KEY ou GOOGLE_API_KEY non définie dans l'environnement "
            "(clé depuis https://aistudio.google.com)"
        )
        return result

    corpus_budget = max(4000, max_context_chars - 6000)
    emails_corpus = build_llm_email_corpus(
        recent_emails,
        max_total_chars=corpus_budget,
        max_body_chars=DEFAULT_OPENAI_BODY_CHARS,
    )
    if not emails_corpus.strip():
        result["erreur"] = "Aucun contenu d'email exploitable pour le contexte assistant."
        return result
    system_instruction = build_project_chat_system_prompt(project_name, emails_corpus)
    prior = norm[:-1]
    last_user = norm[-1]["content"]

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        try:
            gen_model = genai.GenerativeModel(
                model_name=model,
                system_instruction=system_instruction,
            )
            history = _gemini_history_from_messages(prior)
        except TypeError:
            gen_model = genai.GenerativeModel(model_name=model)
            history, last_user = _gemini_history_with_embedded_system(
                system_instruction, prior, last_user
            )

        generation_config = genai.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=0.45,
        )
        chat = gen_model.start_chat(history=history)
        response = chat.send_message(
            last_user,
            generation_config=generation_config,
            request_options={"timeout": llm_timeout_seconds()},
        )

        text_out = None
        truncated_max_tokens = False
        if response.candidates:
            truncated_max_tokens = _gemini_candidate_hits_max_tokens(response.candidates[0])
            try:
                text_out = response.text
            except (ValueError, AttributeError):
                parts: List[str] = []
                for cand in response.candidates:
                    if cand.content and cand.content.parts:
                        for p in cand.content.parts:
                            if hasattr(p, "text") and p.text:
                                parts.append(p.text)
                text_out = "\n".join(parts) if parts else None
        if text_out:
            msg = text_out.strip()
            if truncated_max_tokens and msg:
                msg = msg + _CHAT_TRUNCATION_NOTICE
            result["message"] = msg
        else:
            block_reason = getattr(response.prompt_feedback, "block_reason", None)
            result["erreur"] = (
                "Réponse Gemini vide ou bloquée"
                + (f" (raison: {block_reason})" if block_reason else "")
            )
    except Exception as ex:
        logging.warning("Échec chat Gemini: %s", ex)
        raw = str(ex)
        type_name = type(ex).__name__
        if (
            "ResourceExhausted" in type_name
            or "quota" in raw.lower()
            or "429" in raw
            or ("rate" in raw.lower() and "limit" in raw.lower())
        ):
            result["erreur"] = (
                "Quota ou limite API Gemini atteinte — vérifiez les plafonds sur "
                "https://ai.google.dev/gemini-api/docs/pricing. "
                f"Détail : {raw}"
            )
        else:
            result["erreur"] = raw

    return result


# --- Chat unifié streaming + tool-calling natif (fusion Agent/Conversation) ---
# Chat portefeuille persistant : le "tour" générique ci-dessous étend le simple
# {"role","content"} avec des tours "tool" et des tool_calls assistant, pour
# que le modèle choisisse lui-même quand interroger les Tools
# (email_analyzer/services/tool_executor.py) plutôt qu'une sélection
# déterministe par mots-clés (ancien email_analyzer/agent/planner.py,
# supprimé). Générateurs : jamais d'exception qui remonte à l'appelant (même
# discipline que le reste de ce module) — la boucle appelante
# (ConversationService.stream_message) consomme des événements
# {"type": "token"|"tool_calls"|"done"|"error", ...}.
#
# GenericTurn = {"role": "user"|"assistant"|"tool",
#                "content": Optional[str],
#                "tool_calls": Optional[List[{"id","name","arguments": dict}]],  # assistant seulement
#                "tool_call_id": Optional[str], "name": Optional[str]}           # tool seulement


def _openai_wire_messages(
    system_prompt: str, turns: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Traduit une liste de GenericTurn vers le format `messages` attendu par
    l'API Chat Completions d'OpenAI (rôle `tool` + `tool_calls` assistant
    inclus)."""
    wire: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for turn in turns:
        role = turn.get("role")
        if role == "assistant" and turn.get("tool_calls"):
            wire.append(
                {
                    "role": "assistant",
                    "content": turn.get("content"),
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": json.dumps(call.get("arguments") or {}, ensure_ascii=False),
                            },
                        }
                        for call in turn["tool_calls"]
                    ],
                }
            )
        elif role == "tool":
            wire.append(
                {
                    "role": "tool",
                    "tool_call_id": turn.get("tool_call_id"),
                    "name": turn.get("name"),
                    "content": turn.get("content") or "",
                }
            )
        else:
            wire.append({"role": role, "content": turn.get("content") or ""})
    return wire


def build_openai_tool_specs(tool_specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """``tool_specs`` : format provider-agnostique
    ``[{"name","description","parameters": <JSON Schema>}]`` (voir
    ``email_analyzer.services.tool_executor.TOOL_SPECS``) -> enveloppe
    ``{"type":"function","function":...}`` attendue par le paramètre `tools`
    de l'API OpenAI."""
    return [
        {
            "type": "function",
            "function": {
                "name": spec["name"],
                "description": spec["description"],
                "parameters": spec["parameters"],
            },
        }
        for spec in tool_specs
    ]


def stream_portfolio_assistant_chat_openai(
    system_prompt: str,
    turns: List[Dict[str, Any]],
    model: str,
    api_key: Optional[str],
    tools: Optional[List[Dict[str, Any]]] = None,
    max_tokens: int = CHAT_INTERACTIVE_MAX_OUTPUT_TOKENS,
    timeout: Optional[float] = None,
) -> Iterator[Dict[str, Any]]:
    """Un tour de conversation avec l'assistant portefeuille (OpenAI),
    streaming + tool-calling natif. ``turns`` : liste de GenericTurn
    (historique déjà persisté + tours assistant/tool ajoutés en cours de
    boucle par l'appelant). ``tools`` : format provider-agnostique
    ``[{"name","description","parameters"}]`` (voir
    ``email_analyzer.services.tool_executor.TOOL_SPECS``) — converti ici en
    interne vers le format `tools` d'OpenAI (``build_openai_tool_specs``),
    pour que l'appelant n'ait pas à connaître le format propre à chaque
    fournisseur. Ne lève jamais d'exception : toute erreur devient un
    événement terminal ``{"type": "error", "erreur": ...}``."""
    if not turns:
        yield {"type": "error", "erreur": "Historique invalide : aucun tour à envoyer."}
        return
    if not api_key:
        yield {"type": "error", "erreur": "OPENAI_API_KEY non définie dans l'environnement"}
        return

    try:
        from openai import OpenAI

        wire_tools = build_openai_tool_specs(tools) if tools else None
        client = OpenAI(api_key=api_key, timeout=timeout or llm_timeout_seconds())
        stream = client.chat.completions.create(
            model=model,
            messages=_openai_wire_messages(system_prompt, turns),
            temperature=0.45,
            max_tokens=max_tokens,
            tools=wire_tools,
            tool_choice="auto" if wire_tools else None,
            stream=True,
        )

        text_parts: List[str] = []
        tool_call_fragments: Dict[int, Dict[str, Any]] = {}
        finish_reason: Optional[str] = None

        for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason
            delta = choice.delta
            if delta and delta.content:
                text_parts.append(delta.content)
                yield {"type": "token", "delta": delta.content}
            if delta and delta.tool_calls:
                for tc in delta.tool_calls:
                    frag = tool_call_fragments.setdefault(
                        tc.index, {"id": None, "name": None, "arguments": ""}
                    )
                    if tc.id:
                        frag["id"] = tc.id
                    if tc.function and tc.function.name:
                        frag["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        frag["arguments"] += tc.function.arguments

        if tool_call_fragments and finish_reason == "tool_calls":
            calls = []
            for frag in tool_call_fragments.values():
                try:
                    arguments = json.loads(frag["arguments"]) if frag["arguments"] else {}
                except (json.JSONDecodeError, TypeError) as ex:
                    logging.warning(
                        "Arguments d'outil OpenAI invalides (%s): %s", frag.get("name"), ex
                    )
                    arguments = {}
                calls.append({"id": frag["id"], "name": frag["name"], "arguments": arguments})
            yield {"type": "tool_calls", "calls": calls}
            return

        message = "".join(text_parts).strip()
        if message and finish_reason == "length":
            message = message + _CHAT_TRUNCATION_NOTICE
        if not message:
            yield {"type": "error", "erreur": "Réponse OpenAI vide"}
            return
        yield {"type": "done", "message": message, "modèle": model}
    except Exception as ex:
        logging.warning("Échec chat streaming OpenAI: %s", ex)
        raw = str(ex)
        if "insufficient_quota" in raw:
            raw = (
                "Quota ou crédits API OpenAI insuffisants — vérifiez la facturation sur "
                "https://platform.openai.com. "
                f"Détail : {raw}"
            )
        yield {"type": "error", "erreur": raw}


def _gemini_tools_param(
    tool_specs: Optional[List[Dict[str, Any]]]
) -> Optional[List[Dict[str, Any]]]:
    """Adapte le même format provider-agnostique que ``build_openai_tool_specs``
    vers ``{"function_declarations": [...]}`` attendu par google-generativeai ;
    les valeurs ``type`` du JSON Schema sont mises en MAJUSCULES (OBJECT,
    STRING, INTEGER, NUMBER, BOOLEAN, ARRAY) comme l'exige le proto Schema de
    google-generativeai 0.8.x."""
    if not tool_specs:
        return None

    def _upper_types(schema: Any) -> Any:
        if isinstance(schema, dict):
            out = {k: _upper_types(v) for k, v in schema.items()}
            if isinstance(out.get("type"), str):
                out["type"] = out["type"].upper()
            return out
        if isinstance(schema, list):
            return [_upper_types(v) for v in schema]
        return schema

    return [
        {
            "function_declarations": [
                {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": _upper_types(spec["parameters"]),
                }
                for spec in tool_specs
            ]
        }
    ]


def _gemini_history_with_tools(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Traduit des GenericTurn (y compris tours ``tool`` et
    ``assistant.tool_calls``) en historique Gemini — distinct de
    ``_gemini_history_from_messages`` ci-dessus, qui ne gère que du texte
    simple sans tool-calling."""
    history: List[Dict[str, Any]] = []
    for turn in turns:
        role = turn.get("role")
        if role == "assistant" and turn.get("tool_calls"):
            history.append(
                {
                    "role": "model",
                    "parts": [
                        {"function_call": {"name": call["name"], "args": call.get("arguments") or {}}}
                        for call in turn["tool_calls"]
                    ],
                }
            )
        elif role == "tool":
            history.append(
                {
                    "role": "function",
                    "parts": [
                        {
                            "function_response": {
                                "name": turn.get("name"),
                                "response": {"result": turn.get("content") or ""},
                            }
                        }
                    ],
                }
            )
        elif role == "user":
            history.append({"role": "user", "parts": [turn.get("content") or ""]})
        else:
            history.append({"role": "model", "parts": [turn.get("content") or ""]})
    return history


def _gemini_last_turn_payload(turn: Dict[str, Any]) -> Any:
    """Payload à passer à ``chat.send_message`` pour le dernier tour de
    ``turns`` : texte simple pour un tour utilisateur, partie
    ``function_response`` pour un tour outil (cas d'un rappel du modèle après
    exécution des tools, au sein de la même boucle agentique)."""
    if turn.get("role") == "tool":
        return [
            {
                "function_response": {
                    "name": turn.get("name"),
                    "response": {"result": turn.get("content") or ""},
                }
            }
        ]
    return turn.get("content") or ""


def stream_portfolio_assistant_chat_gemini(
    system_prompt: str,
    turns: List[Dict[str, Any]],
    model: str,
    api_key: Optional[str],
    tools: Optional[List[Dict[str, Any]]] = None,
    max_tokens: int = CHAT_INTERACTIVE_MAX_OUTPUT_TOKENS,
    timeout: Optional[float] = None,
) -> Iterator[Dict[str, Any]]:
    """Équivalent Gemini de ``stream_portfolio_assistant_chat_openai``."""
    if not turns:
        yield {"type": "error", "erreur": "Historique invalide : aucun tour à envoyer."}
        return
    if not api_key:
        yield {
            "type": "error",
            "erreur": (
                "GEMINI_API_KEY ou GOOGLE_API_KEY non définie dans l'environnement "
                "(clé depuis https://aistudio.google.com)"
            ),
        }
        return

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        try:
            gen_model = genai.GenerativeModel(
                model_name=model,
                system_instruction=system_prompt,
                tools=_gemini_tools_param(tools),
            )
        except TypeError:
            logging.warning(
                "SDK google-generativeai trop ancien pour system_instruction/tools "
                "natifs — repli sans tool-calling pour cet appel."
            )
            gen_model = genai.GenerativeModel(model_name=model)
            plain_prior = [
                t for t in turns[:-1] if t.get("role") in ("user", "assistant") and not t.get("tool_calls")
            ]
            last_user = turns[-1].get("content") or "" if turns[-1].get("role") == "user" else ""
            history, last_payload = _gemini_history_with_embedded_system(
                system_prompt, plain_prior, last_user
            )
        else:
            history = _gemini_history_with_tools(turns[:-1])
            last_payload = _gemini_last_turn_payload(turns[-1])

        generation_config = genai.GenerationConfig(max_output_tokens=max_tokens, temperature=0.45)
        chat = gen_model.start_chat(history=history)
        response_stream = chat.send_message(
            last_payload,
            stream=True,
            generation_config=generation_config,
            request_options={"timeout": timeout or llm_timeout_seconds()},
        )

        text_parts: List[str] = []
        calls: List[Dict[str, Any]] = []
        truncated_max_tokens = False
        last_chunk = None

        for chunk in response_stream:
            last_chunk = chunk
            if not chunk.candidates:
                continue
            candidate = chunk.candidates[0]
            if _gemini_candidate_hits_max_tokens(candidate):
                truncated_max_tokens = True
            if not candidate.content or not candidate.content.parts:
                continue
            for part in candidate.content.parts:
                text = getattr(part, "text", None)
                if text:
                    text_parts.append(text)
                    yield {"type": "token", "delta": text}
                fc = getattr(part, "function_call", None)
                if fc is not None and getattr(fc, "name", None):
                    try:
                        arguments = dict(fc.args) if fc.args else {}
                    except (TypeError, ValueError) as ex:
                        logging.warning("Arguments d'outil Gemini invalides (%s): %s", fc.name, ex)
                        arguments = {}
                    calls.append({"id": str(uuid.uuid4()), "name": fc.name, "arguments": arguments})

        if calls:
            yield {"type": "tool_calls", "calls": calls}
            return

        message = "".join(text_parts).strip()
        if message and truncated_max_tokens:
            message = message + _CHAT_TRUNCATION_NOTICE
        if not message:
            block_reason = getattr(getattr(last_chunk, "prompt_feedback", None), "block_reason", None)
            yield {
                "type": "error",
                "erreur": "Réponse Gemini vide ou bloquée"
                + (f" (raison: {block_reason})" if block_reason else ""),
            }
            return
        yield {"type": "done", "message": message, "modèle": model}
    except Exception as ex:
        logging.warning("Échec chat streaming Gemini: %s", ex)
        raw = str(ex)
        type_name = type(ex).__name__
        if (
            "ResourceExhausted" in type_name
            or "quota" in raw.lower()
            or "429" in raw
            or ("rate" in raw.lower() and "limit" in raw.lower())
        ):
            raw = (
                "Quota ou limite API Gemini atteinte — vérifiez les plafonds sur "
                "https://ai.google.dev/gemini-api/docs/pricing. "
                f"Détail : {raw}"
            )
        yield {"type": "error", "erreur": raw}


def build_résumé_assistant_unifié(fournisseur: str, raw: Dict) -> Dict:
    """Construit l'objet résumé_assistant à partir du retour OpenAI ou Gemini."""
    return {
        "fournisseur": fournisseur,
        "texte": raw.get("texte"),
        "modèle": raw.get("modèle"),
        "erreur": raw.get("erreur"),
        "max_tokens": raw.get("max_tokens"),
        "max_input_chars": raw.get("max_input_chars"),
        "max_body_chars": raw.get("max_body_chars"),
    }


# --- Memory extraction (Personal Executive AI Assistant pivot, Phase 5) ---
# Même discipline que extract_structured_project_summary_openai/_gemini
# ci-dessus : jamais d'exception qui remonte à l'appelant, toujours
# {"data": ..., "erreur": ...}. Consommé uniquement par
# email_analyzer.services.memory_service.MemoryService.


class MemoryFactLLM(BaseModel):
    content: str
    category: str  # "preference" | "fact" | "context"
    confidence: float


class MemoryExtractionLLM(BaseModel):
    facts: List[MemoryFactLLM]


_MEMORY_EXTRACTION_SYSTEM_PROMPT = (
    "Tu extrais des faits durables et réutilisables à partir d'un échange "
    "entre un utilisateur et son assistant exécutif personnel. Ne retiens "
    "que ce qui reste vrai au-delà de cette conversation : préférences "
    "explicites (catégorie 'preference'), faits stables sur l'utilisateur, "
    "son organisation ou ses projets (catégorie 'fact'), ou contexte de "
    "travail durable (catégorie 'context'). Ignore la petite conversation, "
    "les questions ponctuelles et tout ce qui n'a de sens que dans "
    "l'instant présent. Si rien de durable ne ressort, retourne une liste "
    "vide. Chaque fait doit être une phrase courte, autonome, sans "
    "pronom ambigu. 'confidence' est ta certitude entre 0 et 1 que ce fait "
    "est correct et stable."
)


def _memory_extraction_user_prompt(turns: List[Dict[str, str]]) -> Optional[str]:
    lines = [f"{t.get('role', '?')}: {t.get('content', '')}" for t in turns if t.get("content")]
    if not lines:
        return None
    return "Échange à analyser :\n\n" + "\n".join(lines)


def extract_memory_candidates_openai(
    turns: List[Dict[str, str]], model: str, api_key: Optional[str]
) -> Dict[str, Any]:
    """Extraction structurée de faits mémorisables via OpenAI Structured
    Outputs (schéma `MemoryExtractionLLM`)."""
    if not api_key:
        return {"data": None, "erreur": "OPENAI_API_KEY non définie dans l'environnement"}
    user_prompt = _memory_extraction_user_prompt(turns)
    if not user_prompt:
        return {"data": None, "erreur": "Aucun échange à analyser"}

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, timeout=llm_timeout_seconds())
        completion = client.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": _MEMORY_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            response_format=MemoryExtractionLLM,
        )
        message = completion.choices[0].message
        if message.refusal:
            return {"data": None, "erreur": f"Réponse refusée par le modèle : {message.refusal}"}
        if message.parsed is None:
            return {"data": None, "erreur": "Extraction structurée vide"}
        return {"data": message.parsed, "erreur": None}
    except Exception as ex:
        logging.warning("Échec extraction mémoire OpenAI: %s", ex)
        return {"data": None, "erreur": str(ex)}


def extract_memory_candidates_gemini(
    turns: List[Dict[str, str]], model: str, api_key: Optional[str]
) -> Dict[str, Any]:
    """Équivalent Gemini de ``extract_memory_candidates_openai``."""
    if not api_key:
        return {
            "data": None,
            "erreur": "GEMINI_API_KEY ou GOOGLE_API_KEY non définie dans l'environnement",
        }
    user_prompt = _memory_extraction_user_prompt(turns)
    if not user_prompt:
        return {"data": None, "erreur": "Aucun échange à analyser"}

    try:
        import google.generativeai as genai
        from pydantic import ValidationError

        genai.configure(api_key=api_key)
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=MemoryExtractionLLM,
            temperature=0.0,
        )
        try:
            gen_model = genai.GenerativeModel(
                model_name=model,
                system_instruction=_MEMORY_EXTRACTION_SYSTEM_PROMPT,
            )
            prompt = user_prompt
        except TypeError:
            gen_model = genai.GenerativeModel(model_name=model)
            prompt = f"{_MEMORY_EXTRACTION_SYSTEM_PROMPT}\n\n{user_prompt}"

        response = gen_model.generate_content(
            prompt,
            generation_config=generation_config,
            request_options={"timeout": llm_timeout_seconds()},
        )
        if not response.candidates:
            block_reason = getattr(response.prompt_feedback, "block_reason", None)
            erreur = "Réponse Gemini vide ou bloquée" + (
                f" (raison: {block_reason})" if block_reason else ""
            )
            return {"data": None, "erreur": erreur}
        try:
            data = MemoryExtractionLLM.model_validate_json(response.text)
            return {"data": data, "erreur": None}
        except (ValidationError, ValueError) as ve:
            return {"data": None, "erreur": f"Sortie JSON non conforme au schéma : {ve}"}
    except Exception as ex:
        logging.warning("Échec extraction mémoire Gemini: %s", ex)
        return {"data": None, "erreur": str(ex)}
