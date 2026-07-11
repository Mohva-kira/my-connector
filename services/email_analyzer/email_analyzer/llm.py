"""Appels LLM (OpenAI, Gemini) pour résumés assistant."""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from .config import (
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
    "exécutive du suivi de projets par email, pensée pour être lue en moins d'une "
    "minute. Réponds en français, dans un style dense, factuel, orienté décision, "
    "sans jargon inutile ; n'invente jamais d'information absente des emails. "
    "Structure impérativement ta réponse en 5 sections numérotées : "
    "1) Synthèse en une phrase. 2) Points clés. 3) Décisions à prendre ou risques "
    "identifiés. 4) Échéances repérées dans les emails. 5) Recommandation d'action "
    "priorisée."
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
    "`llm_risk_level` doit valoir 'FAIBLE', 'MODÉRÉ' ou 'CRITIQUE'."
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


class DeadlineItem(BaseModel):
    description: str
    date_iso: Optional[str]
    confiance: Optional[str]


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


def _structured_user_prompt(threads: List[Dict], project_name: str, tags_reference: Optional[List[Dict]]) -> Optional[str]:
    corpus = build_llm_thread_corpus(threads)
    if not corpus.strip():
        return None
    tags_context = format_tags_reference_context(tags_reference)
    return (
        f"Projet : {project_name}\n\n"
        + (f"{tags_context}\n\n" if tags_context else "")
        + "Voici les fils de discussion (regroupés par sujet, extraits) à analyser :\n\n"
        + corpus
    )


def extract_structured_project_summary_openai(
    threads: List[Dict],
    project_name: str,
    model: str,
    api_key: Optional[str],
    tags_reference: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """Extraction structurée via OpenAI Structured Outputs (schéma
    `ProjectSummaryLLM`). Retourne ``{"data": ProjectSummaryLLM, "erreur": None}``
    ou ``{"data": None, "erreur": "..."}`` — jamais d'exception qui remonte au
    worker arq (même discipline que `generate_openai_assistant_summary`)."""
    if not api_key:
        return {"data": None, "erreur": "OPENAI_API_KEY non définie dans l'environnement"}
    if not threads:
        return {"data": None, "erreur": "Aucun email à analyser pour cette période"}
    user_prompt = _structured_user_prompt(threads, project_name, tags_reference)
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
) -> Dict[str, Any]:
    """Équivalent Gemini de ``extract_structured_project_summary_openai`` —
    ``response_schema=ProjectSummaryLLM`` (google-generativeai >= 0.8, déjà la
    version installée), une tentative puis retry unique sur échec de
    validation Pydantic (rfc-email-pipeline-v2.md §11)."""
    if not api_key:
        return {
            "data": None,
            "erreur": "GEMINI_API_KEY ou GOOGLE_API_KEY non définie dans l'environnement",
        }
    if not threads:
        return {"data": None, "erreur": "Aucun email à analyser pour cette période"}
    user_prompt = _structured_user_prompt(threads, project_name, tags_reference)
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
        response = chat.send_message(last_user, generation_config=generation_config)

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
