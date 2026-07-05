"""Templates de brouillon de réponse (Routine, Alerte, Kickoff)."""

import re
from typing import Any, Dict, List

KICKOFF_KEYWORDS = (
    "kickoff",
    "kick-off",
    "lancement",
    "démarrage",
    "demarrage",
    "onboarding",
    "première réunion",
    "premiere reunion",
    "atelier de cadrage",
    "cadrage initial",
)

RDV_KEYWORDS = (
    "rdv",
    "réunion",
    "reunion",
    "meet",
    "visio",
    "teams",
    "zoom",
    "demain",
    "lundi",
    "mardi",
    "mercredi",
    "jeudi",
    "vendredi",
    "semaine prochaine",
    "à ",
    "a ",
)


def _norm(s: str) -> str:
    return (s or "").lower()


def text_blob_for_kickoff(analysis: Dict) -> str:
    """Concatène les textes utiles pour détecter un kickoff."""
    parts: List[str] = []
    ra = analysis.get("résumé_assistant") or {}
    if isinstance(ra, dict) and ra.get("texte"):
        parts.append(str(ra["texte"]))
    auto = analysis.get("résumé_automatique") or {}
    if isinstance(auto, dict) and auto.get("résumé_automatique"):
        parts.append(str(auto["résumé_automatique"]))
    for em in analysis.get("emails_critiques") or []:
        parts.append(str(em.get("subject", "")))
    return "\n".join(parts)


def detect_kickoff_context(analysis: Dict) -> bool:
    blob = _norm(text_blob_for_kickoff(analysis))
    return any(kw in blob for kw in KICKOFF_KEYWORDS)


def infer_template_id(analysis: Dict) -> str:
    """
    Choisit routine | alerte | kickoff selon risque, tendance et contexte.
    """
    risk = analysis.get("évaluation_risque") or {}
    niveau = risk.get("niveau_risque") or "INDETERMINÉ"
    sentiment = analysis.get("analyse_sentiment") or {}
    tendance = sentiment.get("tendance") or "Neutre"

    if niveau == "CRITIQUE":
        return "alerte"
    if detect_kickoff_context(analysis) and niveau != "CRITIQUE":
        return "kickoff"
    if niveau == "MODÉRÉ" and tendance == "Négative":
        return "alerte"
    if niveau == "MODÉRÉ":
        return "routine"
    if tendance == "Négative" and niveau != "FAIBLE":
        return "alerte"
    return "routine"


TEMPLATES: Dict[str, str] = {
    "routine": """Bonjour,

Concernant le projet « {project_name} », le diagnostic indique une situation {situation_label} (risque {risk_level}, tendance {tendance}).

{reco_line}

Je reste à disposition pour tout point de clarification.

Cordialement,
""",
    "alerte": """Bonjour,

Un point d'attention est nécessaire sur le projet « {project_name} » (niveau de risque : {risk_level}, tendance : {tendance}).

{facteurs_line}

{reco_line}

Proposons un point court (15–30 min) pour débloquer ou recadrer les prochaines étapes.

Cordialement,
""",
    "kickoff": """Bonjour,

Pour le lancement / cadrage du projet « {project_name} », voici les éléments à valider ensemble :

{checklist_bullets}

Une fois ces points alignés, nous pourrons figer le planning et les responsabilités.

Cordialement,
""",
}


def build_checklist_from_analysis(analysis: Dict, limit: int = 8) -> List[str]:
    """Construit une liste de tâches à partir des facteurs de risque et emails critiques."""
    items: List[str] = []
    risk = analysis.get("évaluation_risque") or {}
    for f in risk.get("facteurs_risque") or []:
        if f and f not in items:
            items.append(f)
    for em in (analysis.get("emails_critiques") or [])[:3]:
        subj = (em.get("subject") or "").strip()
        if subj:
            items.append(f"Traiter : {subj[:120]}")
    ra = analysis.get("résumé_assistant") or {}
    if isinstance(ra, dict) and ra.get("texte"):
        for line in str(ra["texte"]).splitlines():
            line = line.strip()
            if re.match(r"^[-*•]\s+", line) or re.match(r"^\d+[\.)]\s+", line):
                items.append(re.sub(r"^[-*•\d\.\)\s]+", "", line).strip())
    out: List[str] = []
    for x in items:
        if x and x not in out:
            out.append(x)
        if len(out) >= limit:
            break
    if not out:
        out = ["Relire le fil récent et confirmer les prochaines livraisons."]
    return out[:limit]


def guess_next_meeting_hint(analysis: Dict) -> str:
    """Heuristique simple pour une section « prochain RDV » à partir des textes."""
    blob = text_blob_for_kickoff(analysis)
    rauto = analysis.get("résumé_automatique")
    if isinstance(rauto, dict):
        blob += " " + _norm(str(rauto.get("résumé_automatique", "")))
    else:
        blob += " " + _norm(str(rauto))
    blob = _norm(blob)
    for kw in RDV_KEYWORDS:
        idx = blob.find(kw)
        if idx != -1:
            snippet = blob[max(0, idx - 40) : idx + 80]
            return snippet.strip()[:200]
    return "Aucune date explicite détectée dans les extraits analysés — à confirmer par email."


def format_template(
    template_key: str,
    project_name: str,
    analysis: Dict,
) -> str:
    risk = analysis.get("évaluation_risque") or {}
    sentiment = analysis.get("analyse_sentiment") or {}
    niveau = risk.get("niveau_risque") or "N/A"
    tendance = sentiment.get("tendance") or "Neutre"
    reco = risk.get("recommandation") or ""
    facteurs = risk.get("facteurs_risque") or []
    facteurs_line = (
        "Points notés : " + "; ".join(facteurs[:4]) + "."
        if facteurs
        else "Points notés : à préciser selon votre dernière lecture du fil."
    )
    reco_line = reco if reco else "Poursuivre le suivi selon le calendrier convenu."

    if niveau == "FAIBLE" and tendance in ("Positive", "Neutre"):
        situation_label = "stable"
    elif niveau == "MODÉRÉ":
        situation_label = "à surveiller"
    else:
        situation_label = "sous tension"

    checklist = build_checklist_from_analysis(analysis)
    checklist_bullets = "\n".join(f"• {c}" for c in checklist[:6])

    tpl = TEMPLATES.get(template_key, TEMPLATES["routine"])
    return tpl.format(
        project_name=project_name,
        risk_level=niveau,
        tendance=tendance,
        situation_label=situation_label,
        reco_line=reco_line,
        facteurs_line=facteurs_line,
        checklist_bullets=checklist_bullets,
    )


def generate_response_draft(project_name: str, analysis: Dict) -> Dict[str, Any]:
    """
    Choisit un template et remplit les placeholders.
    Retour utilisable par une API ou le frontend.
    """
    template_id = infer_template_id(analysis)
    body = format_template(template_id, project_name, analysis)
    return {
        "template_id": template_id,
        "project_name": project_name,
        "body": body,
        "checklist_tasks": build_checklist_from_analysis(analysis),
        "next_meeting_hint": guess_next_meeting_hint(analysis),
    }
