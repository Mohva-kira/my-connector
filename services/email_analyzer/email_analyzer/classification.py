"""Classification multi-critères des emails par projet + taxonomie de tags.

Module pur (aucun accès DB/IMAP/LLM) — remplace la simple recherche du nom du
projet en sous-chaîne (``EmailProjectAnalyzer.check_project_relevance``,
project_mail.py) par un score de confiance combinant plusieurs signaux déjà
disponibles sans appel réseau supplémentaire, et dérive une liste de tags
métier + priorité par règles, sans nouvel appel LLM par email (même
discipline coût/latence que ``ai_intelligent.score_email_importance``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

# Poids de chaque signal de classification (points ajoutés au score si le
# signal est présent). Le nom du projet seul vaut exactement MATCH_THRESHOLD :
# sans rules_matrix ni participants connus, le comportement reproduit
# strictement l'ancienne recherche en sous-chaîne (aucune régression pour un
# projet sans règles configurées).
MATCH_THRESHOLD = 45

_SIGNAL_WEIGHTS: Dict[str, int] = {
    "project_name": 45,
    "keyword": 20,
    "sender_email": 25,
    "sender_domain": 20,
    "known_participant": 20,
    "client_or_company_name": 15,
    "reference_number": 25,
}

_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_REFERENCE_PATTERN = re.compile(r"\b[A-Z]{2,}-\d{2,}\b")


@dataclass(frozen=True)
class ProjectRules:
    """Contenu structuré de ``Project.rules_matrix`` (JSONB non typé en base).

    Toutes les clés sont optionnelles. ``from_dict`` est le seul point
    d'analyse de ce JSON non validé : il ne lève jamais, les valeurs
    manquantes ou mal typées dégradent silencieusement vers des listes vides.
    """

    keywords: List[str] = field(default_factory=list)
    sender_domains: List[str] = field(default_factory=list)
    sender_emails: List[str] = field(default_factory=list)
    client_names: List[str] = field(default_factory=list)
    company_names: List[str] = field(default_factory=list)
    reference_numbers: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: Optional[dict]) -> "ProjectRules":
        if not isinstance(raw, dict):
            return cls()

        def _str_list(key: str) -> List[str]:
            value = raw.get(key)
            if not isinstance(value, list):
                return []
            return [str(item).strip().lower() for item in value if str(item).strip()]

        return cls(
            keywords=_str_list("keywords"),
            sender_domains=_str_list("sender_domains"),
            sender_emails=_str_list("sender_emails"),
            client_names=_str_list("client_names"),
            company_names=_str_list("company_names"),
            reference_numbers=_str_list("reference_numbers"),
        )


@dataclass(frozen=True)
class ClassificationResult:
    score: int
    matched: bool
    signals: Dict[str, int]


def _search_text(email_content: Dict) -> str:
    text = email_content.get("normalized_text")
    if text:
        return text
    return f"{email_content.get('subject', '')} {email_content.get('body', '')}".lower()


def _sender_addresses(email_content: Dict) -> Set[str]:
    addresses: Set[str] = set()
    for field_name in ("from", "to", "cc"):
        value = email_content.get(field_name)
        if value:
            addresses.update(m.lower() for m in _EMAIL_PATTERN.findall(value))
    return addresses


def score_project_relevance(
    email_content: Dict,
    project_name: str,
    rules: Optional[ProjectRules] = None,
    known_participants: Optional[Set[str]] = None,
) -> ClassificationResult:
    """Score de confiance 0-100 qu'un email concerne ``project_name``.

    Reproduit exactement l'ancien comportement (substring du nom de projet)
    quand ``rules`` et ``known_participants`` sont absents : les signaux
    additionnels ne font qu'ajouter des points, jamais en retirer.
    """
    text = _search_text(email_content)
    signals: Dict[str, int] = {}

    name = project_name.strip().lower()
    if name and name in text:
        signals["project_name"] = _SIGNAL_WEIGHTS["project_name"]

    rules = rules or ProjectRules()

    if any(kw in text for kw in rules.keywords):
        signals["keyword"] = _SIGNAL_WEIGHTS["keyword"]

    addresses = _sender_addresses(email_content)

    if rules.sender_emails and addresses & set(rules.sender_emails):
        signals["sender_email"] = _SIGNAL_WEIGHTS["sender_email"]
    elif rules.sender_domains and any(
        addr.split("@", 1)[-1] in rules.sender_domains for addr in addresses
    ):
        # Le domaine ne compte que si l'adresse exacte n'a pas déjà matché,
        # pour éviter de compter deux fois le même signal d'adressage.
        signals["sender_domain"] = _SIGNAL_WEIGHTS["sender_domain"]

    if known_participants and addresses & known_participants:
        signals["known_participant"] = _SIGNAL_WEIGHTS["known_participant"]

    if any(n in text for n in (*rules.client_names, *rules.company_names)):
        signals["client_or_company_name"] = _SIGNAL_WEIGHTS["client_or_company_name"]

    if rules.reference_numbers:
        found_refs = {m.lower() for m in _REFERENCE_PATTERN.findall(text.upper())}
        if found_refs & set(rules.reference_numbers):
            signals["reference_number"] = _SIGNAL_WEIGHTS["reference_number"]

    score = min(100, sum(signals.values()))
    return ClassificationResult(score=score, matched=score >= MATCH_THRESHOLD, signals=signals)


# Taxonomie de tags métier (vision Coach IA) — étend le dictionnaire plat
# `EmailIntelligentAnalyzer.risk_keywords` (ai_intelligent.py) en catégories
# nommées ; ajoute des mots-clés propres pour les catégories sans équivalent
# existant (Client, Sécurité, Juridique, RH, Commercial, Validation,
# Facturation, Production, Support).
TAG_KEYWORDS: Dict[str, List[str]] = {
    "Urgent": ["urgent", "critique", "danger", "alerte", "attention", "asap", "immédiat"],
    "Bloquant": ["bloqué", "annulé", "blocage"],
    "Bug": ["bug", "erreur", "crash"],
    "Finance": ["budget", "dépassement"],
    "Facturation": ["facture", "facturation", "paiement", "règlement"],
    "Livraison": ["retard", "échéance", "report", "livré", "livraison"],
    "Technique": ["serveur", "api", "base de données", "déploiement", "architecture"],
    "Client": ["client", "réclamation", "insatisfait"],
    "Sécurité": ["sécurité", "vulnérabilité", "incident de sécurité", "faille"],
    "Juridique": ["contrat", "juridique", "clause", "litige"],
    "RH": ["recrutement", "congé", "entretien", "embauche"],
    "Commercial": ["devis", "proposition commerciale", "négociation", "offre"],
    "Validation": ["validé", "validation", "approuvé", "à valider"],
    "Production": ["mise en production", "production", "déploiement"],
    "Support": ["support", "assistance", "ticket"],
}

# Bornes de bascule du score d'importance (ai_intelligent.score_email_importance,
# 0-100) vers l'étiquette de priorité métier — même échelle que Unit 19.
PRIORITY_THRESHOLDS = [(80, "Critique"), (60, "Haute"), (35, "Moyenne")]
_DEFAULT_PRIORITY_TAG = "Faible"


def _priority_tag(importance_score: Optional[int]) -> str:
    if importance_score is None:
        return _DEFAULT_PRIORITY_TAG
    for threshold, tag in PRIORITY_THRESHOLDS:
        if importance_score >= threshold:
            return tag
    return _DEFAULT_PRIORITY_TAG


def derive_tags(email_content: Dict, importance_score: Optional[int] = None) -> List[str]:
    """Tags métier + priorité dérivés par règles, sans appel LLM."""
    text = _search_text(email_content)
    tags = [name for name, keywords in TAG_KEYWORDS.items() if any(kw in text for kw in keywords)]
    tags.append(_priority_tag(importance_score))
    return tags
