"""Classification multi-critères des emails par projet + taxonomie de tags.

Module pur (aucun accès DB/IMAP/LLM) — remplace la simple recherche du nom du
projet en sous-chaîne (``EmailProjectAnalyzer.check_project_relevance``,
project_mail.py) par un score de confiance combinant plusieurs signaux déjà
disponibles sans appel réseau supplémentaire, et dérive une liste de tags
métier + priorité par règles, sans nouvel appel LLM par email (même
discipline coût/latence que ``ai_intelligent.score_email_importance``).
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .subject_tickets import parse_subject_ticket_key

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
    # Une clé de sujet ([Banque_Projet_Ticket]) déjà apprise sur ce projet
    # (voir rules.subject_keys) est un identifiant quasi certain — même palier
    # que project_name.
    "subject_ticket_key": 45,
    # Similarité approximative entre le sujet et le nom/les mots-clés du
    # projet (voir fuzzy_similarity) — signal plus faible qu'un "keyword"
    # exact car approximatif, ne se déclenche que si aucun "keyword" exact
    # n'a déjà matché (voir score_project_relevance).
    "fuzzy_theme": 15,
}

_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_REFERENCE_PATTERN = re.compile(r"\b[A-Z]{2,}-\d{2,}\b")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_FUZZY_THEME_THRESHOLD = 0.5


def _token_set(text: str) -> Set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 2}


def fuzzy_similarity(a: str, b: str) -> float:
    """Similarité approximative (0-1) entre deux libellés, tolérante à
    l'ordre des mots ("Wallet Mobile" vs "Mobile Wallet"), à un libellé court
    noyé dans un sujet long ("SPECIFICATION API - SI MOBILE WALLET" contient
    "wallet mobile"), et aux variantes d'écriture proches — mélange d'un
    score de tokens (le max entre Jaccard et un ratio de recouvrement/
    containment, pour ne pas diluer un petit mot-clé dans un long sujet) et
    de ``difflib.SequenceMatcher``, stdlib pur (pas de nouvelle dépendance)."""
    if not a or not b:
        return 0.0
    tokens_a, tokens_b = _token_set(a), _token_set(b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    jaccard = intersection / len(tokens_a | tokens_b)
    containment = intersection / min(len(tokens_a), len(tokens_b))
    token_score = max(jaccard, containment)
    seq_ratio = difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
    return 0.7 * token_score + 0.3 * seq_ratio


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
    # Clés de sujet ([Banque_Projet_Ticket], voir subject_tickets.py)
    # confirmées pour ce projet — alimentées par le flux de découverte
    # (création/rattachement manuel), pas d'extraction automatique.
    subject_keys: List[str] = field(default_factory=list)

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
            subject_keys=_str_list("subject_keys"),
        )


# Champs de rules_matrix qu'un utilisateur peut plausiblement taper en texte
# libre pour désigner un projet (domaine expéditeur, nom de client...) —
# reference_numbers/subject_keys exclus : ce sont des identifiants extraits
# automatiquement, pas des termes qu'on saisit spontanément dans le chat.
_RULES_MATRIX_SEARCH_FIELDS = (
    ("domaine expéditeur", "sender_domains"),
    ("adresse expéditeur", "sender_emails"),
    ("client", "client_names"),
    ("société", "company_names"),
    ("mot-clé", "keywords"),
)


def rules_matrix_field_matches(rules: ProjectRules, needle: str, exact: bool) -> List[tuple]:
    """Cherche ``needle`` (déjà ``.strip().lower()``) dans les champs de
    ``rules_matrix`` qu'un utilisateur peut nommer directement (voir
    ``_RULES_MATRIX_SEARCH_FIELDS``) — complète la résolution de projet par
    nom (``ToolExecutor._resolve_project_id``, ``SearchProjectsTool``) pour le
    cas où l'utilisateur désigne le projet par son domaine expéditeur ou son
    client plutôt que par ``Project.name`` (ex. "teyliom.com" quand le projet
    s'appelle "Résidence Teyliom Marina"). ``exact=True`` : égalité stricte
    (résolution sans ambiguïté) ; ``exact=False`` : sous-chaîne dans les deux
    sens (recherche large). Retourne toutes les correspondances trouvées,
    sous la forme ``[(libellé_français, valeur), ...]``."""
    if not needle:
        return []
    matches: List[tuple] = []
    for label, field_name in _RULES_MATRIX_SEARCH_FIELDS:
        for value in getattr(rules, field_name):
            if exact:
                matched = value == needle
            else:
                matched = needle in value or value in needle
            if matched:
                matches.append((label, value))
    return matches


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

    subject = email_content.get("subject") or ""

    ticket = parse_subject_ticket_key(subject)
    if ticket and rules.subject_keys and ticket.normalized_label.lower() in rules.subject_keys:
        signals["subject_ticket_key"] = _SIGNAL_WEIGHTS["subject_ticket_key"]

    # Détection approximative des sujets thématiques sans motif exact (ex.
    # "Phase pilote Agency Banking, Wallet Mobile..." pour un projet dont le
    # mot-clé est "wallet mobile") — ne se déclenche que si aucun "keyword"
    # exact n'a déjà matché, pour ne pas compter deux fois le même signal.
    if subject and "keyword" not in signals:
        candidates = [project_name, *rules.keywords]
        if any(
            fuzzy_similarity(subject, candidate) >= _FUZZY_THEME_THRESHOLD
            for candidate in candidates
            if candidate
        ):
            signals["fuzzy_theme"] = _SIGNAL_WEIGHTS["fuzzy_theme"]

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
    # Motif [Banque_Projet_Ticket] en tête de sujet == réclamation client,
    # indépendamment du contenu du corps (donc indépendant du tag "Client"
    # ci-dessus, qui reste basé sur les mots-clés du corps/sujet).
    if parse_subject_ticket_key(email_content.get("subject")) is not None:
        tags.append("Réclamation")
    tags.append(_priority_tag(importance_score))
    return tags
