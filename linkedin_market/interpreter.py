"""Interprétation des résultats : Claude, ou un interprète local remplaçable.

Les deux implémentations partagent la même interface ``interpret(summary)``
et reçoivent le même résumé chiffré (aucune donnée brute scrapée n'est
envoyée à Claude, seulement les agrégats). Pour remplacer Claude, il suffit
de passer ``LINKEDIN_INTERPRETER=local`` : l'interprète local s'appuie sur
les poids appris par la descente de gradient. Ce n'est pas un modèle de
langage : il produit un texte à base de règles, dont la pertinence
progresse avec les annotations.
"""

from __future__ import annotations

import json
import logging
from typing import Protocol

from . import config

log = logging.getLogger(__name__)

_FEATURE_LABELS = {
    "couverture_competences": "la couverture des compétences demandées",
    "similarite_texte": "la proximité entre ton profil et le texte de l'offre",
    "similarite_titre": "la proximité du titre de poste",
    "seniorite": "l'adéquation du niveau de séniorité",
    "experience": "les années d'expérience exigées",
    "teletravail": "le télétravail explicite",
    "localisation_compatible": "la compatibilité géographique",
    "freelance": "le format freelance / mission",
}

# Délai maximal d'un appel Claude (secondes) ; le streaming est conservé et ``max_retries`` reste celui du SDK.
CLAUDE_TIMEOUT_SECONDS = 120.0


def clean_setting(value: str | None) -> str:
    """Réglage de lieu ou de fuseau nettoyé ; vide si absent (jamais de valeur de repli, E-18)."""
    return " ".join((value or "").split())


def feature_labels(home_location: str = "") -> dict[str, str]:
    """Libellés des critères du matching ; le lieu de l'utilisateur n'apparaît que s'il est renseigné (E-18)."""
    labels = dict(_FEATURE_LABELS)
    if place := clean_setting(home_location):
        labels["localisation_compatible"] += f" (depuis {place})"
    return labels


def system_prompt(home_location: str = "", timezone: str = "") -> str:
    """Consigne système de l'interprétation ; « basé à X » et « fuseau Y » seulement si renseignés (E-18)."""
    place, zone = clean_setting(home_location), clean_setting(timezone)
    profile = "un freelance" + (f" basé à {place}" if place else "")
    wish = "des missions en full remote" + (f" (fuseau {zone})" if zone else "")
    return f"""Tu es un conseiller en positionnement freelance tech. L'utilisateur est \
{profile} qui cherche {wish}. On te donne un \
résumé chiffré, calculé localement, du marché LinkedIn pour un mot-clé de poste : \
termes fréquents, termes et compétences en hausse sur 30 jours, écart de compétences \
entre son profil et les offres, meilleures offres avec leur score, et poids appris par \
son modèle de matching.

Rédige en français une analyse actionnable, en Markdown, avec ces sections :
1. Lecture du marché (3-5 constats appuyés sur les chiffres)
2. Tendances à exploiter
3. Écart de profil : compétences à acquérir ou à mettre en avant, par priorité
4. Offres à cibler en premier et pourquoi
5. Ajustements concrets du profil LinkedIn (titre, mots-clés, posts à publier)
6. Limites de l'analyse (taille d'échantillon, biais de la collecte)

N'invente aucun chiffre absent du résumé. Si l'échantillon est trop petit pour conclure, dis-le."""


class Interpreter(Protocol):
    name: str

    def interpret(self, summary: dict, home_location: str = "", timezone: str = "") -> str: ...


class ClaudeInterpreter:
    name = "claude"

    def __init__(self) -> None:
        import anthropic  # import local : dépendance optionnelle en mode « local »

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(timeout=CLAUDE_TIMEOUT_SECONDS)
        self.model = config.anthropic_model()

    def interpret(self, summary: dict, home_location: str = "", timezone: str = "") -> str:
        payload = json.dumps(summary, ensure_ascii=False, indent=1)
        return self.complete(system_prompt(home_location, timezone), f"<resume_marche>\n{payload}\n</resume_marche>")

    def complete(self, system: str, user: str) -> str:
        """Un appel Claude (streaming, réflexion adaptative, repli serveur si refus)."""
        with self.client.beta.messages.stream(
            model=self.model,
            max_tokens=16000,
            system=system,
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": user}],
        ) as stream:
            message = stream.get_final_message()
        if message.stop_reason == "refusal":
            raise RuntimeError("Claude a refusé la requête.")
        text = "".join(block.text for block in message.content if block.type == "text").strip()
        if message.stop_reason == "max_tokens":
            text += "\n\n_(Réponse tronquée : limite de longueur atteinte.)_"
        return text


class LocalInterpreter:
    """Interprète à base de règles alimenté par le modèle de matching entraîné."""

    name = "local"

    def interpret(self, summary: dict, home_location: str = "", timezone: str = "") -> str:
        labels = feature_labels(home_location)
        lines = ["## Lecture du marché"]
        counts = summary.get("counts", {})
        lines.append(
            f"- Échantillon : {counts.get('job', 0)} offres, {counts.get('post', 0)} posts, "
            f"{counts.get('person', 0)} profils sur {config.TREND_WINDOW_DAYS} jours."
        )
        terms = [t["term"] for t in summary.get("top_terms", [])[:8]]
        if terms:
            lines.append(f"- Termes dominants : {', '.join(terms)}.")
        skills = summary.get("skill_demand", [])[:6]
        if skills:
            lines.append("- Compétences les plus citées : " + ", ".join(s["skill"] for s in skills) + ".")

        lines.append("\n## Tendances")
        rising = summary.get("rising", {})
        if rising.get("note"):
            lines.append(f"- {rising['note']}")
        for row in rising.get("skills", [])[:5]:
            if row["growth"] > 0:
                lines.append(f"- {row['term']} en hausse ({row['older']} -> {row['recent']} mentions).")
        for row in rising.get("terms", [])[:5]:
            lines.append(f"- Terme montant : « {row['term']} ».")

        gap = summary.get("gap", {})
        lines.append("\n## Écart de profil")
        lines.append(f"- Couverture pondérée des compétences demandées : {gap.get('coverage', 0)} %.")
        missing = [m["skill"] for m in gap.get("missing", [])[:6]]
        if missing:
            lines.append(f"- À acquérir ou à rendre visibles en priorité : {', '.join(missing)}.")

        lines.append("\n## Ce qui pèse dans ton matching")
        model = summary.get("model", {})
        source = "appris sur tes annotations" if model.get("trained") else "a priori (annote des offres pour l'affiner)"
        lines.append(f"- Poids {source} :")
        for row in model.get("importance", [])[:4]:
            direction = "augmente" if row["weight"] > 0 else "diminue"
            feature = labels.get(row["feature"], row["feature"])
            lines.append(f"  - {feature} {direction} le score (poids {row['weight']}).")

        top = summary.get("top_jobs", [])[:5]
        if top:
            lines.append("\n## Offres à cibler")
            for job in top:
                miss = f" — manque : {', '.join(job['missing_skills'][:3])}" if job["missing_skills"] else ""
                lines.append(f"- {job['score']} % · {job['title']} ({job['company'] or 'entreprise inconnue'}){miss}")
        return "\n".join(lines)


def get_interpreter(name: str | None = None) -> Interpreter:
    """Interprète demandé ; toute valeur autre que ``claude`` (dont l'ancienne ``none``, E-7) donne le local."""
    name = (name or config.interpreter_name()).lower()
    if name == "claude":
        try:
            return ClaudeInterpreter()
        except Exception as exc:  # SDK absent ou identifiants manquants
            log.warning("Claude indisponible (%s) : interprète local utilisé.", exc)
    return LocalInterpreter()


def interpret(summary: dict, name: str | None = None, home_location: str = "", timezone: str = "") -> tuple[str, str]:
    """Retourne (texte, nom de l'interprète réellement utilisé : ``claude`` ou ``local``, jamais vide, I-11).

    ``home_location`` et ``timezone`` viennent des réglages (le domaine ne lit pas la base, E-18).
    """
    interpreter = get_interpreter(name)
    try:
        return interpreter.interpret(summary, home_location, timezone), interpreter.name
    except Exception as exc:
        if interpreter.name == "local":
            raise
        log.warning("Échec de l'interprétation par Claude (%s) : interprète local utilisé.", exc)
        return LocalInterpreter().interpret(summary, home_location, timezone), "local"
