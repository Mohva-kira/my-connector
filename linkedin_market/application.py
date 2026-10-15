"""Candidature aux offres jugées pertinentes : fiche d'offre lisible, CV adapté
à l'offre et aux tendances du marché, message de candidature, suivi.

La candidature elle-même reste manuelle : l'outil prépare le dossier, ouvre
le CV et la page de l'offre, puis note le statut quand tu as postulé. Rien
n'est envoyé à LinkedIn ou à un recruteur automatiquement.

Le CV est rédigé par Claude (le profil complet et l'offre lui sont envoyés)
ou, en mode local, assemblé à partir de ton profil sans reformulation.
"""

from __future__ import annotations

import html
import json
import logging
import os
import re
import textwrap
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import config, cv_export, cv_format, interpreter, nlp
from .cv_parcours import format_parcours
from .db import utc_iso, utcnow
from .db.repositories import applications as applications_repo
from .db.repositories import suppressions, user_transaction
from .db.repositories.items import make_ref
from .models import Item, Profile

log = logging.getLogger(__name__)

WIDTH = 100
STATUSES = ("préparée", "contactée", "postulée", "entretien", "refusée", "abandonnée", "externe", "échec")
# Un statut « avancé » n'est jamais rétrogradé automatiquement (ex. un message n'écrase pas « postulée »).
_PROGRESS = {"préparée": 0, "externe": 0, "échec": 0, "contactée": 1, "postulée": 2, "entretien": 3}

# --------------------------------------------------------------- fiche offre


def format_card(job: dict, index: int | None = None) -> str:
    """Fiche courte d'une offre pour le terminal : l'essentiel en 5-7 lignes."""
    head = f"[{index}] " if index is not None else ""
    where = " · ".join(x for x in (job.get("company"), job.get("location")) if x)
    lines = [f"{head}{job['score']:>5} %  {job['title'] or 'Sans titre'}" + (f" — {where}" if where else "")]

    facts = []
    if job.get("freelance"):
        facts.append("Freelance")
    if job.get("remote"):
        facts.append("Remote")
    if job.get("seniority_required"):
        facts.append(job["seniority_required"].capitalize())
    if job.get("years_required"):
        facts.append(f"{job['years_required']}+ ans")
    if job.get("posted_at"):
        facts.append(f"publiée le {job['posted_at']}")
    pad = " " * (len(head) + 9)
    if facts:
        lines.append(pad + " · ".join(facts))

    matched, missing = job.get("matched_skills", []), job.get("missing_skills", [])
    if matched:
        lines.append(pad + "✓ " + ", ".join(matched[:8]))
    if missing:
        lines.append(pad + "✗ " + ", ".join(missing[:8]))
    for sentence in job.get("summary", [])[:3]:
        lines.extend(textwrap.wrap(sentence, WIDTH, initial_indent=pad + "› ", subsequent_indent=pad + "  "))
    lines.append(pad + job["url"])
    return "\n".join(lines)


def relevant_jobs(scored: list[dict], labels: dict[str, int], min_score: float) -> list[dict]:
    """Offres annotées pertinentes, puis offres non annotées au-dessus du seuil."""
    marked = [j for j in scored if labels.get(j["id"]) == 1]
    likely = [j for j in scored if j["id"] not in labels and j["score"] >= min_score]
    return marked + likely


# ------------------------------------------------------------------- suivi
# Persistance PostgreSQL (vague 1, D-1) : ``applications``, ``application_events``, ``company_contacts``.
# Les fichiers du dossier (CV, message, conseils, offre) restent sur disque sous ``DATA_DIR``.

HISTORY_KINDS = ("status", "apply", "message", "contacts", "note")
_LEGACY_HISTORY_KINDS = {"statut": "status", "candidature": "apply"}  # C-6
CONTACT_SOURCES = ("linkedin_equipe_recrutement", "manuel")


ContactOpposed = suppressions.OpposedContact  # coordonnée d'un contact opposé saisie : ``422`` côté API (I-13)


class ApplicationNotFound(KeyError):
    """Aucune candidature préparée pour cette offre."""


def _parse_at(value) -> datetime | None:
    """Horodatage ISO (naïf = heure locale de la machine, fichiers hérités) -> ``datetime`` avec fuseau."""
    if not value:
        return None
    if isinstance(value, datetime):
        moment = value
    else:
        try:
            moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    return moment if moment.tzinfo else moment.astimezone()


def _relative_folder(folder) -> str:
    """Chemin du dossier relatif à ``DATA_DIR`` (D-5), quel que soit l'endroit où il a été créé."""
    return Path(os.path.relpath(Path(folder).resolve(), config.DATA_DIR.resolve())).as_posix()


def _contact_row(contact: dict) -> dict:
    """Contact (API, recherche LinkedIn ou dossier hérité) -> colonnes de ``company_contacts`` (C-4, C-5)."""
    source = contact.get("source") or "manuel"
    source = "linkedin_equipe_recrutement" if source == "linkedin" else source
    return {
        "full_name": (contact.get("full_name") or contact.get("name") or "").strip(),
        "profile_url": (contact.get("profile_url") or "").strip(),
        "email": (contact.get("email") or "").strip() or None,
        "headline": (contact.get("headline") or "").strip(),
        "source": source if source in CONTACT_SOURCES else "manuel",
    }


def _contact_public(contact) -> dict:
    return {
        "full_name": contact.full_name,
        "profile_url": contact.profile_url,
        "email": contact.email,
        "headline": contact.headline,
        "source": contact.source,
        "opposed": contact.opposed_at is not None,
        "collected_at": utc_iso(contact.collected_at),
    }


def record_opposition(recipient: str, reason: str) -> None:
    """Opposition saisie par l'utilisateur (I-13, ``backend/01`` § 5.7.1 V4-1 et V4-9), en **une** transaction.

    1. apparie en Python, par empreinte, les contacts de l'utilisateur (toutes candidatures) ;
    2. enregistre l'empreinte de ``recipient`` et celles de **toutes** les coordonnées des contacts appariés, avant
       minimisation (idempotent : le motif d'origine est conservé) ;
    3. minimise chaque contact apparié non encore opposé (``opposed_at``, email et URL effacés) ;
    4. ajoute un événement ``contacts`` ``{"opposition": true}`` par candidature touchée, sans donnée personnelle.

    Aucune fonction ne supprime une opposition.
    """
    target = suppressions.fingerprint(recipient)
    if target is None:
        raise ValueError("Adresse email ou URL de profil LinkedIn attendue.")
    with user_transaction() as (s, uid):
        matched = [
            c
            for c in applications_repo.all_contacts(s, uid)
            if target in suppressions.fingerprints(c.email, c.profile_url)
        ]
        hashes = {target}
        for contact in matched:
            hashes |= suppressions.fingerprints(contact.email, contact.profile_url)
        suppressions.add(s, uid, hashes, reason)
        fresh = [c for c in matched if c.opposed_at is None]
        applications_repo.oppose_contacts(s, uid, [c.id for c in fresh])
        for application_id in dict.fromkeys(c.application_id for c in fresh):
            applications_repo.add_event(s, uid, application_id, "contacts", {"opposition": True}, None)


def drop_opposed(found: list[dict]) -> list[dict]:
    """Retire les contacts dont l'email ou l'URL de profil est opposé (recherche de recruteurs, V4-10 d)."""
    with user_transaction() as (s, uid):
        return [
            c
            for c in found
            if not suppressions.any_suppressed(s, uid, suppressions.fingerprints(c.get("email"), c.get("profile_url")))
        ]


def _public(app, item, contacts: list, events: list) -> dict:
    """Entrée de candidature telle qu'exposée par l'API (``folder`` absolu, C-9 ; dates UTC, C-1/C-2)."""
    return {
        "title": item.title,
        "company": item.company,
        "url": item.url,
        "score": app.score,
        "folder": str(config.DATA_DIR / app.folder_path),
        "generator": app.generator,
        "status": app.status,
        "apply_state": app.apply_state,
        "prepared_at": utc_iso(app.prepared_at),
        "updated_at": utc_iso(app.updated_at) if app.updated_at != app.prepared_at else None,
        "contacts": [_contact_public(c) for c in contacts],
        "history": [{"at": utc_iso(e.at), "kind": e.kind, **e.details} for e in events],
    }


def _load(s, uid, ref: str, *, lock: bool = False):
    found = applications_repo.get(s, uid, ref, lock=lock)
    if found is None:
        raise ApplicationNotFound(ref)
    return found


def _entry(s, uid, ref: str) -> dict:
    app, item = _load(s, uid, ref)
    contacts = applications_repo.contacts(s, uid, [app.id])[app.id]
    events = applications_repo.events(s, uid, [app.id])[app.id]
    return _public(app, item, contacts, events)


def load_applications() -> dict[str, dict]:
    """Toutes les candidatures, par identifiant d'offre ``job:<id>``."""
    return {row.pop("id"): row for row in list_applications()}


def save_applications(data: dict[str, dict]) -> None:
    """Enregistre (crée ou met à jour) des candidatures au format historique ``applications.json``.

    Sert aux dossiers hérités et aux tests : l'offre absente de l'historique est créée a minima. Les candidatures
    absentes de ``data`` ne sont pas supprimées.
    """
    with user_transaction() as (s, uid):
        for ref, entry in data.items():
            item_id = applications_repo.ensure_item(
                s, uid, ref, title=entry.get("title", ""), company=entry.get("company", ""), url=entry.get("url", "")
            )
            values = {
                "status": entry.get("status") or "préparée",
                "apply_state": entry.get("apply_state") or "",
                "score": float(entry.get("score") or 0),
                "generator": entry.get("generator") or "local",
                "folder_path": _relative_folder(entry["folder"]),
            }
            prepared_at = _parse_at(entry.get("prepared_at"))
            if prepared_at:
                values["prepared_at"] = prepared_at
            application_id = applications_repo.upsert(s, uid, item_id, values)
            if "contacts" in entry:
                applications_repo.replace_contacts(s, uid, application_id, [_contact_row(c) for c in entry["contacts"]])
            for event in entry.get("history", []):
                details = {k: v for k, v in event.items() if k not in ("at", "kind")}
                kind = _LEGACY_HISTORY_KINDS.get(event.get("kind"), event.get("kind"))
                applications_repo.add_event(s, uid, application_id, kind, details, _parse_at(event.get("at")))


def set_status(job_id: str, status: str) -> None:
    """Statut libre (``PATCH`` ou choix explicite de l'utilisateur au CLI) : seule voie de régression (I-03)."""
    if status not in STATUSES:
        raise ValueError(f"Statut inconnu : {status}")
    with user_transaction() as (s, uid):
        found = applications_repo.get(s, uid, job_id, lock=True)
        if found is not None:
            applications_repo.update_fields(s, uid, found[0].id, {"status": status})


# -------------------------------------------------------------- génération


def cv_system_prompt(home_location: str = "", timezone: str = "") -> str:
    """Consigne système du CV ; « depuis X » et « fuseau Y » seulement si renseignés, jamais de repli (E-18)."""
    place, zone = interpreter.clean_setting(home_location), interpreter.clean_setting(timezone)
    remote = "full remote" + (f" depuis {place}" if place else "")
    availability = "freelance, full remote" + (f", fuseau {zone}" if zone else "")
    fr, en = cv_format.HEADINGS["fr"], cv_format.HEADINGS["en"]
    return f"""Tu adaptes le CV d'un freelance tech ({remote}) \
à une offre précise, pour maximiser ses chances auprès du recruteur et des ATS.

Règles impératives :
- N'invente rien : aucune expérience, compétence, certification, diplôme, chiffre, \
client ou employeur absent du profil source. Tu peux sélectionner, réordonner, \
regrouper et reformuler.
- Reprends les mots-clés exacts de l'offre quand le profil les justifie.
- Mets d'abord en avant les compétences du profil demandées par l'offre, puis \
celles du profil qui sont en hausse sur le marché (fournies dans <marche>).
- Les compétences demandées absentes du profil ne vont PAS dans le CV : traite-les \
dans les conseils.
- Rédige dans la langue de l'offre.
- Dans l'offre, `[personne]` remplace une donnée de tiers retirée : ne la reconstitue pas.
- CV d'une à deux pages au format ATS : une seule colonne, ni tableau, ni image, \
ni émoji, ni icône. Expériences les plus pertinentes d'abord, les autres résumées.

Structure Markdown exacte du CV (titres de section mot pour mot) :
# <titre aligné sur l'intitulé du poste>
(n'écris ni nom ni coordonnées : ils sont ajoutés automatiquement)
## {fr["profile"]}
(accroche de 3 lignes reprenant les mots-clés de l'offre)
## {fr["skills"]}
- <Catégorie> : <compétence>, <compétence>
## {fr["experience"]}
### <Poste> — <Entreprise>
*<Lieu> · MM/AAAA – MM/AAAA* (ou « Présent »)
- <verbe d'action + résultat ; chiffres uniquement s'ils figurent dans le profil>
## {fr["education"]}
## {fr["certifications"]} (seulement si le profil en contient)
## {fr["languages"]}
## {fr["availability"]}
({availability})
Offre en anglais : titres {", ".join(en[k] for k in en if k not in ("projects", "publications", "volunteering"))}.

Format de sortie exact, sans autre texte :
=== CV ===
(le CV en Markdown)
=== MESSAGE ===
(message de candidature de 120 mots maximum, prêt à coller sur LinkedIn ou par email)
=== CONSEILS ===
(3 à 5 puces : écarts avec l'offre et comment les traiter en entretien ou dans le message)"""


@dataclass
class Application:
    cv: str
    message: str
    advice: str
    generator: str  # claude | local


def _market_context(results: dict, profile: Profile) -> dict:
    owned = set(profile.skills)
    rising = [r["term"] for r in results.get("rising", {}).get("skills", []) if r["growth"] > 0]
    demand = [d["skill"] for d in results.get("skill_demand", [])]
    return {
        "competences_en_hausse_que_tu_as": [s for s in rising if s in owned][:8],
        "competences_en_hausse_absentes": [s for s in rising if s not in owned][:8],
        "competences_les_plus_demandees": demand[:15],
        "termes_frequents": [t["term"] for t in results.get("top_terms", [])[:15]],
    }


def _split_sections(text: str) -> tuple[str, str, str]:
    parts = re.split(r"^===\s*(CV|MESSAGE|CONSEILS)\s*===\s*$", text, flags=re.MULTILINE)
    found = {parts[i].upper(): parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
    if not found:
        return text.strip(), "", ""
    return found.get("CV", ""), found.get("MESSAGE", ""), found.get("CONSEILS", "")


def _generate_with_claude(
    profile: Profile,
    item: Item,
    job: dict,
    market: dict,
    known_names: Iterable[str] = (),
    home_location: str = "",
    timezone: str = "",
) -> Application:
    """CV par Claude : profil de l'utilisateur + offre **après** rédaction des tiers (E-6, I-11).

    ``profile.text`` est le texte de l'utilisateur : il n'est jamais rédigé. Seul le texte de l'offre l'est.
    """
    claude = interpreter.ClaudeInterpreter()
    offer = {
        k: job.get(k)
        for k in (
            "title",
            "company",
            "location",
            "matched_skills",
            "missing_skills",
            "years_required",
            "seniority_required",
            "freelance",
            "remote",
        )
    }
    summary = {
        "titre": profile.title,
        "competences": profile.skills,
        "annees_experience": profile.years_experience,
        "seniorite": profile.seniority,
    }
    offer_text = nlp.redact_third_parties(item.text, known_names)
    user = (
        f"<profil_source>\n{profile.text}\n</profil_source>\n\n"
        f"<profil_resume>\n{json.dumps(summary, ensure_ascii=False)}\n</profil_resume>\n\n"
        f"<offre>\n{json.dumps(offer, ensure_ascii=False)}\n\nDescription :\n{offer_text}\n</offre>\n\n"
        f"<marche>\n{json.dumps(market, ensure_ascii=False)}\n</marche>"
    )
    cv, message, advice = _split_sections(claude.complete(cv_system_prompt(home_location, timezone), user))
    return Application(cv, message, advice, "claude")


_DISPLAY = {
    "fastapi": "FastAPI",
    "postgresql": "PostgreSQL",
    "mongodb": "MongoDB",
    "mysql": "MySQL",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "node.js": "Node.js",
    "next.js": "Next.js",
    "graphql": "GraphQL",
    "langchain": "LangChain",
    "mlops": "MLOps",
    "devops": "DevOps",
    "power bi": "Power BI",
    "elasticsearch": "Elasticsearch",
    "go": "Go",
    "rest api": "REST API",
    "ia générative": "IA générative",
    "c#": "C# / .NET",
}


def _pretty(skill: str) -> str:
    """Nom affichable d'une compétence canonique (« aws » -> « AWS »)."""
    if skill in _DISPLAY:
        return _DISPLAY[skill]
    return skill.upper() if len(skill) <= 3 or skill in ("ui/ux", "ci/cd") else skill[0].upper() + skill[1:]


_LOCAL_TEXT = {
    "fr": {
        "years": "{n} ans d'expérience",
        "no_years": "Expérience confirmée",
        "level": ", niveau {level}",
        "pitch": "{years}{level}. Expertise directement mobilisable sur ce poste : {top}.",
        "core": "Clés pour ce poste",
        "more": "Autres compétences",
        "remote": ("full remote", " depuis {place}", " (fuseau {zone})"),
        "available": "Freelance en {remote}, disponible pour des équipes internationales.",
    },
    "en": {
        "years": "{n} years of experience",
        "no_years": "Proven experience",
        "level": ", {level} level",
        "pitch": "{years}{level}. Expertise directly applicable to this role: {top}.",
        "core": "Core for this role",
        "more": "Additional skills",
        "remote": ("fully remote", " from {place}", " (time zone {zone})"),
        "available": "Freelance, {remote}, available for international teams.",
    },
}
_MORE_SKILLS_MAX = 15


def _remote_phrase(home_location: str, timezone: str, *, with_zone: bool, lang: str = "fr") -> str:
    """« full remote depuis X (fuseau Y) » sans repli : les fragments vides sont omis (E-18)."""
    place, zone = interpreter.clean_setting(home_location), interpreter.clean_setting(timezone)
    base, from_place, in_zone = _LOCAL_TEXT[lang]["remote"]
    phrase = base + (from_place.format(place=place) if place else "")
    return phrase + (in_zone.format(zone=zone) if zone and with_zone else "")


def _generate_locally(
    profile: Profile, item: Item, job: dict, market: dict, home_location: str = "", timezone: str = ""
) -> Application:
    """Assemblage sans reformulation au format ATS (``cv_format``), dans la langue de l'offre."""
    lang = cv_format.language_of(item.text)
    text = _LOCAL_TEXT[lang]
    matched = job.get("matched_skills", [])
    # Compétences en hausse que l'utilisateur possède d'abord, puis le reste du profil.
    others = [s for s in market["competences_en_hausse_que_tu_as"] if s not in matched]
    others += [s for s in profile.skills if s not in matched and s not in others]
    years = text["years"].format(n=profile.years_experience) if profile.years_experience else text["no_years"]
    level = text["level"].format(level=profile.seniority) if profile.seniority else ""
    top = ", ".join(map(_pretty, matched[:4] or profile.skills[:4]))
    parcours = format_parcours(profile.text)

    def section(key: str, *body: str) -> list[str]:
        return ["", f"## {cv_format.heading(key, lang)}", "", *body]

    skills = [f"- **{text['core']}** : {', '.join(map(_pretty, matched))}"] if matched else []
    if others:
        skills.append(f"- **{text['more']}** : {', '.join(map(_pretty, others[:_MORE_SKILLS_MAX]))}")
    cv = [
        f"# {profile.title or job['title']}",
        *section("profile", text["pitch"].format(years=years, level=level, top=top)),
        *section("skills", *skills),
        *section("experience", parcours.experience),
    ]
    for key, body in parcours.sections:
        cv += section(key, body)
    remote = _remote_phrase(home_location, timezone, with_zone=True, lang=lang)
    cv += section("availability", text["available"].format(remote=remote))

    fr = _LOCAL_TEXT["fr"]  # le message reste en français
    msg_years = fr["years"].format(n=profile.years_experience) if profile.years_experience else fr["no_years"]
    msg_level = fr["level"].format(level=profile.seniority) if profile.seniority else ""
    company = job.get("company") or "votre équipe"
    remote = _remote_phrase(home_location, timezone, with_zone=False)
    message = (
        f"Bonjour,\n\nVotre offre « {job['title']} » a retenu mon attention. {msg_years}{msg_level}, "
        f"je maîtrise {top}, au cœur du poste. Freelance en {remote}, je peux "
        f"rejoindre {company} rapidement. Je serais ravi d'échanger sur vos besoins.\n\nCordialement,"
    )
    missing = job.get("missing_skills", [])
    advice = "\n".join(
        [
            f"- Compétence demandée absente du profil : **{_pretty(s)}** — prépare un exemple proche "
            "ou un plan de montée en compétence."
            for s in missing[:5]
        ]
        or ["- Aucun écart de compétence détecté : insiste sur des résultats chiffrés."]
    )
    return Application("\n".join(cv), message, advice, "local")


def known_contact_names() -> list[str]:
    """Noms complets des contacts de l'utilisateur (``company_contacts``), à retirer du texte envoyé à Claude (E-6)."""
    with user_transaction() as (s, uid):
        return sorted({c.full_name for c in applications_repo.all_contacts(s, uid) if c.full_name})


def generate(
    profile: Profile,
    item: Item,
    job: dict,
    results: dict,
    generator: str | None = None,
    *,
    known_names: Iterable[str] = (),
    home_location: str = "",
    timezone: str = "",
    identity: cv_format.Identity | None = None,
) -> Application:
    """Dossier de candidature par Claude, ou assemblé localement si Claude est indisponible.

    ``known_names``, ``home_location``, ``timezone`` et ``identity`` sont fournis par l'appelant (le domaine ne lit
    pas la base). L'en-tête de contact est posé ici, après la génération : Claude ne reçoit aucune coordonnée.
    """
    market = _market_context(results, profile)
    app = None
    if (generator or config.interpreter_name()) == "claude":
        try:
            app = _generate_with_claude(profile, item, job, market, known_names, home_location, timezone)
            if not app.cv:
                log.warning("Réponse de Claude vide : CV assemblé localement.")
                app = None
        except Exception as exc:
            log.warning("CV par Claude impossible (%s) : CV assemblé localement.", exc)
    app = app or _generate_locally(profile, item, job, market, home_location, timezone)
    app.cv = cv_format.apply_identity(app.cv, identity or cv_format.Identity())
    issues = cv_format.validate_cv(app.cv)
    if issues:
        app.advice = "\n".join([app.advice, *(f"- Format ATS : {issue}" for issue in issues)]).strip()
    return app


# ---------------------------------------------------------------- fichiers

_CV_HTML = """<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CV — __TITLE__</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.2/marked.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/dompurify/3.1.6/purify.min.js"></script>
<style>
:root { --bg: #f6f6f4; --paper: #ffffff; --text: #0b0b0b; --text-2: #52514e; --accent: #2a78d6; --border: #e3e2de; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
.bar { max-width: 210mm; margin: 16px auto 0; padding: 0 16px; color: var(--text-2); font-size: 13px; }
.bar button { font: inherit; padding: 6px 14px; border-radius: 6px; border: 1px solid var(--border);
  background: var(--paper); cursor: pointer; margin-right: 8px; }
article { box-sizing: border-box; max-width: 210mm; margin: 12px auto 32px; padding: 18mm 16mm;
  background: var(--paper); border: 1px solid var(--border); }
h1 { font-size: 24px; margin: 0 0 6px; }
h2 { font-size: 14px; text-transform: uppercase; letter-spacing: .05em; color: var(--accent);
  border-bottom: 1px solid var(--border); padding-bottom: 3px; margin: 18px 0 8px; }
h3 { font-size: 14px; margin: 12px 0 4px; }
ul { padding-left: 18px; margin: 4px 0; }
p { margin: 4px 0; }
@media (max-width: 600px) { article { padding: 20px 16px; } }
@media print { body { background: none; } .bar { display: none; }
  article { border: none; margin: 0; padding: 0; max-width: none; } @page { size: A4; margin: 14mm; } }
</style>
</head>
<body>
<div class="bar"><button onclick="window.print()">Imprimer / PDF</button>
Offre : <a href="__URL__" target="_blank" rel="noopener">__TITLE__</a></div>
<article id="cv"></article>
<script id="md" type="text/markdown">__MD__</script>
<script>
document.getElementById('cv').innerHTML =
  DOMPurify.sanitize(marked.parse(document.getElementById('md').textContent));
</script>
</body>
</html>
"""


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", nlp.normalize(text)).strip("-")[:40] or "offre"


def write(app: Application, item: Item, job: dict) -> Path:
    """Écrit le dossier de candidature et l'enregistre dans le suivi."""
    config.ensure_data_dirs()
    folder = (
        config.APPLICATIONS_DIR / f"{datetime.now():%Y%m%d}-{_slug(job.get('company') or '')}-{_slug(job['title'])}"
    )
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "cv.md").write_text(app.cv, encoding="utf-8")
    (folder / "message.txt").write_text(app.message, encoding="utf-8")
    (folder / "conseils.md").write_text(app.advice, encoding="utf-8")
    (folder / "offre.txt").write_text(
        f"{job['title']} — {job.get('company', '')}\n{job['url']}\n\n{item.text}", encoding="utf-8"
    )
    page = (
        _CV_HTML.replace("__TITLE__", html.escape(job["title"] or "Offre"))
        .replace("__URL__", html.escape(job["url"]))
        .replace("__MD__", app.cv.replace("</", "<\\/"))
    )
    (folder / "cv.html").write_text(page, encoding="utf-8")

    with user_transaction() as (s, uid):
        item_id = applications_repo.ensure_item(
            s, uid, job["id"], title=job["title"], company=job.get("company", ""), url=job["url"]
        )
        # Re-préparer ne doit effacer ni le statut, ni les contacts, ni l'historique.
        applications_repo.upsert(
            s,
            uid,
            item_id,
            {
                "score": float(job["score"]),
                "generator": app.generator,
                "folder_path": _relative_folder(folder),
                "prepared_at": utcnow(),
            },
        )
    return folder


# ------------------------------------------------- suivi enrichi (interface web)


def get_application(job_id: str) -> dict:
    with user_transaction() as (s, uid):
        return _entry(s, uid, job_id)


def list_applications() -> list[dict]:
    """Candidatures, de la plus récente préparation à la plus ancienne."""
    with user_transaction() as (s, uid):
        rows = applications_repo.list_all(s, uid)
        ids = [app.id for app, _ in rows]
        contacts = applications_repo.contacts(s, uid, ids)
        events = applications_repo.events(s, uid, ids)
        return [
            {"id": make_ref(item.kind, item.external_id), **_public(app, item, contacts[app.id], events[app.id])}
            for app, item in rows
        ]


def update_application(job_id: str, **fields) -> dict:
    """Met à jour ``apply_state``, ``status`` et/ou ``contacts`` (liste remplacée) ; touche ``updated_at``."""
    with user_transaction() as (s, uid):
        app, _ = _load(s, uid, job_id, lock=True)
        values = {k: fields[k] for k in ("apply_state", "status") if k in fields}
        applications_repo.update_fields(s, uid, app.id, values)
        if "contacts" in fields:
            applications_repo.replace_contacts(s, uid, app.id, [_contact_row(c) for c in fields["contacts"]])
        return _entry(s, uid, job_id)


def add_history(job_id: str, kind: str, **details) -> dict:
    """Ajoute un événement à l'historique (``status``, ``apply``, ``message``, ``contacts``, ``note`` ; C-6)."""
    kind = _LEGACY_HISTORY_KINDS.get(kind, kind)
    if kind not in HISTORY_KINDS:
        raise ValueError(f"Type d'historique inconnu : {kind}")
    payload = json.loads(json.dumps(details, ensure_ascii=False, default=str))
    with user_transaction() as (s, uid):
        app, _ = _load(s, uid, job_id)
        applications_repo.add_event(s, uid, app.id, kind, payload, None)
        return _entry(s, uid, job_id)


def advance_status(job_id: str, status: str) -> None:
    """Passe au statut demandé sauf s'il ferait reculer la candidature (I-03).

    Lecture et écriture dans **une** transaction, la candidature verrouillée (``SELECT … FOR UPDATE``).
    """
    if status not in STATUSES:
        raise ValueError(f"Statut inconnu : {status}")
    with user_transaction() as (s, uid):
        app, _ = _load(s, uid, job_id, lock=True)
        current = app.status
        if _PROGRESS.get(status, 0) > _PROGRESS.get(current, 0) or current in ("préparée", "externe", "échec"):
            applications_repo.update_fields(s, uid, app.id, {"status": status})


def folder_of(entry: dict) -> Path:
    """Dossier d'une candidature, refusé s'il sort de ``applications/`` (fichier JSON modifiable à la main)."""
    folder = Path(entry["folder"]).resolve()
    if config.APPLICATIONS_DIR.resolve() not in folder.parents:
        raise ValueError("Dossier de candidature hors du répertoire de données.")
    return folder


def read_files(entry: dict) -> dict[str, str]:
    folder = folder_of(entry)

    def read(name: str) -> str:
        path = folder / name
        return path.read_text(encoding="utf-8") if path.is_file() else ""

    return {
        "cv": read("cv.md"),
        "message": read("message.txt"),
        "advice": read("conseils.md"),
        "offer": read("offre.txt"),
    }


def write_files(job_id: str, *, cv: str | None = None, message: str | None = None, advice: str | None = None) -> None:
    """Enregistre les textes édités ; régénère ``cv.html`` quand le CV change."""
    entry = get_application(job_id)
    folder = folder_of(entry)
    if message is not None:
        (folder / "message.txt").write_text(message, encoding="utf-8")
    if advice is not None:
        (folder / "conseils.md").write_text(advice, encoding="utf-8")
    if cv is not None:
        (folder / "cv.md").write_text(cv, encoding="utf-8")
        (folder / "cv.html").write_text(
            _CV_HTML.replace("__TITLE__", html.escape(entry.get("title") or "Offre"))
            .replace("__URL__", html.escape(entry.get("url", "")))
            .replace("__MD__", cv.replace("</", "<\\/")),
            encoding="utf-8",
        )
        cv_export.remove(folder)
    update_application(job_id)
