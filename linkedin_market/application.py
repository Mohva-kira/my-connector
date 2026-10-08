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
import re
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import config, interpreter, nlp
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


def load_applications() -> dict[str, dict]:
    if not config.APPLICATIONS_FILE.exists():
        return {}
    return json.loads(config.APPLICATIONS_FILE.read_text(encoding="utf-8"))


def save_applications(data: dict[str, dict]) -> None:
    config.ensure_data_dirs()
    config.APPLICATIONS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def set_status(job_id: str, status: str) -> None:
    data = load_applications()
    if job_id in data:
        data[job_id]["status"] = status
        data[job_id]["updated_at"] = datetime.now().isoformat(timespec="seconds")
        save_applications(data)


# -------------------------------------------------------------- génération

CV_SYSTEM_PROMPT = """Tu adaptes le CV d'un freelance tech (full remote depuis Abidjan) \
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
- CV d'une à deux pages : titre aligné sur le poste, accroche de 3 lignes, \
compétences clés, expériences en puces orientées résultats (les plus pertinentes \
d'abord, les autres résumées), formation, langues, disponibilité (freelance, full \
remote, fuseau GMT).

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
    generator: str          # claude | local


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


def _generate_with_claude(profile: Profile, item: Item, job: dict, market: dict) -> Application:
    claude = interpreter.ClaudeInterpreter()
    offer = {k: job.get(k) for k in ("title", "company", "location", "matched_skills", "missing_skills",
                                     "years_required", "seniority_required", "freelance", "remote")}
    user = (
        f"<profil_source>\n{profile.text}\n</profil_source>\n\n"
        f"<profil_resume>\n{json.dumps({'titre': profile.title, 'competences': profile.skills, 'annees_experience': profile.years_experience, 'seniorite': profile.seniority}, ensure_ascii=False)}\n</profil_resume>\n\n"
        f"<offre>\n{json.dumps(offer, ensure_ascii=False)}\n\nDescription :\n{item.text}\n</offre>\n\n"
        f"<marche>\n{json.dumps(market, ensure_ascii=False)}\n</marche>"
    )
    cv, message, advice = _split_sections(claude.complete(CV_SYSTEM_PROMPT, user))
    return Application(cv, message, advice, "claude")


_DISPLAY = {
    "fastapi": "FastAPI", "postgresql": "PostgreSQL", "mongodb": "MongoDB", "mysql": "MySQL",
    "javascript": "JavaScript", "typescript": "TypeScript", "node.js": "Node.js", "next.js": "Next.js",
    "graphql": "GraphQL", "langchain": "LangChain", "mlops": "MLOps", "devops": "DevOps", "power bi": "Power BI",
    "elasticsearch": "Elasticsearch", "go": "Go", "rest api": "REST API", "ia générative": "IA générative", "c#": "C# / .NET",
}


def _pretty(skill: str) -> str:
    """Nom affichable d'une compétence canonique (« aws » -> « AWS »)."""
    if skill in _DISPLAY:
        return _DISPLAY[skill]
    return skill.upper() if len(skill) <= 3 or skill in ("ui/ux", "ci/cd") else skill[0].upper() + skill[1:]


def _generate_locally(profile: Profile, item: Item, job: dict, market: dict) -> Application:
    """Assemblage sans reformulation : atouts ciblés en tête, profil source ensuite."""
    trending = set(market["competences_en_hausse_que_tu_as"])
    matched = job.get("matched_skills", [])
    extra_trending = [s for s in market["competences_en_hausse_que_tu_as"] if s not in matched]
    years = f"{profile.years_experience} ans d'expérience" if profile.years_experience else "Expérience confirmée"
    level = f", niveau {profile.seniority}" if profile.seniority else ""
    top = ", ".join(map(_pretty, matched[:4] or profile.skills[:4]))

    cv = [f"# {profile.title or job['title']}", "",
          f"**Candidature : {job['title']}{' — ' + job['company'] if job.get('company') else ''}**", "",
          "## Accroche", "",
          f"{years}{level}. Expertise directement mobilisable sur ce poste : {top}. "
          "Freelance en full remote depuis Abidjan (GMT), disponible pour des équipes "
          "internationales.", "",
          "## Atouts pour cette offre", ""]
    cv += [f"- **{_pretty(s)}**" + (" — compétence en forte demande actuellement" if s in trending else "") for s in matched]
    if extra_trending:
        cv += ["", "## Autres compétences recherchées sur le marché", "", ", ".join(map(_pretty, extra_trending))]
    # Deux espaces en fin de ligne : saut de ligne Markdown, la mise en page du profil est conservée.
    cv += ["", "## Parcours", "", "  \n".join(profile.text.strip().splitlines())]

    company = job.get("company") or "votre équipe"
    message = (
        f"Bonjour,\n\nVotre offre « {job['title']} » a retenu mon attention. {years}{level}, "
        f"je maîtrise {top}, au cœur du poste. Freelance en full remote depuis Abidjan, je peux "
        f"rejoindre {company} rapidement. Je serais ravi d'échanger sur vos besoins.\n\nCordialement,"
    )
    missing = job.get("missing_skills", [])
    advice = "\n".join(
        [f"- Compétence demandée absente du profil : **{_pretty(s)}** — prépare un exemple proche ou un plan de montée en compétence."
         for s in missing[:5]]
        or ["- Aucun écart de compétence détecté : insiste sur des résultats chiffrés."]
    )
    return Application("\n".join(cv), message, advice, "local")


def generate(profile: Profile, item: Item, job: dict, results: dict, generator: str | None = None) -> Application:
    market = _market_context(results, profile)
    if (generator or config.interpreter_name()) == "claude":
        try:
            app = _generate_with_claude(profile, item, job, market)
            if app.cv:
                return app
            log.warning("Réponse de Claude vide : CV assemblé localement.")
        except Exception as exc:
            log.warning("CV par Claude impossible (%s) : CV assemblé localement.", exc)
    return _generate_locally(profile, item, job, market)


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
    folder = config.APPLICATIONS_DIR / f"{datetime.now():%Y%m%d}-{_slug(job.get('company') or '')}-{_slug(job['title'])}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "cv.md").write_text(app.cv, encoding="utf-8")
    (folder / "message.txt").write_text(app.message, encoding="utf-8")
    (folder / "conseils.md").write_text(app.advice, encoding="utf-8")
    (folder / "offre.txt").write_text(
        f"{job['title']} — {job.get('company', '')}\n{job['url']}\n\n{item.text}", encoding="utf-8")
    page = (_CV_HTML.replace("__TITLE__", html.escape(job["title"] or "Offre"))
            .replace("__URL__", html.escape(job["url"]))
            .replace("__MD__", app.cv.replace("</", "<\\/")))
    (folder / "cv.html").write_text(page, encoding="utf-8")

    data = load_applications()
    previous = data.get(job["id"], {})  # re-préparer ne doit pas effacer statut, contacts ni historique
    data[job["id"]] = {
        **{k: previous[k] for k in ("contacts", "history", "apply_state", "updated_at") if k in previous},
        "title": job["title"], "company": job.get("company", ""), "url": job["url"],
        "score": job["score"], "folder": str(folder), "generator": app.generator,
        "status": previous.get("status", "préparée"),
        "prepared_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_applications(data)
    return folder


# ------------------------------------------------- suivi enrichi (interface web)


class ApplicationNotFound(KeyError):
    """Aucune candidature préparée pour cette offre."""


def _normalize(entry: dict) -> dict:
    """Candidatures écrites par l'ancien CLI : ajoute les champs récents sans rien supprimer."""
    entry.setdefault("contacts", [])
    entry.setdefault("history", [])
    entry.setdefault("apply_state", "")
    return entry


def get_application(job_id: str) -> dict:
    data = load_applications()
    if job_id not in data:
        raise ApplicationNotFound(job_id)
    return _normalize(data[job_id])


def list_applications() -> list[dict]:
    rows = [{"id": job_id, **_normalize(entry)} for job_id, entry in load_applications().items()]
    return sorted(rows, key=lambda r: r.get("prepared_at", ""), reverse=True)


def update_application(job_id: str, **fields) -> dict:
    data = load_applications()
    if job_id not in data:
        raise ApplicationNotFound(job_id)
    data[job_id].update(fields)
    data[job_id]["updated_at"] = datetime.now().isoformat(timespec="seconds")
    save_applications(data)
    return _normalize(data[job_id])


def add_history(job_id: str, kind: str, **details) -> dict:
    entry = get_application(job_id)
    history = entry["history"] + [{"at": datetime.now().isoformat(timespec="seconds"), "kind": kind, **details}]
    return update_application(job_id, history=history)


def advance_status(job_id: str, status: str) -> None:
    """Passe au statut demandé sauf s'il ferait reculer la candidature."""
    if status not in STATUSES:
        raise ValueError(f"Statut inconnu : {status}")
    current = get_application(job_id)["status"]
    if _PROGRESS.get(status, 0) > _PROGRESS.get(current, 0) or current in ("préparée", "externe", "échec"):
        set_status(job_id, status)


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

    return {"cv": read("cv.md"), "message": read("message.txt"), "advice": read("conseils.md"), "offer": read("offre.txt")}


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
            .replace("__MD__", cv.replace("</", "<\\/")), encoding="utf-8")
        (folder / config.CV_PDF_NAME).unlink(missing_ok=True)  # PDF périmé : régénéré à la demande
    update_application(job_id)
