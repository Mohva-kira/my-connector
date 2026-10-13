"""Historique des éléments collectés, annotations, rapports et cache du profil LinkedIn (PostgreSQL).

Façade de persistance (D-1) : signatures publiques inchangées, chaque fonction ouvre une transaction courte et
délègue aux dépôts ``db/repositories``. L'historique cumule les exécutions : c'est lui qui permet de mesurer
l'évolution des termes sur 30 jours, même si LinkedIn ne renvoie qu'une fenêtre partielle à chaque passage.
"""

from __future__ import annotations

from pathlib import Path

from . import config, nlp
from .db import utc_iso, utcnow
from .db.models import CollectedItem
from .db.repositories import items as items_repo
from .db.repositories import profile as profile_repo
from .db.repositories import reports as reports_repo
from .db.repositories import user_transaction
from .models import Item


def _row(item: Item, collected_at) -> dict:
    """Ligne ``collected_items`` ; ``person`` : texte supprimé, compétences seules (I-09, E-16)."""
    text = nlp.strip_linkedin_chrome(item.text or "")
    kind, external_id = item.id.partition(":")[0], item.id.partition(":")[2]
    return {
        "source_code": item.source_code or "linkedin",
        "kind": item.kind or kind,
        "external_id": external_id or item.id,
        "url": item.url or "",
        "title": item.title or "",
        "text": None if item.kind == "person" else text,
        "skills": nlp.extract_skills(f"{item.title} {text}"),
        "keywords": items_repo.normalize_keywords([item.keyword]),
        "company": item.company or "",
        "location": item.location or "",
        "posted_at": items_repo.to_date(item.posted_at),
        "collected_at": collected_at,
    }


def _item(row: CollectedItem, keyword: str | None = None) -> Item:
    """``Item`` lu en base. Un profil tiers n'a pas de texte : ses compétences en tiennent lieu (tendances).

    ``keyword`` : mot-clé demandé (fenêtre des tendances) ; à défaut, le premier mot-clé de collecte.
    """
    text = ", ".join(row.skills) if row.kind == "person" else nlp.strip_linkedin_chrome(row.text or "")
    return Item(
        kind=row.kind,
        id=items_repo.make_ref(row.kind, row.external_id),
        url=row.url,
        title=row.title,
        text=text,
        keyword=keyword if keyword is not None else next(iter(row.keywords), ""),
        company=row.company,
        location=row.location,
        posted_at=row.posted_at.isoformat() if row.posted_at else "",
        scraped_at=utc_iso(row.collected_at) or "",
        source_code=row.source_code,
    )


# ----------------------------------------------------------------- éléments
def append_items(items: list[Item]) -> None:
    """Ajoute ou met à jour les éléments (dernière version gagnante), par lots de 100 (une transaction par lot)."""
    now = utcnow()
    rows: dict[tuple[str, str, str], dict] = {}
    for item in items:
        row = _row(item, now)
        key = (row["source_code"], row["kind"], row["external_id"])
        if key in rows:  # même élément vu deux fois : mots-clés fusionnés
            row["keywords"] = items_repo.normalize_keywords([*rows[key]["keywords"], *row["keywords"]])
        rows[key] = row
    batch = list(rows.values())
    for start in range(0, len(batch), items_repo.BATCH_SIZE):
        with user_transaction() as (s, uid):
            items_repo.upsert_items(s, uid, batch[start : start + items_repo.BATCH_SIZE])


def import_item(item: Item) -> bool:
    """Ajoute **un** élément saisi à la main sans rien réécrire (``False`` s'il est déjà dans l'historique)."""
    with user_transaction() as (s, uid):
        return items_repo.insert_new_item(s, uid, _row(item, utcnow()))


def load_items(keyword: str, window_days: int = config.TREND_WINDOW_DAYS) -> list[Item]:
    """Éléments du mot-clé publiés (ou collectés) dans la fenêtre glissante."""
    with user_transaction() as (s, uid):
        return [_item(row, keyword) for row in items_repo.window_items(s, uid, keyword, window_days)]


def find_item(item_id: str) -> Item | None:
    """Élément de l'historique par identifiant ``<kind>:<external_id>``, quel que soit son mot-clé."""
    with user_transaction() as (s, uid):
        row = items_repo.find_item(s, uid, item_id)
        return _item(row) if row else None


# --------------------------------------------------------------- annotations
def load_labels() -> dict[str, int]:
    """id d'offre -> 1 (pertinente) / 0 (pas pertinente)."""
    with user_transaction() as (s, uid):
        return items_repo.get_labels(s, uid)


def save_labels(labels: dict[str, int]) -> None:
    """Remplace les annotations ; une annotation sur un élément absent de l'historique est ignorée."""
    with user_transaction() as (s, uid):
        items_repo.replace_labels(s, uid, labels)


def set_label(item_id: str, value: int | None) -> int | None:
    """Pose (1 / 0) ou retire (``None``) l'annotation d'un élément ; renvoie le nombre d'annotations.

    ``None`` si l'élément n'est plus dans l'historique (purgé ou inconnu) : rien n'est écrit, l'appelant le
    signale à l'utilisateur au lieu de perdre l'annotation en silence.
    """
    with user_transaction() as (s, uid):
        if not items_repo.set_label(s, uid, item_id, value):
            return None
        return len(items_repo.get_labels(s, uid))


def load_labeled_items() -> dict[str, Item]:
    """Toutes les offres annotées, quel que soit le mot-clé ou leur âge."""
    with user_transaction() as (s, uid):
        return {item.id: item for item in map(_item, items_repo.labeled_items(s, uid))}


# ------------------------------------------------------------ profil LinkedIn
def load_linkedin_profile_text() -> str:
    """Dernier profil LinkedIn importé (réutilisé avec ``--no-scrape``), ``""`` s'il n'y en a pas."""
    with user_transaction() as (s, uid):
        return profile_repo.get_linkedin_text(s, uid) or ""


def save_linkedin_profile_text(text: str) -> None:
    with user_transaction() as (s, uid):
        profile_repo.save_linkedin_text(s, uid, text, nlp.extract_skills(text))


# ------------------------------------------------------------------ rapports
def _valid_name(name: str) -> bool:
    return bool(name) and "/" not in name and "\\" not in name and not name.startswith(".")


def _data_path(relative: str) -> Path | None:
    """Chemin absolu sous ``DATA_DIR`` d'un chemin relatif stocké en base ; ``None`` s'il en sortirait."""
    root = config.DATA_DIR.resolve()
    path = (root / relative).resolve()
    return path if path.is_relative_to(root) else None


def list_reports() -> list[dict]:
    """Rapports générés, du plus récent au plus ancien (``generated_at`` = création en base, UTC, C-1)."""
    with user_transaction() as (s, uid):
        rows = reports_repo.list_reports(s, uid)
    result = []
    for row in rows:
        path = _data_path(row["html_path"])
        result.append(
            {
                "name": row["slug"],
                "keyword": row["keyword"],
                "generated_at": utc_iso(row["created_at"]),
                "jobs": row["jobs"],
                "html": bool(path and path.is_file()),
            }
        )
    return result


def load_report(name: str) -> dict | None:
    """Contenu JSON d'un rapport par nom ; refuse tout nom qui contiendrait un chemin."""
    if not _valid_name(name):
        return None
    with user_transaction() as (s, uid):
        row = reports_repo.get_report(s, uid, name)
        return dict(row.summary) if row else None


def report_html_path(name: str) -> Path | None:
    """Fichier HTML d'un rapport (sous ``DATA_DIR``), ``None`` si le rapport ou le fichier est absent."""
    if not _valid_name(name):
        return None
    with user_transaction() as (s, uid):
        row = reports_repo.get_report(s, uid, name)
        relative = row.html_path if row else None
    path = _data_path(relative) if relative else None
    return path if path and path.is_file() else None
