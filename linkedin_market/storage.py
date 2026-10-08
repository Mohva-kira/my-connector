"""Historique local des éléments collectés et des annotations utilisateur.

L'historique cumule les exécutions : c'est lui qui permet de mesurer
l'évolution des termes sur 30 jours, même si LinkedIn ne renvoie qu'une
fenêtre partielle à chaque passage.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

from . import config, nlp
from .models import Item


def append_items(items: list[Item]) -> None:
    config.ensure_data_dirs()
    with config.HISTORY_FILE.open("a", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")


def _item(data: dict) -> Item:
    """Item nettoyé : l'historique peut contenir l'interface LinkedIn capturée avant le filtrage."""
    item = Item.from_dict(data)
    item.text = nlp.strip_linkedin_chrome(item.text)
    return item


def load_items(keyword: str, window_days: int = config.TREND_WINDOW_DAYS) -> list[Item]:
    """Éléments du mot-clé publiés dans la fenêtre ; la dernière version d'un id gagne."""
    if not config.HISTORY_FILE.exists():
        return []
    cutoff = (date.today() - timedelta(days=window_days)).isoformat()
    latest: dict[str, Item] = {}
    with config.HISTORY_FILE.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                item = _item(json.loads(line))
            except json.JSONDecodeError:
                continue
            if item.keyword.lower() == keyword.lower() and (item.posted_at or "9999") >= cutoff:
                latest[item.id] = item
    return list(latest.values())


def load_labels() -> dict[str, int]:
    """id d'offre -> 1 (pertinente) / 0 (pas pertinente)."""
    if not config.LABELS_FILE.exists():
        return {}
    return json.loads(config.LABELS_FILE.read_text(encoding="utf-8"))


def save_labels(labels: dict[str, int]) -> None:
    config.ensure_data_dirs()
    config.LABELS_FILE.write_text(json.dumps(labels, indent=2), encoding="utf-8")


def load_labeled_items() -> dict[str, Item]:
    """Toutes les offres annotées, quel que soit le mot-clé ou leur âge."""
    labels = load_labels()
    found: dict[str, Item] = {}
    if not labels or not config.HISTORY_FILE.exists():
        return found
    with config.HISTORY_FILE.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("id") in labels:
                found[data["id"]] = _item(data)
    return found


def find_item(item_id: str) -> Item | None:
    """Dernière version d'un élément de l'historique, quel que soit son mot-clé."""
    if not config.HISTORY_FILE.exists():
        return None
    found: Item | None = None
    with config.HISTORY_FILE.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("id") == item_id:
                found = _item(data)
    return found


def list_reports() -> list[dict]:
    """Rapports JSON générés, du plus récent au plus ancien."""
    if not config.REPORTS_DIR.exists():
        return []
    rows = []
    for path in sorted(config.REPORTS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows.append({
            "name": path.stem, "keyword": data.get("request", {}).get("keyword", ""),
            "generated_at": data.get("generated_at", ""), "jobs": len(data.get("top_jobs", [])),
            "html": path.with_suffix(".html").exists(),
        })
    return sorted(rows, key=lambda r: r["generated_at"], reverse=True)


def load_report(name: str) -> dict | None:
    """Rapport par nom (sans extension) ; refuse tout nom qui sortirait de ``reports/``."""
    if not name or "/" in name or "\\" in name or name.startswith("."):
        return None
    path = config.REPORTS_DIR / f"{name}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
