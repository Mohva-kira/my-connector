"""Termes fréquents, compétences demandées et termes en hausse sur 30 jours."""

from __future__ import annotations

import math
from collections import Counter
from datetime import date, timedelta

from . import config, nlp
from .models import Item

MIN_DOC_FREQ = 2


def _doc_grams(item: Item) -> set[str]:
    # Titre et texte séparément : pas de n-gramme à cheval sur les deux.
    return set(nlp.ngrams(nlp.tokens(item.title), 3)) | set(nlp.ngrams(nlp.tokens(item.text), 3))


def top_terms(items: list[Item], limit: int = 30) -> list[dict]:
    """Termes (1-3 mots) classés par nombre de documents qui les contiennent.

    Un n-gramme n'est gardé que s'il apparaît plus souvent que chacun de ses
    mots isolés le laisse attendre : « power bi » remonte, « de power » non.
    """
    if not items:
        return []
    df = Counter(g for item in items for g in _doc_grams(item))
    n = len(items)
    rows = []
    for gram, count in df.items():
        if count < MIN_DOC_FREQ:
            continue
        words = gram.split()
        if len(words) > 1 and count < 0.6 * min(df[w] for w in words):
            continue
        rows.append({"term": gram, "docs": count, "share": round(count / n, 3), "words": len(words)})
    # Les expressions de 2-3 mots portent plus d'information à fréquence égale.
    rows.sort(key=lambda r: r["docs"] * (1 + 0.35 * (r["words"] - 1)), reverse=True)
    return _dedupe_overlaps(rows)[:limit]


def _dedupe_overlaps(rows: list[dict]) -> list[dict]:
    kept: list[dict] = []
    for row in rows:
        if any(row["term"] in k["term"] and row["docs"] <= k["docs"] for k in kept):
            continue
        kept.append(row)
    return kept


def skill_demand(items: list[Item]) -> list[dict]:
    """Part des éléments qui citent chaque compétence, par source."""
    by_source: dict[str, list[Item]] = {}
    for item in items:
        by_source.setdefault(item.kind, []).append(item)
    counts: dict[str, Counter] = {s: Counter() for s in by_source}
    for source, group in by_source.items():
        for item in group:
            counts[source].update(nlp.extract_skills(f"{item.title} {item.text}"))
    skills = Counter()
    for c in counts.values():
        skills.update(c)
    rows = []
    for skill, total in skills.most_common():
        row = {"skill": skill, "total": total}
        for source, group in by_source.items():
            row[source] = round(counts[source][skill] / len(group), 3)
        rows.append(row)
    return rows


def rising_terms(items: list[Item], today: date | None = None, limit: int = 20) -> dict:
    """Compare la moitié récente de la fenêtre (0-15 j) à l'ancienne (15-30 j).

    Croissance = log((part récente + lissage) / (part ancienne + lissage)),
    le lissage de Laplace évite qu'un terme vu une seule fois explose.
    """
    today = today or date.today()
    half = config.TREND_WINDOW_DAYS // 2
    split = (today - timedelta(days=half)).isoformat()
    recent = [i for i in items if i.posted_at >= split]
    older = [i for i in items if i.posted_at < split]
    result = {"recent_docs": len(recent), "older_docs": len(older), "terms": [], "skills": []}
    if len(recent) < 3 or len(older) < 3:
        result["note"] = (
            "Pas assez de données datées sur les deux périodes : relance la collecte "
            "dans quelques jours pour mesurer l'évolution."
        )
        return result

    def growth(counter_recent: Counter, counter_old: Counter, keys) -> list[dict]:
        rows = []
        for key in keys:
            r, o = counter_recent[key], counter_old[key]
            if r + o < MIN_DOC_FREQ:
                continue
            share_r = (r + 0.5) / (len(recent) + 1)
            share_o = (o + 0.5) / (len(older) + 1)
            rows.append({"term": key, "recent": r, "older": o, "growth": round(math.log(share_r / share_o), 3)})
        rows.sort(key=lambda x: x["growth"], reverse=True)
        return rows

    terms_r = Counter(g for i in recent for g in _doc_grams(i))
    terms_o = Counter(g for i in older for g in _doc_grams(i))
    candidates = [t["term"] for t in top_terms(items, limit=200)]
    result["terms"] = [r for r in growth(terms_r, terms_o, candidates) if r["growth"] > 0][:limit]

    skills_r = Counter(s for i in recent for s in nlp.extract_skills(f"{i.title} {i.text}"))
    skills_o = Counter(s for i in older for s in nlp.extract_skills(f"{i.title} {i.text}"))
    result["skills"] = growth(skills_r, skills_o, set(skills_r) | set(skills_o))
    return result


def weekly_series(items: list[Item], skills: list[str], today: date | None = None) -> dict:
    """Part hebdomadaire des éléments citant chaque compétence (graphique d'évolution)."""
    today = today or date.today()
    weeks = math.ceil(config.TREND_WINDOW_DAYS / 7)
    starts = [today - timedelta(days=7 * (weeks - k)) for k in range(weeks)]
    labels = [s.isoformat() for s in starts]
    buckets: list[list[Item]] = [[] for _ in starts]
    for item in items:
        if not item.posted_at:
            continue
        d = date.fromisoformat(item.posted_at)
        for k, start in enumerate(starts):
            if start < d <= start + timedelta(days=7):
                buckets[k].append(item)
                break
    series = {}
    for skill in skills:
        series[skill] = [
            round(sum(skill in nlp.extract_skills(f"{i.title} {i.text}") for i in b) / len(b), 3) if b else None
            for b in buckets
        ]
    return {"weeks": labels, "counts": [len(b) for b in buckets], "series": series}
