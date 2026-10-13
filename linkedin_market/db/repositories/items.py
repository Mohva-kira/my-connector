"""Éléments collectés (``collected_items``) et annotations (``labels``).

Un élément est désigné côté application par ``<kind>:<external_id>`` (``job:4012345678``) ; la base stocke
``kind`` + ``external_id`` et ce dépôt recompose l'identifiant (contrat d'API § 1.1).
"""

from collections.abc import Iterable
from datetime import date, datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import Date, Text, case, cast, delete, func, literal_column, select, tuple_, update
from sqlalchemy.dialects.postgresql import ARRAY, array, insert
from sqlalchemy.orm import Session

from ..models import ITEM_KINDS, CollectedItem, Label

BATCH_SIZE = 100
# Rétention (I-10) : posts et profils tiers 30 jours, offres sans candidature ni annotation 90 jours.
RETENTION_DAYS = {"job": 90, "post": 30, "person": 30}


def split_ref(ref: str) -> tuple[str, str] | None:
    """``job:123`` -> ``("job", "123")`` ; ``None`` si la forme est invalide."""
    kind, sep, external_id = (ref or "").partition(":")
    if not sep or kind not in ITEM_KINDS or not external_id:
        return None
    return kind, external_id


def make_ref(kind: str, external_id: str) -> str:
    return f"{kind}:{external_id}"


def normalize_keyword(keyword: str | None) -> str:
    """Mot-clé tel que stocké dans ``collected_items.keywords`` : ``lower(trim())``."""
    return (keyword or "").strip().lower()


def normalize_keywords(keywords: Iterable[str | None]) -> list[str]:
    """Mots-clés normalisés, sans vide ni doublon, dans l'ordre de première apparition."""
    return list(dict.fromkeys(k for k in map(normalize_keyword, keywords) if k))


# Fusion sans doublon à la recollecte : les mots-clés déjà connus d'abord, puis les nouveaux.
_MERGED_KEYWORDS = literal_column(
    "collected_items.keywords || ARRAY(SELECT k FROM unnest(excluded.keywords) WITH ORDINALITY AS t(k, n) "
    "WHERE k <> ALL(collected_items.keywords) ORDER BY n)",
    type_=ARRAY(Text()),
)


def purge_after(kind: str, collected_at: datetime) -> datetime:
    return collected_at + timedelta(days=RETENTION_DAYS[kind])


def upsert_items(session: Session, user_id: UUID, rows: list[dict[str, Any]]) -> int:
    """Insère ou met à jour des éléments (au plus ``BATCH_SIZE`` par appel : une transaction par lot).

    Chaque ligne porte ``source_code``, ``kind``, ``external_id``, ``task_id``, ``url``, ``title``, ``text``,
    ``skills``, ``keywords``, ``company``, ``location``, ``posted_at``, ``collected_at``. ``purge_after`` est
    calculé ici ; il reste ``NULL`` pour une offre déjà annotée ou candidatée. ``person`` : ``text`` est
    toujours ``NULL`` (I-09). Les mots-clés sont normalisés et, à la recollecte, fusionnés sans doublon avec
    ceux déjà connus : l'élément reste visible dans les tendances de chaque mot-clé.
    """
    if not rows:
        return 0
    if len(rows) > BATCH_SIZE:
        raise ValueError(f"Lot trop grand ({len(rows)} > {BATCH_SIZE}).")
    values = []
    for row in rows:
        value = {**row, "user_id": user_id, "purge_after": purge_after(row["kind"], row["collected_at"])}
        value["keywords"] = normalize_keywords(row.get("keywords") or [])
        if row["kind"] == "person":
            value["text"] = None
        values.append(value)
    statement = insert(CollectedItem).values(values)
    excluded = statement.excluded
    statement = statement.on_conflict_do_update(
        constraint="uq_collected_items_user_id_source_code_kind_external_id",
        set_={
            "title": excluded.title,
            "text": excluded.text,
            "skills": excluded.skills,
            "keywords": _MERGED_KEYWORDS,
            "collected_at": excluded.collected_at,
            "purge_after": case(
                (CollectedItem.purge_after.is_(None), None),
                else_=excluded.purge_after,
            ),
        },
        where=CollectedItem.user_id == user_id,
    )
    session.execute(statement)
    return len(values)


def insert_new_item(session: Session, user_id: UUID, row: dict[str, Any]) -> bool:
    """Insère un élément **sans jamais réécrire** un élément existant (import manuel) ; ``False`` si déjà présent.

    Même colonnes que ``upsert_items`` ; ``purge_after`` calculé ici (90 jours pour une offre, I-10).
    """
    value = {**row, "user_id": user_id, "purge_after": purge_after(row["kind"], row["collected_at"])}
    value["keywords"] = normalize_keywords(row.get("keywords") or [])
    inserted = session.execute(
        insert(CollectedItem)
        .values(value)
        .on_conflict_do_nothing(constraint="uq_collected_items_user_id_source_code_kind_external_id")
        .returning(CollectedItem.id)
    ).scalar_one_or_none()
    return inserted is not None


def window_items(session: Session, user_id: UUID, keyword: str, window_days: int) -> list[CollectedItem]:
    """Éléments collectés sous ce mot-clé (insensible à la casse) dans la fenêtre glissante (tendances : 30 jours).

    ``keywords @> ARRAY[lower(trim(:kw))]`` : une offre collectée sous plusieurs mots-clés appartient à chacun.
    """
    day = func.coalesce(CollectedItem.posted_at, cast(CollectedItem.collected_at, Date))
    wanted = array([func.lower(func.trim(normalize_keyword(keyword)))], type_=Text())
    return list(
        session.execute(
            select(CollectedItem)
            .where(
                CollectedItem.user_id == user_id,
                CollectedItem.keywords.contains(wanted),
                day >= func.current_date() - window_days,
            )
            .order_by(CollectedItem.collected_at, CollectedItem.id)
        ).scalars()
    )


def find_item(session: Session, user_id: UUID, ref: str) -> CollectedItem | None:
    parts = split_ref(ref)
    if parts is None:
        return None
    return session.execute(
        select(CollectedItem)
        .where(CollectedItem.user_id == user_id, CollectedItem.kind == parts[0], CollectedItem.external_id == parts[1])
        .order_by(CollectedItem.collected_at.desc())
        .limit(1)
    ).scalar_one_or_none()


def _ids_by_ref(session: Session, user_id: UUID, refs: Iterable[str]) -> dict[str, UUID]:
    pairs = [p for p in (split_ref(r) for r in refs) if p is not None]
    if not pairs:
        return {}
    rows = session.execute(
        select(CollectedItem.kind, CollectedItem.external_id, CollectedItem.id).where(
            CollectedItem.user_id == user_id,
            tuple_(CollectedItem.kind, CollectedItem.external_id).in_(pairs),
        )
    )
    return {make_ref(kind, external_id): item_id for kind, external_id, item_id in rows}


def get_labels(session: Session, user_id: UUID) -> dict[str, int]:
    """``<kind>:<external_id>`` -> 1 (pertinente) / 0 (pas pertinente)."""
    rows = session.execute(
        select(CollectedItem.kind, CollectedItem.external_id, Label.value)
        .join(CollectedItem, CollectedItem.id == Label.item_id)
        .where(Label.user_id == user_id, CollectedItem.user_id == user_id)
        .order_by(Label.labeled_at, CollectedItem.external_id)
    )
    return {make_ref(kind, external_id): value for kind, external_id, value in rows}


def replace_labels(session: Session, user_id: UUID, labels: dict[str, int]) -> int:
    """Remplace les annotations ; une annotation sur un élément absent de l'historique est ignorée.

    Une offre annotée sort de la purge des 90 jours (``purge_after = NULL``). Renvoie le nombre d'annotations.
    """
    ids = _ids_by_ref(session, user_id, labels)
    session.execute(delete(Label).where(Label.user_id == user_id, Label.item_id.not_in(list(ids.values()) or [None])))
    for ref, item_id in ids.items():
        statement = insert(Label).values(item_id=item_id, user_id=user_id, value=int(labels[ref]))
        session.execute(
            statement.on_conflict_do_update(
                index_elements=["item_id"],
                set_={"value": statement.excluded.value, "labeled_at": func.now()},
                where=Label.value != statement.excluded.value,
            )
        )
    if ids:
        session.execute(
            update(CollectedItem)
            .where(CollectedItem.user_id == user_id, CollectedItem.id.in_(list(ids.values())))
            .values(purge_after=None)
        )
    return len(ids)


def set_label(session: Session, user_id: UUID, ref: str, value: int | None) -> bool:
    """Pose ou retire l'annotation d'un élément ; ``False`` (rien écrit) s'il n'est plus dans l'historique.

    Une offre annotée sort de la purge des 90 jours (``purge_after = NULL``).
    """
    item = find_item(session, user_id, ref)
    if item is None:
        return False
    if value is None:
        session.execute(delete(Label).where(Label.user_id == user_id, Label.item_id == item.id))
        return True
    statement = insert(Label).values(item_id=item.id, user_id=user_id, value=int(value))
    session.execute(
        statement.on_conflict_do_update(
            index_elements=["item_id"],
            set_={"value": statement.excluded.value, "labeled_at": func.now()},
            where=Label.value != statement.excluded.value,
        )
    )
    item.purge_after = None
    return True


def labeled_items(session: Session, user_id: UUID) -> list[CollectedItem]:
    """Tous les éléments annotés, quel que soit le mot-clé ou leur âge."""
    return list(
        session.execute(
            select(CollectedItem)
            .join(Label, Label.item_id == CollectedItem.id)
            .where(CollectedItem.user_id == user_id, Label.user_id == user_id)
        ).scalars()
    )


def to_date(value: str | None) -> date | None:
    """Date ISO ``AAAA-MM-JJ`` (ou datetime ISO) -> ``date`` ; ``None`` si vide ou illisible."""
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None
