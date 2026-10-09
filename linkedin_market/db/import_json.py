"""Import idempotent des fichiers historiques de ``linkedin_market_data/`` (``backend/03`` § 3.4, ERD § 4).

Ce n'est pas une révision Alembic (D-2) : ``prepare_database()`` l'appelle après ``upgrade head``. Chaque fichier
présent et non suffixé ``.migrated`` est lu ; tout est inséré dans **une seule transaction** (``ON CONFLICT DO
NOTHING`` sur les contraintes d'unicité). L'empreinte SHA-256 de chaque fichier importé est inscrite dans
``legacy_imports`` **dans la même transaction**. Après le ``COMMIT`` seulement, chaque fichier est renommé en
``<nom>.migrated`` (jamais supprimé) ; si ce renommage est interrompu, la relance reconnaît l'empreinte et se
contente de renommer le fichier, sans réimport (les journaux sans contrainte d'unicité ne sont pas doublés).
Les dossiers ``reports/`` (HTML) et ``applications/`` restent en place. Aucun contenu importé n'est journalisé
(données de tiers, cookies).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from .. import config, nlp
from . import crypto
from .models import (
    ACTIVITY_KINDS,
    APPLICATION_EVENT_KINDS,
    APPLICATION_STATUSES,
    APPLY_STATES,
    CAUTION_MAX_ITEMS,
    PROFILE_SOURCES,
    SENIORITIES,
    TASK_KINDS,
    TASK_STATUSES,
    ActivityEvent,
    Application,
    ApplicationEvent,
    CandidateProfile,
    CollectedItem,
    CompanyContact,
    Label,
    LegacyImport,
    MatchingModelRecord,
    MemorizedAnswer,
    PlatformSession,
    ProfileSkillOverride,
    Report,
    Task,
    UserSettings,
)
from .repositories import items as items_repo
from .session import transaction
from .users import ensure_initial_user

log = logging.getLogger(__name__)

MIGRATED_SUFFIX = ".migrated"
TASKS_KEPT = 100
_TASK_NAMESPACE = uuid.UUID("5f1d7c4e-8a51-4b0e-9a7a-2f6c8f0d1e33")
_LEGACY_HISTORY_KINDS = {"statut": "status", "candidature": "apply"}  # C-6


# ------------------------------------------------------------------ outils
def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            rows.append(data)
    return rows


def _moment(value: Any) -> datetime | None:
    """Horodatage ISO ; une date naïve est l'heure locale de la machine (``astimezone()``)."""
    if not value:
        return None
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.astimezone()


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _relative(path: Path | str, data_dir: Path) -> str:
    return Path(os.path.relpath(Path(path).resolve(), data_dir.resolve())).as_posix()


def _int_between(value: Any, low: int, high: int) -> int | None:
    try:
        number = int(value)
    except TypeError, ValueError:
        return None
    return number if low <= number <= high else None


# ---------------------------------------------------------------- importeurs
def _settings(s: Session, uid: UUID, data: dict, counts: Counter) -> None:
    values: dict[str, Any] = {}
    for key in ("keyword", "location", "contact_email", "contact_phone", "signature"):
        if isinstance(data.get(key), str):
            values[key] = data[key].strip()
    for key in ("remote_only", "linkedin_skill_sync"):
        if key in data:
            values[key] = bool(data[key])
    caution = data.get("caution") if data.get("caution") in CAUTION_MAX_ITEMS else None
    if caution:
        values["caution"] = caution
    cap = CAUTION_MAX_ITEMS[caution or "prudent"]
    max_items = _int_between(data.get("max_items"), 1, 200)
    values["max_items"] = max_items if max_items is not None and max_items <= cap else None  # E-3
    if data.get("profile_source") in PROFILE_SOURCES:
        values["profile_source"] = data["profile_source"]
    if data.get("interpreter") in ("claude", "local", "none"):
        values["interpreter"] = "local" if data["interpreter"] == "none" else data["interpreter"]  # E-7
    try:
        score = float(data.get("apply_min_score"))
        if 0 <= score <= 100:
            values["apply_min_score"] = score
    except TypeError, ValueError:
        pass
    for key in ("apply_daily_max", "message_daily_max"):
        number = _int_between(data.get(key), 0, 200)
        if number is not None:
            values[key] = number
    s.execute(update(UserSettings).where(UserSettings.user_id == uid).values(**values))
    counts["user_settings"] += 1
    answers = data.get("answers") if isinstance(data.get("answers"), dict) else {}
    for question, answer in answers.items():
        key = nlp.normalize(str(question))[:200]
        if key.strip():
            inserted = _insert(
                s,
                insert(MemorizedAnswer)
                .values(user_id=uid, question_key=key, answer=str(answer)[:500])
                .on_conflict_do_nothing(),
            )
            counts["memorized_answers"] += inserted


def _ensure_profile(s: Session, uid: UUID) -> None:
    s.execute(insert(CandidateProfile).values(user_id=uid).on_conflict_do_nothing(index_elements=["user_id"]))


def _profile_settings(s: Session, uid: UUID, data: dict, counts: Counter) -> None:
    _ensure_profile(s, uid)
    seniority = str(data.get("seniority") or "").strip().lower()
    s.execute(
        update(CandidateProfile)
        .where(CandidateProfile.user_id == uid)
        .values(
            title=str(data.get("title") or "").strip()[:120],
            years_experience=_int_between(data.get("years_experience"), 0, 60),
            seniority=seniority if seniority in SENIORITIES else "",
        )
    )
    counts["candidate_profiles"] += 1
    added = [nlp.normalize(str(x)) for x in data.get("added_skills") or []]
    removed = [nlp.normalize(str(x)) for x in data.get("removed_skills") or [] if nlp.normalize(str(x)) not in added]
    for skills, action in ((added, "ajoutée"), (removed, "retirée")):
        for skill in dict.fromkeys(x for x in skills if x and len(x) <= 60):
            inserted = _insert(
                s, insert(ProfileSkillOverride).values(user_id=uid, skill=skill, action=action).on_conflict_do_nothing()
            )
            counts["profile_skill_overrides"] += inserted


def _linkedin_profile(s: Session, uid: UUID, text: str, counts: Counter) -> None:
    _ensure_profile(s, uid)
    s.execute(
        update(CandidateProfile)
        .where(CandidateProfile.user_id == uid)
        .values(text=text, skills=nlp.extract_skills(text), origin="linkedin")
    )
    counts["candidate_profiles"] += 1


def _history(s: Session, uid: UUID, rows: list[dict], kept_refs: set[str], now: datetime, counts: Counter) -> None:
    """Dernière version de chaque ``id`` ; lignes hors rétention ignorées sauf offre annotée ou candidatée.

    Les mots-clés de toutes les lignes d'un même ``id`` sont fusionnés : une offre collectée sous deux mots-clés
    reste dans les tendances des deux.
    """
    latest: dict[str, dict] = {}
    keywords: dict[str, list[str]] = {}
    for data in rows:
        if isinstance(data.get("id"), str):
            latest[data["id"]] = data
            keywords.setdefault(data["id"], []).append(str(data.get("keyword") or ""))
    for ref, data in latest.items():
        kind = data.get("kind") or data.get("source")
        parts = items_repo.split_ref(ref)
        if parts is None or kind != parts[0]:
            continue
        collected_at = _moment(data.get("scraped_at")) or now
        kept = ref in kept_refs
        purge_after = None if kept else items_repo.purge_after(kind, collected_at)
        if purge_after is not None and purge_after <= now:
            counts["ignored_expired_items"] += 1
            continue
        title = str(data.get("title") or "")
        text = nlp.strip_linkedin_chrome(str(data.get("text") or ""))
        inserted = _insert(
            s,
            insert(CollectedItem)
            .values(
                user_id=uid,
                source_code="linkedin",
                kind=kind,
                external_id=parts[1],
                url=str(data.get("url") or ""),
                title=title,
                text=None if kind == "person" else text,  # I-09
                skills=nlp.extract_skills(f"{title} {text}"),
                keywords=items_repo.normalize_keywords(keywords[ref]),
                company=str(data.get("company") or ""),
                location=str(data.get("location") or ""),
                posted_at=items_repo.to_date(data.get("posted_at")),
                collected_at=collected_at,
                purge_after=purge_after,
            )
            .on_conflict_do_nothing(),
        )
        counts["collected_items"] += inserted


def _item_id(s: Session, uid: UUID, ref: str) -> UUID | None:
    found = items_repo.find_item(s, uid, ref)
    return found.id if found else None


def _labels(s: Session, uid: UUID, labels: dict, counts: Counter) -> None:
    for ref, value in labels.items():
        item_id = _item_id(s, uid, ref)
        if item_id is None or value not in (0, 1):
            counts["ignored_labels"] += 1  # libellé orphelin
            continue
        inserted = _insert(
            s, insert(Label).values(item_id=item_id, user_id=uid, value=int(value)).on_conflict_do_nothing()
        )
        counts["labels"] += inserted
        s.execute(update(CollectedItem).where(CollectedItem.id == item_id).values(purge_after=None))


def _matching_model(s: Session, uid: UUID, data: dict, labels: dict, counts: Counter) -> None:
    """Modèle appris importé ; ``n_positive`` / ``n_negative`` = nombre de 1 et de 0 de ``labels.json``."""
    exists = s.execute(select(MatchingModelRecord.id).where(MatchingModelRecord.user_id == uid).limit(1)).first()
    if exists or not data.get("weights"):
        return
    values = list(labels.values())
    s.add(
        MatchingModelRecord(
            user_id=uid,
            version=1,
            is_active=True,
            learned=True,
            feature_names=list(data.get("features") or []),
            weights={k: data.get(k) for k in ("weights", "bias", "mean", "std", "train_accuracy", "loss_history")},
            n_positive=sum(1 for v in values if v == 1),
            n_negative=sum(1 for v in values if v == 0),
            trained_at=_moment(data.get("trained_at")) or datetime.now(UTC),
        )
    )
    s.flush()
    counts["matching_models"] += 1


def _report(s: Session, uid: UUID, path: Path, data: dict, data_dir: Path, counts: Counter) -> None:
    used = data.get("interpreter")
    inserted = _insert(
        s,
        insert(Report)
        .values(
            user_id=uid,
            slug=path.stem,
            keyword=str((data.get("request") or {}).get("keyword") or ""),
            summary=_json_safe(data),
            interpretation=str(data.get("interpretation") or ""),
            interpreter_used=used if used in ("claude", "local") else "local",
            html_path=_relative(path.with_suffix(".html"), data_dir),
            created_at=_moment(data.get("generated_at")) or datetime.fromtimestamp(path.stat().st_mtime, UTC),
        )
        .on_conflict_do_nothing(constraint="uq_reports_user_id_slug"),
    )
    counts["reports"] += inserted


def _contact(contact: dict) -> dict:
    source = contact.get("source") or "manuel"
    source = "linkedin_equipe_recrutement" if source == "linkedin" else source  # E-15
    return {
        "full_name": str(contact.get("full_name") or contact.get("name") or "").strip(),
        "profile_url": str(contact.get("profile_url") or "").strip(),
        "email": str(contact.get("email") or "").strip() or None,
        "headline": str(contact.get("headline") or "").strip(),
        "source": source if source in ("linkedin_equipe_recrutement", "manuel") else "manuel",
    }


def _applications(s: Session, uid: UUID, data: dict, data_dir: Path, now: datetime, counts: Counter) -> None:
    for ref, entry in data.items():
        parts = items_repo.split_ref(ref)
        if parts is None or parts[0] != "job" or not isinstance(entry, dict) or not entry.get("folder"):
            counts["ignored_applications"] += 1
            continue
        item_id = _item_id(s, uid, ref)
        if item_id is None:  # offre absente de l'historique : élément minimal
            item_id = s.execute(
                insert(CollectedItem)
                .values(
                    user_id=uid,
                    source_code="linkedin",
                    kind="job",
                    external_id=parts[1],
                    title=str(entry.get("title") or ""),
                    company=str(entry.get("company") or ""),
                    url=str(entry.get("url") or ""),
                    text="",
                    collected_at=_moment(entry.get("prepared_at")) or now,
                )
                .returning(CollectedItem.id)
            ).scalar_one()
            counts["collected_items"] += 1
        s.execute(update(CollectedItem).where(CollectedItem.id == item_id).values(purge_after=None))
        status = entry.get("status") if entry.get("status") in APPLICATION_STATUSES else "préparée"
        apply_state = entry.get("apply_state") if entry.get("apply_state") in APPLY_STATES else ""
        prepared_at = _moment(entry.get("prepared_at")) or now
        application_id = s.execute(
            insert(Application)
            .values(
                user_id=uid,
                item_id=item_id,
                status=status,
                apply_state=apply_state,
                score=float(entry.get("score") or 0),
                generator="claude" if entry.get("generator") == "claude" else "local",
                folder_path=_relative(entry["folder"], data_dir),
                prepared_at=prepared_at,
                updated_at=_moment(entry.get("updated_at")) or prepared_at,
            )
            .on_conflict_do_nothing(constraint="uq_applications_user_id_item_id")
            .returning(Application.id)
        ).scalar_one_or_none()
        if application_id is None:
            continue
        counts["applications"] += 1
        for event in entry.get("history") or []:
            kind = _LEGACY_HISTORY_KINDS.get(event.get("kind"), event.get("kind"))
            if kind not in APPLICATION_EVENT_KINDS:
                continue
            details = _json_safe({k: v for k, v in event.items() if k not in ("at", "kind")})
            s.execute(
                insert(ApplicationEvent).values(
                    application_id=application_id,
                    kind=kind,
                    details=details,
                    at=_moment(event.get("at")) or prepared_at,
                )
            )
            counts["application_events"] += 1
        for contact in entry.get("contacts") or []:
            if isinstance(contact, dict):
                s.execute(
                    insert(CompanyContact).values(
                        user_id=uid, application_id=application_id, legal_basis="interet_legitime", **_contact(contact)
                    )
                )
                counts["company_contacts"] += 1


def _session(s: Session, uid: UUID, data: dict, now: datetime, counts: Counter) -> None:
    logged_in_at = _moment(data.get("logged_in_at"))
    if logged_in_at is None or now - logged_in_at > timedelta(hours=config.SESSION_TTL_HOURS):
        counts["ignored_sessions"] += 1  # plus de 72 h : une nouvelle connexion sera demandée
        return
    token = crypto.encrypt(json.dumps(data.get("cookies") or []).encode("utf-8"))
    inserted = _insert(
        s,
        insert(PlatformSession)
        .values(
            user_id=uid,
            source_code="linkedin",
            cookies_encrypted=token,
            last_login_at=logged_in_at,
            expires_at=logged_in_at + timedelta(hours=config.SESSION_TTL_HOURS),
        )
        .on_conflict_do_nothing(),
    )
    counts["platform_sessions"] += inserted


def _activity(s: Session, uid: UUID, rows: list[dict], counts: Counter) -> None:
    for event in rows:
        at, kind = _moment(event.get("at")), event.get("kind")
        if at is None or kind not in ACTIVITY_KINDS:
            counts["ignored_activity_events"] += 1
            continue
        data = _json_safe({k: v for k, v in event.items() if k not in ("at", "kind")})
        s.execute(insert(ActivityEvent).values(user_id=uid, kind=kind, data=data, at=at))
        counts["activity_events"] += 1


def _tasks(s: Session, uid: UUID, rows: list[dict], counts: Counter) -> None:
    """Les 100 dernières ; une tâche au ``kind`` ou au statut inconnu est ignorée sans faire échouer l'import."""
    for task in rows[-TASKS_KEPT:]:
        created_at = _moment(task.get("created_at"))
        if task.get("kind") not in TASK_KINDS or task.get("status") not in TASK_STATUSES or created_at is None:
            counts["ignored_tasks"] += 1
            continue
        logs = [str(line) for line in task.get("logs") or []][-400:]
        inserted = _insert(
            s,
            insert(Task)
            .values(
                id=uuid.uuid5(_TASK_NAMESPACE, str(task.get("id"))),
                user_id=uid,
                kind=task["kind"],
                title=str(task.get("title") or ""),
                status=task["status"],
                browser=bool(task.get("browser")),
                result=_json_safe(task.get("result")),
                error=str(task.get("error") or "") or None,
                needs_input=None,
                meta=_json_safe(task.get("meta") or {}),
                logs=logs,
                created_at=created_at,
                started_at=_moment(task.get("started_at")),
                finished_at=_moment(task.get("finished_at")),
            )
            .on_conflict_do_nothing(),
        )
        counts["tasks"] += inserted


# --------------------------------------------------------------------- entrée
def _insert(s: Session, statement) -> int:
    """Exécute un ``INSERT … ON CONFLICT DO NOTHING`` ; 1 si la ligne a été insérée, 0 sinon."""
    table = statement.table
    first_pk = next(iter(table.primary_key.columns))
    return int(s.execute(statement.returning(first_pk)).first() is not None)


def _pending(path: Path) -> bool:
    return path.is_file() and not path.name.endswith(MIGRATED_SUFFIX)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_legacy_files(data_dir: Path) -> dict[str, int]:
    """Importe les fichiers historiques non encore migrés ; renvoie le nombre de lignes par table.

    Une erreur annule tout (aucune ligne insérée, aucun fichier renommé). Relancer est sans effet : les fichiers
    importés sont renommés ``.migrated`` après le ``COMMIT`` ; un fichier resté sous son nom (renommage
    interrompu) dont le couple (nom relatif, empreinte) figure dans ``legacy_imports`` est seulement renommé
    (``already_imported_files``).
    """
    data_dir = Path(data_dir)
    files = {
        name: data_dir / name
        for name in (
            "settings.json",
            "profile_settings.json",
            "linkedin_profile.txt",
            "history.jsonl",
            "labels.json",
            "matching_model.json",
            "applications.json",
            "session.json",
            "activity.jsonl",
            "tasks.jsonl",
        )
    }
    found = {name: path for name, path in files.items() if _pending(path)}
    reports_dir = data_dir / "reports"
    found_reports = sorted(p for p in reports_dir.glob("*.json") if _pending(p)) if reports_dir.is_dir() else []
    if not found and not found_reports:
        return {}
    digests = {path: _sha256(path) for path in [*found.values(), *found_reports]}
    # Un fichier est reconnu par son nom relatif ET son empreinte : un rapport identique sous un autre nom est un
    # rapport distinct.
    keys = {path: (_relative(path, data_dir), digest) for path, digest in digests.items()}

    counts: Counter = Counter()
    now = datetime.now(UTC)
    with transaction() as s:
        uid = ensure_initial_user(s)
        known = {
            (file_name, sha256)
            for file_name, sha256 in s.execute(
                select(LegacyImport.file_name, LegacyImport.sha256).where(
                    LegacyImport.user_id == uid, LegacyImport.sha256.in_(set(digests.values()))
                )
            )
        }
        pending = {name: path for name, path in found.items() if keys[path] not in known}
        report_files = [path for path in found_reports if keys[path] not in known]
        counts["already_imported_files"] = len(digests) - len(pending) - len(report_files)
        if "settings.json" in pending:
            _settings(s, uid, _read_json(pending["settings.json"]), counts)
        if "profile_settings.json" in pending:
            _profile_settings(s, uid, _read_json(pending["profile_settings.json"]), counts)
        if "linkedin_profile.txt" in pending:
            _linkedin_profile(s, uid, pending["linkedin_profile.txt"].read_text(encoding="utf-8"), counts)
        # Lus même déjà importés : ils protègent de la purge les offres annotées / candidatées et comptent les
        # annotations du modèle importé.
        labels = _read_json(found["labels.json"]) if "labels.json" in found else {}
        applications = _read_json(found["applications.json"]) if "applications.json" in found else {}
        if "history.jsonl" in pending:
            kept = set(labels) | set(applications)
            _history(s, uid, _read_jsonl(pending["history.jsonl"]), kept, now, counts)
        if "labels.json" in pending:
            _labels(s, uid, labels, counts)
        if "matching_model.json" in pending:
            model_labels = labels or items_repo.get_labels(s, uid)
            _matching_model(s, uid, _read_json(pending["matching_model.json"]), model_labels, counts)
        for path in report_files:
            _report(s, uid, path, _read_json(path), data_dir, counts)
        if "applications.json" in pending:
            _applications(s, uid, applications, data_dir, now, counts)
        if "session.json" in pending:
            _session(s, uid, _read_json(pending["session.json"]), now, counts)
        if "activity.jsonl" in pending:
            _activity(s, uid, _read_jsonl(pending["activity.jsonl"]), counts)
        if "tasks.jsonl" in pending:
            _tasks(s, uid, _read_jsonl(pending["tasks.jsonl"]), counts)
        # Même transaction que l'import : un COMMIT réussi = fichiers reconnus à la relance.
        for path in [*pending.values(), *report_files]:
            s.execute(
                insert(LegacyImport)
                .values(user_id=uid, file_name=keys[path][0], sha256=keys[path][1])
                .on_conflict_do_nothing()
            )
    # Après le COMMIT seulement : les fichiers d'origine sont conservés sous un autre nom.
    for path in digests:
        path.rename(path.with_name(path.name + MIGRATED_SUFFIX))
    if not counts["already_imported_files"]:
        del counts["already_imported_files"]
    log.info("Import des fichiers historiques terminé : %s.", dict(sorted(counts.items())))
    return dict(counts)
