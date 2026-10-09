"""Profil candidat (``candidate_profiles``) et surcharges de compétences (``profile_skill_overrides``)."""

from typing import Any
from uuid import UUID

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ..models import CandidateProfile, ProfileSkillOverride

ADDED, REMOVED = "ajoutée", "retirée"


def _row(session: Session, user_id: UUID) -> CandidateProfile:
    session.execute(insert(CandidateProfile).values(user_id=user_id).on_conflict_do_nothing(index_elements=["user_id"]))
    return session.execute(select(CandidateProfile).where(CandidateProfile.user_id == user_id)).scalar_one()


def get_overrides(session: Session, user_id: UUID) -> dict[str, Any]:
    """Surcharges : ``title``, ``years_experience``, ``seniority``, ``added_skills``, ``removed_skills``."""
    row = _row(session, user_id)
    rows = session.execute(
        select(ProfileSkillOverride.skill, ProfileSkillOverride.action)
        .where(ProfileSkillOverride.user_id == user_id)
        .order_by(ProfileSkillOverride.created_at, ProfileSkillOverride.skill)
    ).all()
    return {
        "title": row.title,
        "years_experience": row.years_experience,
        "seniority": row.seniority,
        "added_skills": [skill for skill, action in rows if action == ADDED],
        "removed_skills": [skill for skill, action in rows if action == REMOVED],
    }


def save_overrides(session: Session, user_id: UUID, overrides: dict[str, Any]) -> None:
    """Enregistre les surcharges complètes (remplace les listes de compétences)."""
    _row(session, user_id)
    session.execute(
        update(CandidateProfile)
        .where(CandidateProfile.user_id == user_id)
        .values(
            title=overrides["title"],
            years_experience=overrides["years_experience"],
            seniority=overrides["seniority"],
        )
    )
    wanted = {skill: ADDED for skill in overrides["added_skills"]}
    wanted |= {skill: REMOVED for skill in overrides["removed_skills"] if skill not in wanted}
    current = dict(
        session.execute(
            select(ProfileSkillOverride.skill, ProfileSkillOverride.action).where(
                ProfileSkillOverride.user_id == user_id
            )
        ).all()
    )
    stale = [skill for skill, action in current.items() if wanted.get(skill) != action]
    if stale:
        session.execute(
            delete(ProfileSkillOverride).where(
                ProfileSkillOverride.user_id == user_id, ProfileSkillOverride.skill.in_(stale)
            )
        )
    for skill, action in wanted.items():
        if current.get(skill) != action:
            session.execute(insert(ProfileSkillOverride).values(user_id=user_id, skill=skill, action=action))


def get_linkedin_text(session: Session, user_id: UUID) -> str | None:
    """Dernier profil LinkedIn importé (cache), ``None`` si jamais importé."""
    return session.execute(
        select(CandidateProfile.text).where(CandidateProfile.user_id == user_id)
    ).scalar_one_or_none()


def save_linkedin_text(session: Session, user_id: UUID, text: str, skills: list[str]) -> None:
    _row(session, user_id)
    session.execute(
        update(CandidateProfile)
        .where(CandidateProfile.user_id == user_id)
        .values(text=text, skills=skills, origin="linkedin", synced_at=func.now())
    )
