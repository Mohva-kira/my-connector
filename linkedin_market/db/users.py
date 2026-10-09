"""Utilisateur unique de l'instance (une instance = un utilisateur ; I-01 : tout porte ``user_id``)."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from .. import config
from .models import User, UserSettings

DEFAULT_EMAIL = "local@localhost"


class UserResolutionError(RuntimeError):
    """La base ne contient pas exactement un utilisateur : l'instance ne sait pas lequel utiliser."""


def _default_email() -> str:
    email, _ = config.linkedin_credentials()
    return (email or "").strip().lower() or DEFAULT_EMAIL


def ensure_user(session: Session, email: str | None = None, display_name: str = "") -> UUID:
    """Crée (si besoin) l'utilisateur et sa ligne ``user_settings`` ; renvoie son ``id``. Idempotent.

    ``email`` : par défaut ``LINKEDIN_EMAIL`` en minuscules, sinon ``local@localhost``.
    """
    address = (email or "").strip().lower() or _default_email()
    session.execute(
        insert(User).values(email=address, display_name=display_name).on_conflict_do_nothing(index_elements=["email"])
    )
    user_id = session.execute(select(User.id).where(User.email == address)).scalar_one()
    session.execute(insert(UserSettings).values(user_id=user_id).on_conflict_do_nothing(index_elements=["user_id"]))
    return user_id


def ensure_initial_user(session: Session) -> UUID:
    """Initialisation : crée l'utilisateur seulement si la table ``users`` est vide (A5), puis renvoie son ``id``."""
    if session.execute(select(User.id).limit(1)).first() is None:
        ensure_user(session)
    return current_user_id(session)


def current_user_id(session: Session) -> UUID:
    """``id`` de l'unique utilisateur de l'instance ; lève ``UserResolutionError`` s'il y en a 0 ou plusieurs."""
    ids = session.execute(select(User.id).order_by(User.created_at).limit(2)).scalars().all()
    if not ids:
        raise UserResolutionError(
            "Aucun utilisateur en base : la base n'a pas été initialisée (relance l'application ou le CLI)."
        )
    if len(ids) > 1:
        raise UserResolutionError(
            "La base contient plusieurs utilisateurs alors qu'une instance n'en sert qu'un : "
            "utilise une base dédiée par personne (LM_DATABASE_URL)."
        )
    return ids[0]
