"""Dépôts SQL : une fonction par opération, signature ``(session, user_id, …)``, ``user_id`` filtré partout (I-01).

Les modules de persistance historiques (``settings``, ``storage``, ``report``, ``matching``) restent les façades
publiques : chacune ouvre une transaction courte avec ``user_transaction()`` et délègue ici (D-1).
"""

from collections.abc import Iterator
from contextlib import contextmanager
from uuid import UUID

from sqlalchemy.orm import Session

from ..session import transaction
from ..users import current_user_id


@contextmanager
def user_transaction() -> Iterator[tuple[Session, UUID]]:
    """Transaction courte et ``user_id`` de l'instance."""
    with transaction() as session:
        yield session, current_user_id(session)
