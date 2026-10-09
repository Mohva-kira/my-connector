"""Moteur et sessions SQLAlchemy (``backend/03`` § 1).

Le moteur est créé à la première utilisation (et non à l'import) : importer ce module ne demande pas
``LM_DATABASE_URL``. Une ``Session`` n'est jamais partagée entre threads : chaque fonction de persistance ouvre
sa propre transaction courte avec ``transaction()``.
"""

import threading
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from .. import config

SessionLocal = sessionmaker(expire_on_commit=False)

_engine: Engine | None = None
_lock = threading.Lock()


def get_engine() -> Engine:
    """Moteur unique du processus (``pool_size=5`` ≥ 4 tâches + requêtes HTTP), lié à ``SessionLocal``."""
    global _engine
    with _lock:
        if _engine is None:
            _engine = create_engine(config.database_url(), pool_pre_ping=True, pool_size=5, max_overflow=2)
            SessionLocal.configure(bind=_engine)
        return _engine


def dispose_engine() -> None:
    """Ferme les connexions du pool ; le prochain ``get_engine()`` relit ``LM_DATABASE_URL``."""
    global _engine
    with _lock:
        if _engine is not None:
            _engine.dispose()
            _engine = None
            SessionLocal.configure(bind=None)


@contextmanager
def transaction() -> Iterator[Session]:
    """Session dans une transaction : ``COMMIT`` en sortie normale, ``ROLLBACK`` sur exception."""
    get_engine()
    with SessionLocal.begin() as session:
        yield session
