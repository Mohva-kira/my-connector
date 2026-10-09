"""Environnement Alembic de linkedin_market.

Connexion, par ordre de priorité : ``config.attributes["connection"]`` (appel programmatique : démarrage,
tests), option ``sqlalchemy.url`` posée par le code appelant, sinon ``LM_DATABASE_URL`` (``config.database_url()``).
"""

from alembic import context
from linkedin_market import config as lm_config
from linkedin_market.db.models import Base
from sqlalchemy import Connection, create_engine, pool

target_metadata = Base.metadata


def _url() -> str:
    return context.config.get_main_option("sqlalchemy.url") or lm_config.database_url()


def _configure(**kwargs) -> None:
    context.configure(target_metadata=target_metadata, compare_type=True, **kwargs)


def run_migrations_offline() -> None:
    """Génère le SQL sans connexion (``alembic upgrade head --sql``)."""
    _configure(url=_url(), literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def _run(connection: Connection) -> None:
    _configure(connection=connection)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = context.config.attributes.get("connection")
    if connection is not None:
        _run(connection)
        return
    engine = create_engine(_url(), poolclass=pool.NullPool)
    try:
        with engine.connect() as conn:
            _run(conn)
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
