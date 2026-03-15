"""Alembic environment configuration.

Reads DATABASE_URL from .env and runs migrations using raw SQL (no ORM models).
"""

import os
from logging.config import fileConfig

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = None

def _get_url() -> str:
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    # SQLAlchemy needs postgresql:// not postgres://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    # psycopg3 driver
    if "psycopg" not in url and "postgresql://" in url:
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode — generates SQL script."""
    context.configure(
        url=_get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live database."""
    connectable = create_engine(_get_url())
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
