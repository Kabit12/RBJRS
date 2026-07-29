"""
Schema Sync Utility
====================
Repairs schema drift between the SQLAlchemy models and an existing database file.

Why this is needed:
    create_app() calls db.create_all(), which creates *missing tables* but never
    adds *new columns* to tables that already exist. So a database created before
    a model gained a column keeps working until some query selects that column,
    then fails with:

        sqlite3.OperationalError: no such column: resumes.embedding

Usage:
    python sync_schema.py            # show drift, apply fixes, stamp alembic
    python sync_schema.py --check    # report drift only, change nothing

Safe to run repeatedly — it only ever ADDs columns that are missing, and makes a
timestamped backup of the SQLite file before touching it. It never drops or
alters existing columns, so no data is lost.
"""

import os
import shutil
import sys
from datetime import datetime, timezone

from sqlalchemy import inspect, text

from app import create_app
from app.config import config_map
from app.extensions import db


def sql_literal(value):
    """Render a Python default as a SQL literal for a DEFAULT clause."""
    if isinstance(value, bool):
        return '1' if value else '0'
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def default_clause(column):
    """
    Build the ' NOT NULL DEFAULT x' suffix for a new column.

    SQLite refuses ADD COLUMN ... NOT NULL without a default, so a non-nullable
    column needs one. Prefer the model's own default; fall back to a harmless
    zero-value for the column type.
    """
    if column.nullable:
        return ''

    default = column.default
    if default is not None and not default.is_callable:
        return f' NOT NULL DEFAULT {sql_literal(default.arg)}'

    # No usable scalar default (missing, or a Python callable like datetime.now)
    python_type = None
    try:
        python_type = column.type.python_type
    except NotImplementedError:
        pass

    if python_type is bool:
        return ' NOT NULL DEFAULT 0'
    if python_type in (int, float):
        return ' NOT NULL DEFAULT 0'
    if python_type is datetime:
        return ' NOT NULL DEFAULT CURRENT_TIMESTAMP'
    return " NOT NULL DEFAULT ''"


def find_drift(engine):
    """Return [(table_name, column), ...] for every column missing from the DB."""
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    drift = []

    for table in db.metadata.sorted_tables:
        if table.name not in existing_tables:
            # create_all() handles whole missing tables; nothing to patch here.
            continue
        existing_columns = {c['name'] for c in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name not in existing_columns:
                drift.append((table.name, column))

    return drift


def backup_sqlite(engine):
    """Copy the SQLite file next to itself with a timestamp. No-op for other DBs."""
    if engine.dialect.name != 'sqlite':
        return None
    path = engine.url.database
    if not path or path == ':memory:' or not os.path.exists(path):
        return None
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    backup_path = f'{path}.{stamp}.bak'
    shutil.copy2(path, backup_path)
    return backup_path


def stamp_alembic_if_needed(engine):
    """
    Give Alembic a baseline if the database has no alembic_version table.

    A database built purely by create_all() has no migration history, so the
    first `flask db upgrade` would replay migrations whose columns already
    exist and crash. Stamping head records "this DB is already current".
    """
    if 'alembic_version' in inspect(engine).get_table_names():
        return False

    from flask_migrate import stamp
    stamp()
    return True


def main():
    check_only = '--check' in sys.argv

    # DevelopmentConfig sets SQLALCHEMY_ECHO, which would bury this script's
    # output under thousands of lines of logged SQL. Clear it on the config
    # classes themselves: the engine reads this when create_app() builds it,
    # so it has to be off before that call, not after.
    for config_class in set(config_map.values()):
        config_class.SQLALCHEMY_ECHO = False

    app = create_app()
    with app.app_context():
        engine = db.engine

        # Picks up any entirely new tables before we diff columns.
        if not check_only:
            db.create_all()

        drift = find_drift(engine)

        if not drift:
            print('Schema is in sync - no missing columns.')
        else:
            print(f'Found {len(drift)} missing column(s):')
            for table_name, column in drift:
                print(f'  {table_name}.{column.name} ({column.type})')

            if check_only:
                print('\n--check specified; no changes made.')
                return 1

            backup_path = backup_sqlite(engine)
            if backup_path:
                print(f'\nBackup written to {backup_path}')

            print()
            with engine.begin() as conn:
                for table_name, column in drift:
                    type_sql = column.type.compile(engine.dialect)
                    statement = (
                        f'ALTER TABLE {table_name} '
                        f'ADD COLUMN {column.name} {type_sql}{default_clause(column)}'
                    )
                    conn.execute(text(statement))
                    print(f'applied: {statement}')

            remaining = find_drift(engine)
            if remaining:
                print(f'\nWARNING: {len(remaining)} column(s) still missing: {remaining}')
                return 1
            print(f'\nAdded {len(drift)} column(s). Schema now matches the models.')

        if not check_only and stamp_alembic_if_needed(engine):
            print('Stamped Alembic at head (database had no migration history).')

    return 0


if __name__ == '__main__':
    sys.exit(main())
