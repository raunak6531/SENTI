"""
load.py
-------
Phase 3 — Load layer for the Cinematic Sentiment & Trend Engine.

Responsibilities:
    1. Ensure the database schema is up-to-date by running an idempotent
       migration that adds the four sentiment columns to the reviews table
       (safe to run multiple times; uses ADD COLUMN IF NOT EXISTS).
    2. Read transformed_movies.json and upsert rows into the movies table.
    3. Read transformed_reviews.json and upsert rows into the reviews table.

Both upserts use PostgreSQL's INSERT … ON CONFLICT DO UPDATE syntax via
SQLAlchemy's dialect-specific helper, so re-running load.py is always safe
and will simply refresh stale data rather than creating duplicates.

All database writes are wrapped in a single transaction:  if any statement
fails the entire load is rolled back and no partial data is committed.

Run:
    python load.py

Prerequisites:
    - schema.sql must already be applied to the Supabase database.
    - transformed_movies.json and transformed_reviews.json must exist
      (run extract.py then transform.py first).

Environment variables required (see .env.example):
    DATABASE_URL — Supabase PostgreSQL connection string.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine

from config import settings

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger("load")

# ---------------------------------------------------------------------------
# Table names (single source of truth)
# ---------------------------------------------------------------------------
MOVIES_TABLE = "movies"
REVIEWS_TABLE = "reviews"

# ---------------------------------------------------------------------------
# Idempotent Migration SQL
# ---------------------------------------------------------------------------
# ADD COLUMN IF NOT EXISTS is a PostgreSQL extension that makes this block
# safe to run on every invocation — existing columns are silently skipped.
MIGRATION_SQL = """
ALTER TABLE reviews
    ADD COLUMN IF NOT EXISTS direction_score       INTEGER,
    ADD COLUMN IF NOT EXISTS cinematography_score  INTEGER,
    ADD COLUMN IF NOT EXISTS pacing_score          INTEGER,
    ADD COLUMN IF NOT EXISTS overall_sentiment     TEXT;
"""


# ---------------------------------------------------------------------------
# Engine Factory
# ---------------------------------------------------------------------------

def build_engine() -> Engine:
    """
    Create and return a SQLAlchemy :class:`~sqlalchemy.engine.Engine` using
    the connection string from :attr:`~config.Settings.DATABASE_URL`.

    The engine is configured with a conservative pool size suitable for a
    short-lived ETL script (one connection, no overflow).

    Returns:
        A connected :class:`~sqlalchemy.engine.Engine`.

    Raises:
        sqlalchemy.exc.OperationalError: If the database is unreachable.
    """
    engine = create_engine(
        settings.DATABASE_URL,
        pool_size=1,
        max_overflow=0,
        pool_pre_ping=True,  # verify the connection before using it
    )
    logger.info("SQLAlchemy engine created for host: %s", engine.url.host)
    return engine


# ---------------------------------------------------------------------------
# Data Loading Helpers
# ---------------------------------------------------------------------------

def _load_json_to_df(filepath: str | Path) -> pd.DataFrame:
    """
    Load a transformed JSON staging file into a :class:`pandas.DataFrame`.

    Args:
        filepath: Path to the JSON file.

    Returns:
        A :class:`pandas.DataFrame` containing the records.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Staging file not found: {path.resolve()}\n"
            "Run transform.py first to generate the transformed data files."
        )
    with path.open(encoding="utf-8") as fh:
        records: list[dict[str, Any]] = json.load(fh)
    df = pd.DataFrame(records)
    logger.info("Loaded %d records from %s", len(df), path.resolve())
    return df


def _prepare_movies(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Select and type-cast the movies DataFrame columns to match the
    ``movies`` table schema exactly.

    Columns mapped:
        movie_id        — INTEGER (primary key)
        title           — TEXT
        release_date    — DATE (ISO 8601 string or None)
        popularity_score — NUMERIC

    Args:
        df: Raw loaded movies DataFrame.

    Returns:
        A list of row dicts ready for bulk upsert.
    """
    schema_cols = ["movie_id", "title", "release_date", "popularity_score"]
    missing = [c for c in schema_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Movies DataFrame is missing required columns: {missing}")

    df = df[schema_cols].copy()

    # Coerce types
    df["movie_id"] = pd.to_numeric(df["movie_id"], errors="coerce").astype("Int64")
    df["popularity_score"] = pd.to_numeric(df["popularity_score"], errors="coerce")

    # Convert date strings to Python date objects (None for NaT)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce").dt.date
    df["release_date"] = df["release_date"].where(df["release_date"].notna(), other=None)

    df.dropna(subset=["movie_id", "title"], inplace=True)

    # Replace pandas NA/NaN with None so psycopg2 writes NULL correctly
    records = df.where(df.notna(), other=None).to_dict(orient="records")
    logger.info("Movies prepared: %d rows to upsert.", len(records))
    return records


def _prepare_reviews(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Select and type-cast the reviews DataFrame columns to match the
    (migrated) ``reviews`` table schema exactly.

    Columns mapped:
        review_id             — TEXT (primary key)
        movie_id              — INTEGER (foreign key)
        author                — TEXT
        content               — TEXT
        created_at            — TIMESTAMPTZ (ISO 8601 string or None)
        direction_score       — INTEGER or NULL
        cinematography_score  — INTEGER or NULL
        pacing_score          — INTEGER or NULL
        overall_sentiment     — TEXT

    Args:
        df: Raw loaded reviews DataFrame.

    Returns:
        A list of row dicts ready for bulk upsert.
    """
    schema_cols = [
        "review_id", "movie_id", "author", "content", "created_at",
        "direction_score", "cinematography_score", "pacing_score", "overall_sentiment",
    ]
    missing = [c for c in schema_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Reviews DataFrame is missing required columns: {missing}")

    df = df[schema_cols].copy()

    # Coerce types
    df["movie_id"] = pd.to_numeric(df["movie_id"], errors="coerce").astype("Int64")
    for score_col in ("direction_score", "cinematography_score", "pacing_score"):
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce").where(
            df[score_col].notna(), other=None
        )

    df.dropna(subset=["review_id", "movie_id"], inplace=True)

    records = df.where(df.notna(), other=None).to_dict(orient="records")
    logger.info("Reviews prepared: %d rows to upsert.", len(records))
    return records


# ---------------------------------------------------------------------------
# Database Operations
# ---------------------------------------------------------------------------

def run_migration(engine: Engine) -> None:
    """
    Execute the idempotent schema migration inside its own transaction.

    Adds the four sentiment columns to the ``reviews`` table if they do not
    already exist.  Safe to call on every pipeline run.

    Args:
        engine: Active SQLAlchemy engine.
    """
    logger.info("Running schema migration (ADD COLUMN IF NOT EXISTS)…")
    with engine.begin() as conn:
        conn.execute(text(MIGRATION_SQL))
    logger.info("Migration complete.")


def upsert_movies(engine: Engine, records: list[dict[str, Any]]) -> None:
    """
    Upsert movie records into the ``movies`` table.

    Conflict resolution strategy (ON CONFLICT movie_id DO UPDATE):
        - Updates ``popularity_score`` and ``title`` with the latest values.
        - Leaves ``release_date`` unchanged (TMDB date data rarely changes).

    All rows are written in a single transaction — any failure rolls back
    the entire batch.

    Args:
        engine:  Active SQLAlchemy engine.
        records: List of row dicts from :func:`_prepare_movies`.
    """
    if not records:
        logger.warning("No movie records to upsert — skipping.")
        return

    logger.info("Upserting %d movie row(s) into '%s'…", len(records), MOVIES_TABLE)

    stmt = pg_insert(
        # Build a lightweight Table reference so SQLAlchemy can construct the
        # INSERT statement without requiring a full ORM model definition.
        _reflect_table(engine, MOVIES_TABLE)
    ).values(records)

    stmt = stmt.on_conflict_do_update(
        index_elements=["movie_id"],
        set_={
            "title": stmt.excluded.title,
            "popularity_score": stmt.excluded.popularity_score,
        },
    )

    with engine.begin() as conn:
        result = conn.execute(stmt)

    logger.info("Movies upsert complete — %d row(s) affected.", result.rowcount)


def upsert_reviews(engine: Engine, records: list[dict[str, Any]]) -> None:
    """
    Upsert review records into the ``reviews`` table.

    Conflict resolution strategy (ON CONFLICT review_id DO UPDATE):
        - Updates all mutable columns (content, sentiment scores, etc.) with
          the latest values from the Transform phase.

    All rows are written in a single transaction.

    Args:
        engine:  Active SQLAlchemy engine.
        records: List of row dicts from :func:`_prepare_reviews`.
    """
    if not records:
        logger.warning("No review records to upsert — skipping.")
        return

    logger.info("Upserting %d review row(s) into '%s'…", len(records), REVIEWS_TABLE)

    stmt = pg_insert(_reflect_table(engine, REVIEWS_TABLE)).values(records)

    stmt = stmt.on_conflict_do_update(
        index_elements=["review_id"],
        set_={
            "author": stmt.excluded.author,
            "content": stmt.excluded.content,
            "created_at": stmt.excluded.created_at,
            "direction_score": stmt.excluded.direction_score,
            "cinematography_score": stmt.excluded.cinematography_score,
            "pacing_score": stmt.excluded.pacing_score,
            "overall_sentiment": stmt.excluded.overall_sentiment,
        },
    )

    with engine.begin() as conn:
        result = conn.execute(stmt)

    logger.info("Reviews upsert complete — %d row(s) affected.", result.rowcount)


# ---------------------------------------------------------------------------
# Table Reflection Helper
# ---------------------------------------------------------------------------

def _reflect_table(engine: Engine, table_name: str):  # type: ignore[return]
    """
    Reflect a PostgreSQL table from the live database into a SQLAlchemy
    :class:`~sqlalchemy.schema.Table` object.

    Reflection reads the actual column definitions from the database,
    so the upsert statements always match the real schema — no manual
    column mapping required.

    Args:
        engine:     Active SQLAlchemy engine.
        table_name: The name of the table to reflect.

    Returns:
        A :class:`~sqlalchemy.schema.Table` instance.
    """
    from sqlalchemy import MetaData, Table

    metadata = MetaData()
    table = Table(table_name, metadata, autoload_with=engine)
    return table


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_load() -> None:
    """
    Main load orchestrator.

    Steps:
        1. Build the SQLAlchemy engine.
        2. Run the idempotent schema migration.
        3. Load and prepare movies from transformed_movies.json.
        4. Upsert movies into PostgreSQL.
        5. Load and prepare reviews from transformed_reviews.json.
        6. Upsert reviews into PostgreSQL.
    """
    logger.info("=" * 60)
    logger.info("Cinematic Sentiment & Trend Engine — Load Phase")
    logger.info("=" * 60)

    # ── Step 1: Engine ────────────────────────────────────────────────────
    engine = build_engine()

    # ── Step 2: Migration ─────────────────────────────────────────────────
    run_migration(engine)

    # ── Step 3 & 4: Movies ────────────────────────────────────────────────
    movies_df = _load_json_to_df(settings.TRANSFORMED_MOVIES_PATH)
    movie_records = _prepare_movies(movies_df)
    upsert_movies(engine, movie_records)

    # ── Step 5 & 6: Reviews ───────────────────────────────────────────────
    reviews_df = _load_json_to_df(settings.TRANSFORMED_REVIEWS_PATH)
    review_records = _prepare_reviews(reviews_df)
    upsert_reviews(engine, review_records)

    logger.info("-" * 60)
    logger.info(
        "Load complete. %d movie(s), %d review(s) upserted into Supabase.",
        len(movie_records),
        len(review_records),
    )
    logger.info("=" * 60)

    engine.dispose()


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_load()
