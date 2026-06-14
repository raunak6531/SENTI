"""
transform.py
------------
Phase 2 — Transformation layer for the Cinematic Sentiment & Trend Engine.

Responsibilities:
    1. Load raw JSON staging files produced by extract.py.
    2. Clean and type-cast data using Pandas.
    3. Run Gemini-powered aspect-based sentiment analysis on every review,
       using a strict Pydantic response schema to guarantee structured output.
    4. Merge AI analysis back into the reviews DataFrame.
    5. Persist two clean staging files ready for Phase 3 (Load):
       - transformed_movies.json
       - transformed_reviews.json

Run:
    python transform.py

Environment variables required (see .env.example):
    TMDB_API_KEY   — (loaded via config; not used here directly)
    DATABASE_URL   — (loaded via config; not used here directly)
    GEMINI_API_KEY — Your Google Gemini API key.
"""

from __future__ import annotations

import json
import logging
import time
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel, Field, field_validator

from config import settings

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger("transform")

# ---------------------------------------------------------------------------
# Gemini Client (initialised once)
# ---------------------------------------------------------------------------
_gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
_GEMINI_MODEL = "gemini-2.0-flash"

# ---------------------------------------------------------------------------
# Pydantic Schema — Gemini Structured Output Contract
# ---------------------------------------------------------------------------


class SentimentLabel(str, Enum):
    """Controlled vocabulary for overall review sentiment."""

    POSITIVE = "Positive"
    NEUTRAL = "Neutral"
    NEGATIVE = "Negative"


class ReviewSentiment(BaseModel):
    """
    Strict schema for Gemini's aspect-based sentiment analysis output.

    Each numeric score represents how positively a specific cinematic aspect
    was discussed in the review text (1 = very negative, 10 = very positive).
    A ``null`` value means the aspect was not meaningfully mentioned.
    """

    direction_score: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description="Score (1-10) for directorial quality, or null if not discussed.",
    )
    cinematography_score: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description="Score (1-10) for cinematography/visuals, or null if not discussed.",
    )
    pacing_score: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description="Score (1-10) for narrative pacing/editing, or null if not discussed.",
    )
    overall_sentiment: SentimentLabel = Field(
        description="Overall emotional tone of the review.",
    )

    @field_validator("direction_score", "cinematography_score", "pacing_score", mode="before")
    @classmethod
    def clamp_score(cls, v: Any) -> Optional[int]:
        """Coerce out-of-range integers to the nearest boundary instead of erroring."""
        if v is None:
            return None
        v = int(v)
        return max(1, min(10, v))


# Fallback returned when AI analysis fails — keeps the pipeline running.
_FALLBACK_SENTIMENT = ReviewSentiment(
    direction_score=None,
    cinematography_score=None,
    pacing_score=None,
    overall_sentiment=SentimentLabel.NEUTRAL,
)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_json(filepath: str | Path) -> list[dict[str, Any]]:
    """
    Load a JSON array from a local file.

    Args:
        filepath: Path to the JSON file.

    Returns:
        List of records as plain dictionaries.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the JSON does not decode to a list.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Staging file not found: {path.resolve()}\n"
            "Run extract.py first to generate the raw data files."
        )
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(data).__name__}.")
    logger.info("Loaded %d records from %s", len(data), path.resolve())
    return data  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Pandas Cleaning
# ---------------------------------------------------------------------------


def clean_movies(raw: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Select and type-cast relevant movie fields from raw TMDB data.

    Transformations applied:
    - Keep only the columns needed by the database schema.
    - Coerce ``release_date`` to ``datetime64`` (NaT for unparseable values).
    - Cast ``popularity`` to float and round to 4 decimal places.
    - Drop rows where ``id`` or ``title`` is missing.

    Args:
        raw: List of raw movie dicts from the TMDB API.

    Returns:
        Cleaned :class:`pandas.DataFrame`.
    """
    df = pd.DataFrame(raw)

    # Keep only the columns we care about (defensive: ignore absent columns)
    keep_cols = {"id": "movie_id", "title": "title", "release_date": "release_date", "popularity": "popularity_score"}
    available = {k: v for k, v in keep_cols.items() if k in df.columns}
    df = df[list(available.keys())].rename(columns=available)

    # Type casts
    df["movie_id"] = pd.to_numeric(df["movie_id"], errors="coerce").astype("Int64")
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["popularity_score"] = pd.to_numeric(df["popularity_score"], errors="coerce").round(4)

    # Drop rows missing primary key or title
    before = len(df)
    df.dropna(subset=["movie_id", "title"], inplace=True)
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d movie row(s) with missing movie_id or title.", dropped)

    df.drop_duplicates(subset=["movie_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info("Movies cleaned: %d records retained.", len(df))
    return df


def clean_reviews(raw: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Select, type-cast, and sanitise relevant review fields.

    Transformations applied:
    - Rename fields to match the database schema.
    - Coerce ``created_at`` to timezone-aware ``datetime64[ns, UTC]``.
    - Truncate ``content`` to ``MAX_REVIEW_CHARS`` characters to stay within
      Gemini's practical context budget.
    - Drop rows missing ``id``, ``movie_id``, or ``content``.

    Args:
        raw: List of raw review dicts (enriched with ``movie_id`` by extract.py).

    Returns:
        Cleaned :class:`pandas.DataFrame`.
    """
    df = pd.DataFrame(raw)

    keep_cols = {
        "id": "review_id",
        "movie_id": "movie_id",
        "author": "author",
        "content": "content",
        "created_at": "created_at",
    }
    available = {k: v for k, v in keep_cols.items() if k in df.columns}
    df = df[list(available.keys())].rename(columns=available)

    # Type casts
    df["movie_id"] = pd.to_numeric(df["movie_id"], errors="coerce").astype("Int64")
    df["created_at"] = pd.to_datetime(df.get("created_at"), errors="coerce", utc=True)

    # Sanitise content
    df["content"] = df["content"].astype(str).str.strip()
    long_mask = df["content"].str.len() > settings.MAX_REVIEW_CHARS
    if long_mask.any():
        logger.info(
            "Truncating %d review(s) to %d characters.", long_mask.sum(), settings.MAX_REVIEW_CHARS
        )
        df.loc[long_mask, "content"] = df.loc[long_mask, "content"].str[: settings.MAX_REVIEW_CHARS]

    # Drop rows missing critical fields
    before = len(df)
    df.dropna(subset=["review_id", "movie_id", "content"], inplace=True)
    df = df[df["content"].str.len() > 0]  # Remove effectively empty content
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d review row(s) with missing key fields.", dropped)

    df.drop_duplicates(subset=["review_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info("Reviews cleaned: %d records retained.", len(df))
    return df


# ---------------------------------------------------------------------------
# Gemini Sentiment Analysis
# ---------------------------------------------------------------------------


def _build_prompt(review_text: str) -> str:
    """
    Construct the system + user prompt for aspect-based sentiment analysis.

    Args:
        review_text: The cleaned review content.

    Returns:
        A formatted prompt string.
    """
    return (
        "You are an expert film critic analyst. Analyse the following movie review "
        "and return a structured JSON object.\n\n"
        "Rules:\n"
        "- Score each cinematic aspect on a scale of 1 (very negative) to 10 (very positive).\n"
        "- If a specific aspect is not meaningfully discussed, return null for its score.\n"
        "- Choose the overall_sentiment that best represents the review's emotional tone.\n\n"
        f"Review:\n---\n{review_text}\n---"
    )


def analyse_review(review_text: str, review_id: str) -> ReviewSentiment:
    """
    Call the Gemini API to perform aspect-based sentiment analysis on a review.

    Uses ``response_schema`` to constrain Gemini's output to the
    :class:`ReviewSentiment` Pydantic schema, ensuring clean, parseable JSON.

    Falls back to :data:`_FALLBACK_SENTIMENT` on any error so the pipeline
    continues uninterrupted.

    Args:
        review_text: The cleaned review text.
        review_id:   The review's ID (used only for log messages).

    Returns:
        A :class:`ReviewSentiment` instance (may be the fallback).
    """
    try:
        response = _gemini_client.models.generate_content(
            model=_GEMINI_MODEL,
            contents=_build_prompt(review_text),
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ReviewSentiment,
                temperature=0.1,  # Low temperature for deterministic structured output
            ),
        )

        # The SDK parses the response into the Pydantic model automatically
        # when response_schema is a BaseModel subclass.
        parsed: ReviewSentiment = response.parsed  # type: ignore[assignment]
        if parsed is None:
            raise ValueError("Gemini returned an empty parsed response.")
        return parsed

    except Exception as exc:  # noqa: BLE001 — broad catch is intentional here
        logger.error(
            "Gemini analysis failed for review_id='%s': %s — using fallback values.",
            review_id,
            exc,
        )
        return _FALLBACK_SENTIMENT


# ---------------------------------------------------------------------------
# AI Enrichment Orchestrator
# ---------------------------------------------------------------------------


def enrich_reviews_with_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Iterate through the cleaned reviews DataFrame and add AI sentiment columns.

    New columns added:
    - ``direction_score``       (int or pd.NA)
    - ``cinematography_score``  (int or pd.NA)
    - ``pacing_score``          (int or pd.NA)
    - ``overall_sentiment``     (str)

    A polite sleep of :attr:`~config.Settings.GEMINI_RATE_LIMIT_SLEEP` seconds
    is inserted between calls to avoid hitting the Gemini free-tier rate limit.

    Args:
        df: Cleaned reviews DataFrame (output of :func:`clean_reviews`).

    Returns:
        The same DataFrame with four new sentiment columns appended.
    """
    total = len(df)
    logger.info("Starting Gemini sentiment analysis for %d review(s)…", total)

    results: list[dict[str, Any]] = []

    for idx, row in df.iterrows():
        review_id: str = str(row["review_id"])
        content: str = str(row["content"])

        logger.info("  [%d/%d] Analysing review_id='%s'…", int(idx) + 1, total, review_id)  # type: ignore[arg-type]

        sentiment = analyse_review(content, review_id)
        results.append({
            "direction_score": sentiment.direction_score,
            "cinematography_score": sentiment.cinematography_score,
            "pacing_score": sentiment.pacing_score,
            "overall_sentiment": sentiment.overall_sentiment.value,
        })

        # Rate-limit courtesy sleep (skip after the last item)
        if int(idx) < total - 1:  # type: ignore[arg-type]
            time.sleep(settings.GEMINI_RATE_LIMIT_SLEEP)

    sentiment_df = pd.DataFrame(results, index=df.index)
    enriched = pd.concat([df, sentiment_df], axis=1)

    logger.info("Sentiment analysis complete.")
    return enriched


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _df_to_json_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Serialise a DataFrame to a JSON-safe list of dicts.

    Handles pandas ``NaT``, ``pd.NA``, and ``numpy.nan`` values by converting
    them to Python ``None`` (which serialises to JSON ``null``).

    Args:
        df: The DataFrame to serialise.

    Returns:
        List of row dicts, safe for ``json.dumps``.
    """
    records = df.where(df.notna(), other=None).to_dict(orient="records")
    # Convert any remaining non-serialisable types (e.g. Timestamp → str)
    for record in records:
        for key, val in record.items():
            if isinstance(val, pd.Timestamp):
                record[key] = val.isoformat()
    return records  # type: ignore[return-value]


def save_transformed(df: pd.DataFrame, filepath: str | Path) -> None:
    """
    Save a transformed DataFrame to a UTF-8 JSON file.

    Args:
        df:       The DataFrame to persist.
        filepath: Destination path.
    """
    path = Path(filepath)
    records = _df_to_json_records(df)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved %d record(s) → %s", len(records), path.resolve())


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_transformation() -> None:
    """
    Main transformation orchestrator.

    Steps:
        1. Load raw JSON staging files (output of extract.py).
        2. Clean movies and reviews DataFrames with Pandas.
        3. Enrich reviews with Gemini aspect-based sentiment scores.
        4. Persist two transformed staging files.
    """
    logger.info("=" * 60)
    logger.info("Cinematic Sentiment & Trend Engine — Transformation Phase")
    logger.info("=" * 60)

    # ── Step 1: Load ──────────────────────────────────────────────────────
    raw_movies = load_json(settings.RAW_MOVIES_PATH)
    raw_reviews = load_json(settings.RAW_REVIEWS_PATH)

    # ── Step 2: Clean ─────────────────────────────────────────────────────
    movies_df = clean_movies(raw_movies)
    reviews_df = clean_reviews(raw_reviews)

    # ── Step 3: AI Enrichment ─────────────────────────────────────────────
    enriched_reviews_df = enrich_reviews_with_sentiment(reviews_df)

    # ── Step 4: Persist ───────────────────────────────────────────────────
    save_transformed(movies_df, settings.TRANSFORMED_MOVIES_PATH)
    save_transformed(enriched_reviews_df, settings.TRANSFORMED_REVIEWS_PATH)

    logger.info("-" * 60)
    logger.info(
        "Transformation complete. %d movies, %d enriched reviews saved.",
        len(movies_df),
        len(enriched_reviews_df),
    )
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_transformation()
