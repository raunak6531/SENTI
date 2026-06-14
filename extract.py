"""
extract.py
----------
Phase 1 — Extraction layer for the Cinematic Sentiment & Trend Engine.

Responsibilities:
    1. Fetch the top N currently trending movies from the TMDB API.
    2. For each movie, fetch the first page of user reviews.
    3. Apply error handling, structured logging, and API rate-limit back-off.
    4. Persist raw data to local JSON files (raw_movies.json / raw_reviews.json)
       as a staging "Data Lake" before the Transform step.

Run:
    python extract.py

Environment variables required (see .env.example):
    TMDB_API_KEY  — Your TMDB v3 API key.
    DATABASE_URL  — Supabase connection string (not used here, loaded for consistency).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import requests
from requests import Response, Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import settings

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger("extract")

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
MovieRecord = dict[str, Any]
ReviewRecord = dict[str, Any]


# ---------------------------------------------------------------------------
# HTTP Session Factory
# ---------------------------------------------------------------------------

def _build_session() -> Session:
    """
    Build a :class:`requests.Session` with automatic retries on transient
    network failures (5xx responses, connection errors, read timeouts).

    Returns:
        A configured :class:`requests.Session`.
    """
    retry_strategy = Retry(
        total=3,
        backoff_factor=1.5,           # waits: 0 s, 1.5 s, 3 s between retries
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# ---------------------------------------------------------------------------
# TMDB API Helpers
# ---------------------------------------------------------------------------

def _tmdb_get(
    session: Session,
    endpoint: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Perform an authenticated GET request against the TMDB v3 REST API.

    Args:
        session:  The shared :class:`requests.Session`.
        endpoint: API path, e.g. ``/trending/movie/week``.
        params:   Optional query-string parameters (``api_key`` is injected
                  automatically).

    Returns:
        Parsed JSON response as a dictionary.

    Raises:
        requests.HTTPError: On non-2xx responses after all retries are
            exhausted.
        requests.RequestException: On network-level failures.
    """
    url = f"{settings.TMDB_BASE_URL}{endpoint}"
    all_params: dict[str, Any] = {"api_key": settings.TMDB_API_KEY, **(params or {})}

    logger.debug("GET %s  params=%s", url, {k: v for k, v in all_params.items() if k != "api_key"})

    response: Response = session.get(url, params=all_params, timeout=10)

    # TMDB returns 429 with a Retry-After header when rate-limited.
    if response.status_code == 429:
        retry_after = int(response.headers.get("Retry-After", 2))
        logger.warning("Rate-limited by TMDB. Sleeping %d seconds…", retry_after)
        time.sleep(retry_after)
        response = session.get(url, params=all_params, timeout=10)

    response.raise_for_status()
    return response.json()  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Extraction Functions
# ---------------------------------------------------------------------------

def fetch_trending_movies(session: Session) -> list[MovieRecord]:
    """
    Fetch the top N currently trending movies from the TMDB weekly
    trending endpoint, paginating if necessary.

    TMDB returns 20 results per page, so a single page is sufficient for
    the default limit of 20.

    Args:
        session: Shared HTTP session.

    Returns:
        A list of raw movie dictionaries from the TMDB API, trimmed to
        :attr:`~config.Settings.TRENDING_MOVIE_LIMIT` items.
    """
    logger.info("Fetching trending movies (limit=%d)…", settings.TRENDING_MOVIE_LIMIT)

    data = _tmdb_get(session, "/trending/movie/week", params={"language": "en-US"})
    movies: list[MovieRecord] = data.get("results", [])

    trimmed = movies[: settings.TRENDING_MOVIE_LIMIT]
    logger.info("Retrieved %d trending movies.", len(trimmed))
    return trimmed


def fetch_reviews_for_movie(session: Session, movie_id: int, title: str) -> list[ReviewRecord]:
    """
    Fetch the first page of user reviews for a specific movie.

    Each review record is augmented with ``movie_id`` and ``movie_title``
    fields for easier downstream processing.

    Args:
        session:  Shared HTTP session.
        movie_id: TMDB integer movie ID.
        title:    Human-readable movie title (used in log messages).

    Returns:
        A list of raw review dictionaries, each enriched with ``movie_id``
        and ``movie_title``. Returns an empty list if the movie has no
        reviews or if the request fails.
    """
    logger.info("  Fetching reviews for '%s' (id=%d)…", title, movie_id)

    try:
        data = _tmdb_get(
            session,
            f"/movie/{movie_id}/reviews",
            params={"language": "en-US", "page": 1},
        )
    except requests.HTTPError as exc:
        # A 404 here means the movie has no reviews yet — non-fatal.
        if exc.response is not None and exc.response.status_code == 404:
            logger.warning("    No reviews found for movie id=%d (404).", movie_id)
            return []
        logger.error("    HTTP error fetching reviews for id=%d: %s", movie_id, exc)
        return []
    except requests.RequestException as exc:
        logger.error("    Network error fetching reviews for id=%d: %s", movie_id, exc)
        return []

    reviews: list[ReviewRecord] = data.get("results", [])

    # Enrich each review with its parent movie context
    for review in reviews:
        review["movie_id"] = movie_id
        review["movie_title"] = title

    logger.info("    Found %d review(s).", len(reviews))
    return reviews


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_to_json(data: list[dict[str, Any]], filepath: str | Path) -> None:
    """
    Serialise a list of dictionaries to a UTF-8 encoded JSON file.

    The file is overwritten on each run to reflect the latest extraction.

    Args:
        data:     The data to serialise.
        filepath: Destination path for the JSON file.
    """
    path = Path(filepath)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved %d record(s) → %s", len(data), path.resolve())


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_extraction() -> None:
    """
    Main extraction orchestrator.

    Steps:
        1. Build an HTTP session.
        2. Fetch trending movies.
        3. For each movie, fetch reviews with a small inter-request sleep.
        4. Persist both datasets to local JSON staging files.
    """
    logger.info("=" * 60)
    logger.info("Cinematic Sentiment & Trend Engine — Extraction Phase")
    logger.info("=" * 60)

    session = _build_session()

    # ── Step 1: Trending movies ──────────────────────────────────────────
    try:
        movies = fetch_trending_movies(session)
    except requests.RequestException as exc:
        logger.critical("Failed to fetch trending movies: %s — aborting.", exc)
        return

    if not movies:
        logger.warning("No trending movies returned. Exiting.")
        return

    # ── Step 2: Reviews per movie ────────────────────────────────────────
    all_reviews: list[ReviewRecord] = []

    for index, movie in enumerate(movies, start=1):
        movie_id: int = movie["id"]
        title: str = movie.get("title", "Unknown")

        logger.info("[%d/%d] Processing '%s'", index, len(movies), title)

        reviews = fetch_reviews_for_movie(session, movie_id, title)
        all_reviews.extend(reviews)

        # Polite delay between requests to avoid hitting rate limits
        if index < len(movies):
            time.sleep(settings.API_RATE_LIMIT_SLEEP)

    # ── Step 3: Persist raw data ─────────────────────────────────────────
    save_to_json(movies, settings.RAW_MOVIES_PATH)
    save_to_json(all_reviews, settings.RAW_REVIEWS_PATH)

    logger.info("-" * 60)
    logger.info(
        "Extraction complete. %d movies, %d reviews saved to disk.",
        len(movies),
        len(all_reviews),
    )
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_extraction()
