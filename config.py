"""
config.py
---------
Central configuration module for the Cinematic Sentiment & Trend Engine.

Loads all sensitive credentials from a .env file using python-dotenv so that
no secrets are ever hard-coded in source files.

Usage:
    from config import settings
    print(settings.TMDB_API_KEY)
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load variables from a .env file in the project root (if it exists).
# Variables already present in the OS environment are NOT overwritten.
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Immutable application settings loaded from environment variables."""

    TMDB_API_KEY: str
    DATABASE_URL: str
    GROQ_API_KEY: str

    # TMDB REST API base URL (v3)
    TMDB_BASE_URL: str = "https://api.themoviedb.org/3"

    # Maximum trending movies to fetch per run
    TRENDING_MOVIE_LIMIT: int = 20

    # Seconds to wait between TMDB API calls to respect rate limits
    API_RATE_LIMIT_SLEEP: float = 0.25

    # Seconds to wait between Groq API calls — 2.1 s keeps us under 30 RPM
    GEMINI_RATE_LIMIT_SLEEP: float = 2.1

    # Maximum review character length before truncation (LLM context budget)
    MAX_REVIEW_CHARS: int = 2000

    # Local staging file paths — raw Data Lake (Phase 1 output)
    RAW_MOVIES_PATH: str = "raw_movies.json"
    RAW_REVIEWS_PATH: str = "raw_reviews.json"

    # Local staging file paths — transformed (Phase 2 output)
    TRANSFORMED_MOVIES_PATH: str = "transformed_movies.json"
    TRANSFORMED_REVIEWS_PATH: str = "transformed_reviews.json"


def _require_env(var_name: str) -> str:
    """
    Retrieve a required environment variable or raise a clear error.

    Args:
        var_name: The name of the environment variable.

    Returns:
        The string value of the environment variable.

    Raises:
        EnvironmentError: If the variable is missing or empty.
    """
    value = os.getenv(var_name, "").strip()
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{var_name}' is not set. "
            "Copy .env.example to .env and fill in your credentials."
        )
    return value


# Singleton settings instance — import this throughout the project.
settings = Settings(
    TMDB_API_KEY=_require_env("TMDB_API_KEY"),
    DATABASE_URL=_require_env("DATABASE_URL"),
    GROQ_API_KEY=_require_env("GROQ_API_KEY"),
)
