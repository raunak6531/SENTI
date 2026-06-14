-- =============================================================================
-- Cinematic Sentiment & Trend Engine — Database Schema
-- Target: Supabase (PostgreSQL)
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Table: movies
-- Stores core metadata for each film pulled from the TMDB API.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS movies (
    movie_id        INTEGER         PRIMARY KEY,        -- TMDB movie ID (natural key)
    title           TEXT            NOT NULL,
    release_date    DATE,                               -- NULL-safe: some titles lack a date
    popularity_score NUMERIC(10, 4)                     -- TMDB floating-point popularity metric
);

-- Index to speed up popularity-based ranking queries
CREATE INDEX IF NOT EXISTS idx_movies_popularity ON movies (popularity_score DESC);

-- -----------------------------------------------------------------------------
-- Table: reviews
-- Stores user reviews associated with a movie.
-- A single movie may have many reviews (one-to-many relationship).
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS reviews (
    review_id   TEXT            PRIMARY KEY,            -- TMDB review UUID (natural key)
    movie_id    INTEGER         NOT NULL
                    REFERENCES movies (movie_id)
                    ON DELETE CASCADE,                  -- Cascade removes orphan reviews
    author      TEXT,
    content     TEXT,
    created_at  TIMESTAMPTZ                             -- Store with timezone (UTC from TMDB)
);

-- Index to efficiently retrieve all reviews for a given movie
CREATE INDEX IF NOT EXISTS idx_reviews_movie_id ON reviews (movie_id);
