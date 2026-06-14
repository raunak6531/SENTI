# 🎬 Cinematic Sentiment & Trend Engine

> An end-to-end Data Engineering ETL pipeline that pulls trending movie data from the TMDB API, runs **AI-powered aspect-based sentiment analysis** on user reviews using **Groq / Llama 3**, and loads the enriched dataset into a **Supabase PostgreSQL** database.

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ETL Pipeline                             │
│                                                                 │
│  ┌───────────┐    ┌─────────────────┐    ┌──────────────────┐  │
│  │ TMDB API  │───▶│  extract.py     │───▶│ raw_movies.json  │  │
│  │           │    │  (Phase 1)      │    │ raw_reviews.json │  │
│  └───────────┘    └─────────────────┘    └────────┬─────────┘  │
│                                                   │            │
│  ┌───────────┐    ┌─────────────────┐             │            │
│  │ Groq API  │───▶│  transform.py   │◀────────────┘            │
│  │ Llama 3.1 │    │  (Phase 2)      │                          │
│  └───────────┘    └────────┬────────┘                          │
│                            │                                   │
│                   ┌────────▼────────┐                          │
│                   │transformed_     │                          │
│                   │movies.json      │                          │
│                   │transformed_     │                          │
│                   │reviews.json     │                          │
│                   └────────┬────────┘                          │
│                            │                                   │
│                   ┌────────▼────────┐    ┌──────────────────┐  │
│                   │  load.py        │───▶│ Supabase         │  │
│                   │  (Phase 3)      │    │ PostgreSQL       │  │
│                   └─────────────────┘    └──────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

- **Extraction** — Fetches the top 20 trending movies and their first-page reviews from the TMDB API with automatic retry logic and rate-limit handling.
- **Transformation** — Cleans data with Pandas (type coercion, date formatting, review truncation) and enriches each review with AI-generated scores using the Groq API.
- **Sentiment Analysis** — Llama 3.1 8B assigns structured aspect scores per review:
  - 🎬 `direction_score` (1–10 or null)
  - 📷 `cinematography_score` (1–10 or null)
  - ⏱️ `pacing_score` (1–10 or null)
  - 🧭 `overall_sentiment` (Positive / Neutral / Negative)
- **Load** — Performs idempotent upserts into Supabase PostgreSQL using `INSERT ... ON CONFLICT DO UPDATE`. Re-running the pipeline is always safe.
- **Schema Migration** — `load.py` automatically runs `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` before each load, keeping the database schema in sync without manual intervention.

---

## 🗂️ Project Structure

```
cinema-trend-analysis/
│
├── extract.py            # Phase 1: Fetch data from TMDB API → local JSON
├── transform.py          # Phase 2: Clean with Pandas + AI sentiment (Groq)
├── load.py               # Phase 3: Upsert into Supabase PostgreSQL
│
├── config.py             # Centralised settings loaded from .env
├── schema.sql            # Initial PostgreSQL table definitions
│
├── requirements.txt      # Pinned Python dependencies
├── .env.example          # Template for environment variables
├── .gitignore            # Excludes .env, *.json staging files, caches
└── README.md             # You are here
```

> **Staging files** (`raw_*.json`, `transformed_*.json`) are generated at runtime and excluded from version control via `.gitignore`.

---

## 🗃️ Database Schema

```sql
-- Core movie metadata
CREATE TABLE movies (
    movie_id         INTEGER      PRIMARY KEY,
    title            TEXT         NOT NULL,
    release_date     DATE,
    popularity_score NUMERIC(10,4)
);

-- User reviews + AI sentiment scores (columns added by load.py migration)
CREATE TABLE reviews (
    review_id             TEXT    PRIMARY KEY,
    movie_id              INTEGER NOT NULL REFERENCES movies(movie_id) ON DELETE CASCADE,
    author                TEXT,
    content               TEXT,
    created_at            TIMESTAMPTZ,
    direction_score       INTEGER,        -- 1-10 or NULL
    cinematography_score  INTEGER,        -- 1-10 or NULL
    pacing_score          INTEGER,        -- 1-10 or NULL
    overall_sentiment     TEXT            -- 'Positive' | 'Neutral' | 'Negative'
);
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- A [Supabase](https://supabase.com) project with the schema applied (see [Database Setup](#-database-setup))
- A [TMDB API key](https://www.themoviedb.org/settings/api) (free)
- A [Groq API key](https://console.groq.com/keys) (free tier: 30 RPM)

### 1. Clone the repository

```bash
git clone https://github.com/raunak6531/SENTI.git
cd SENTI
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your credentials:

```env
TMDB_API_KEY=your_tmdb_api_key_here
GROQ_API_KEY=gsk_your_groq_api_key_here
DATABASE_URL=postgresql://postgres.[ref]:[password]@aws-0-region.pooler.supabase.com:5432/postgres
```

> ⚠️ **Never commit your `.env` file.** It is already excluded by `.gitignore`.

### 4. Apply the database schema

Run [schema.sql](schema.sql) once in the **Supabase SQL Editor** (Dashboard → SQL Editor → New query → paste → Run).

---

## 🚀 Running the Pipeline

Run each phase in order:

```bash
# Phase 1 — Extract raw data from TMDB
python extract.py

# Phase 2 — Clean + AI sentiment analysis (takes ~3 min for 89 reviews at 30 RPM)
python transform.py

# Phase 3 — Load into Supabase
python load.py
```

Each script logs its progress to stdout:

```
2026-06-14 17:34:41  INFO  load — ============================================================
2026-06-14 17:34:41  INFO  load — Cinematic Sentiment & Trend Engine — Load Phase
2026-06-14 17:34:47  INFO  load — Movies upsert complete — 20 row(s) affected.
2026-06-14 17:34:52  INFO  load — Reviews upsert complete — 89 row(s) affected.
2026-06-14 17:34:52  INFO  load — Load complete. 20 movie(s), 89 review(s) upserted into Supabase.
```

---

## 🛡️ Resilience & Safety

| Concern | How it's handled |
|---|---|
| TMDB rate limits | `urllib3.Retry` with exponential back-off + `Retry-After` header parsing |
| Groq rate limits | 2.1 s sleep between requests (≤ 30 RPM free-tier limit) |
| Groq API failure | Per-review `try/except` — falls back to `null` scores so the pipeline never crashes |
| Malformed AI JSON | `json.JSONDecodeError` + `pydantic.ValidationError` caught separately |
| Duplicate pipeline runs | All upserts use `ON CONFLICT DO UPDATE` — idempotent by design |
| Schema drift | `ADD COLUMN IF NOT EXISTS` migration runs automatically before every load |
| Partial DB writes | Each upsert runs inside `engine.begin()` — full rollback on any failure |

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Data wrangling | Pandas 2.2 |
| AI / LLM | Groq API · Llama 3.1 8B Instant |
| Schema validation | Pydantic v2 |
| Database ORM | SQLAlchemy 2.0 |
| Database driver | psycopg2-binary |
| Database | Supabase (PostgreSQL 15) |
| External API | TMDB v3 REST API |
| HTTP client | requests + urllib3 Retry |
| Config | python-dotenv |

---

## 🔮 Roadmap

- [ ] **Phase 4** — Dashboard / visualisation (sentiment trends, top-rated films by aspect)
- [ ] **Phase 5** — Scheduled automation (cron / GitHub Actions to refresh data weekly)
- [ ] Expand to fetch multiple pages of reviews per movie
- [ ] Add genre and cast metadata from TMDB

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
