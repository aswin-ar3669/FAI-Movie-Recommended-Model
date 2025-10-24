import pandas as pd
import numpy as np

def load_tmdb_5000(movies_path: str, credits_path: str):
    """
    Load and merge TMDB 5000 Movies and Credits CSVs from Kaggle into a single
    cleaned DataFrame with columns required by the app.

    Inputs:
    - tmdb_5000_movies.csv
    - tmdb_5000_credits.csv

    Returns columns (when present in the source):
    - id (int), title (str), overview (str), vote_average (float), vote_count (int), popularity (float)
    - runtime (float), budget (float), revenue (float)
    - release_date (datetime), release_year (int)
    - genres (str JSON), keywords (str JSON), cast (str JSON), crew (str JSON), production_countries (str JSON)
    """
    # Read CSVs
    movies = pd.read_csv(movies_path, encoding="utf-8")
    credits = pd.read_csv(credits_path, encoding="utf-8")

    # Standardize credits column names (Kaggle: movie_id, title, cast, crew)
    if 'movie_id' in credits.columns:
        credits = credits.rename(columns={'movie_id': 'id'})

    # Sanity check for join key
    if 'id' not in movies.columns:
        raise ValueError("tmdb_5000_movies.csv must contain column 'id'")

    # Merge credits into movies on 'id'
    credits_keep = [c for c in ['id','cast','crew'] if c in credits.columns]
    df = movies.merge(credits[credits_keep], on='id', how='left')

    # Type conversions
    numeric_cols = ['vote_average', 'vote_count', 'popularity', 'budget', 'revenue', 'runtime']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Title normalization
    if 'title' not in df.columns and 'original_title' in df.columns:
        df = df.rename(columns={'original_title': 'title'})
    if 'title' not in df.columns:
        raise ValueError("Movies must provide either 'title' or 'original_title' column")

    df['title'] = df['title'].astype(str)
    df['overview'] = df['overview'].fillna('')

    # Release date and year
    if 'release_date' in df.columns:
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['release_year'] = df['release_date'].dt.year
    else:
        df['release_year'] = np.nan

    # Ensure JSON-like text columns exist as strings (parsing is deferred to model code)
    for col in ['genres', 'keywords', 'cast', 'crew', 'production_countries']:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Drop duplicate ids
    df = df.drop_duplicates(subset=['id']).reset_index(drop=True)

    # Keep the common columns used by your pages
    keep = [
        'id','title','overview','vote_average','vote_count','popularity','runtime','budget','revenue',
        'release_date','release_year','genres','keywords','cast','crew','production_countries'
    ]
    df = df[[c for c in keep if c in df.columns]]

    return df