# model_comparison_and_classification.py
# Streamlit pages: Model Comparison (recommenders) + Classification Lab
# Drop-in: integrate with your existing app where df (movies) is prepared.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from ast import literal_eval
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer

# Optional implicit ALS. If not available, a simple MF fallback is used.
try:
    import implicit  # pip install implicit

    IMPLICIT_AVAILABLE = True
except Exception:
    IMPLICIT_AVAILABLE = False


# ---------------------------
# Utilities used by recommenders
# ---------------------------

def weighted_rating_row(x, m, C):
    v = x['vote_count']
    R = x['vote_average']
    return (v / (v + m) * R) + (m / (m + v) * C)


def ensure_release_year(df):
    if 'release_year' not in df.columns:
        if 'release_date' in df.columns:
            df = df.copy()
            df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
        else:
            df = df.copy()
            df['release_year'] = np.nan
    return df


def parse_json_list_column(x, key='name'):
    try:
        if isinstance(x, str):
            arr = literal_eval(x)
            if isinstance(arr, list):
                return [d.get(key, '') for d in arr if isinstance(d, dict)]
        elif isinstance(x, list):
            return [str(v) for v in x]
    except:
        return []
    return []


def clean_tokens(tokens):
    if not isinstance(tokens, list):
        return []
    return [t.lower().replace(' ', '') for t in tokens if isinstance(t, str) and t.strip()]


def build_content_soup(df):
    df = df.copy()
    # Director from crew
    if 'crew' in df.columns:
        df['director'] = df['crew'].apply(lambda c: next(
            (d.get('name') for d in (literal_eval(c) if isinstance(c, str) else c or []) if
             isinstance(d, dict) and d.get('job') == 'Director'), ''), 1)
    else:
        df['director'] = ''
    # lists
    for col in ['cast', 'genres', 'keywords']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: clean_tokens(parse_json_list_column(x)))
        else:
            df[col] = [[] for _ in range(len(df))]
    df['director'] = df['director'].fillna('').astype(str).apply(lambda s: s.lower().replace(' ', ''))
    df['soup'] = (df['keywords'].apply(lambda xs: ' '.join(xs)) + ' ' +
                  df['cast'].apply(lambda xs: ' '.join(xs)) + ' ' +
                  df['director'] + ' ' +
                  df['genres'].apply(lambda xs: ' '.join(xs))).str.strip()
    return df


def tfidf_similarity_from_overview(df):
    texts = df['overview'].fillna('No overview available').astype(str)
    tfidf = TfidfVectorizer(stop_words='english', max_features=15000, max_df=0.95, min_df=1, ngram_range=(1, 2))
    mat = tfidf.fit_transform(texts)
    return cosine_similarity(mat, mat)


def count_similarity_from_soup(df_soup):
    texts = df_soup['soup'].fillna('no meta').astype(str)
    count = CountVectorizer(stop_words='english', max_features=15000, max_df=0.95, min_df=1)
    mat = count.fit_transform(texts)
    return cosine_similarity(mat, mat)


def topn_from_scores(scores, exclude_idx=None, topn=10):
    # scores: numpy array shape (n_items,)
    indices = np.argsort(-scores)
    if exclude_idx is not None:
        indices = indices[indices != exclude_idx]
    return indices[:topn], scores[indices[:topn]]


# ---------------------------
# Collaborative filtering (ALS via implicit or simple MF fallback)
# ---------------------------

def build_interaction_matrix(interactions_df, user_col='userId', item_col='id', rating_col=None, use_binary=True):
    # Map ids to indices
    uids = interactions_df[user_col].astype('category').cat.codes.values
    iids = interactions_df[item_col].astype('category').cat.codes.values
    users = interactions_df[user_col].astype('category')
    items = interactions_df[item_col].astype('category')
    if rating_col and rating_col in interactions_df.columns and not use_binary:
        vals = interactions_df[rating_col].astype(float).values
    else:
        vals = np.ones(len(interactions_df), dtype=float)
    mat = coo_matrix((vals, (uids, iids)))
    return users, items, mat.tocsr()


def train_implicit_als(item_user_mat, factors=64, reg=0.01, iterations=15, alpha=40.0):
    # implicit expects item-user matrix
    model = implicit.als.AlternatingLeastSquares(
        factors=factors, regularization=reg, iterations=iterations
    )
    # Apply confidence weighting
    item_user_conf = (item_user_mat * alpha).astype('double')
    model.fit(item_user_conf)
    return model


def cf_recommend_items_for_user_implicit(model, user_index, user_item_csr, N=10):
    rec_items, rec_scores = model.recommend(user_index, user_item_csr, N=N, filter_already_liked_items=True)
    return rec_items, rec_scores


# ---------------------------
# Evaluation Metrics for recommenders
# ---------------------------

def precision_at_k(recommended, ground_truth, k=10):
    if len(recommended) == 0:
        return 0.0
    rec_k = recommended[:k]
    hits = len(set(rec_k) & set(ground_truth))
    return hits / k


def recall_at_k(recommended, ground_truth, k=10):
    if len(ground_truth) == 0:
        return 0.0
    rec_k = recommended[:k]
    hits = len(set(rec_k) & set(ground_truth))
    return hits / len(ground_truth)


# ---------------------------
# Model Comparison Page
# ---------------------------

def page_model_comparison(df_movies):
    st.header("🧪 Model Comparison")
    st.markdown("Compare Demographic, Content TF‑IDF, Content Metadata")
    df_movies = df_movies.copy()
    df_movies['vote_average'] = pd.to_numeric(df_movies.get('vote_average', np.nan), errors='coerce').fillna(0.0)
    df_movies['vote_count'] = pd.to_numeric(df_movies.get('vote_count', np.nan), errors='coerce').fillna(0).astype(int)
    df_movies = ensure_release_year(df_movies)

    user_col_opt = 'userId'
    item_col_opt = 'id'

    interactions_df = None
    items_cat = None

    st.subheader("⚙️ Comparison Settings")
    col1, col2 = st.columns(2)
    with col1:
        topN = st.slider("Top-N recommendations", min_value=5, max_value=30, value=10, step=5)
    with col2:
        quantile_q = st.slider("Demographic vote-count quantile", 0.1, 0.95, 0.6, 0.05)

    # Select context: either user-centric (for CF/hybrid) or item-centric (for content)
    mode = st.radio("Recommendation mode", ["TfidfVectorizer" , "CountVectorizer"])

    selected_title = st.selectbox("Anchor movie", df_movies['title'].dropna().unique().tolist()[:5000])

    run_btn = st.button("Run Comparison")
    if not run_btn:
        return

    # Precompute demographic scores
    C = df_movies['vote_average'].mean()
    m = df_movies['vote_count'].quantile(quantile_q)
    dem_df = df_movies[df_movies['vote_count'] >= m].copy()
    if len(dem_df) == 0:
        dem_df = df_movies.copy()
    dem_df['score'] = dem_df.apply(lambda x: weighted_rating_row(x, m, C), axis=1)
    dem_df = dem_df.sort_values('score', ascending=False)

    # Precompute content similarities
    # TF-IDF from overview
    if 'overview' in df_movies.columns:
        st.write("Computing TF-IDF similarities...")
        tfidf_sim = tfidf_similarity_from_overview(df_movies)
    else:
        tfidf_sim = None
        st.warning("No 'overview' column found. TF-IDF content model unavailable.")

    # Metadata soup model
    st.write("Computing metadata similarities...")
    soup_df = build_content_soup(df_movies)
    meta_sim = count_similarity_from_soup(soup_df) if soup_df['soup'].str.len().sum() > 0 else None

    # Prepare mapping from title->index and id mapping for CF
    title_to_idx = pd.Series(df_movies.index, index=df_movies['title']).drop_duplicates()
    id_to_idx = None
    if interactions_df is not None and items_cat is not None:
        # Map movie IDs in df_movies to CF item indices if possible
        if 'id' in df_movies.columns:
            # items_cat.categories aligns with encoded items mapping used in user_item_csr
            item_ids_order = list(items_cat.cat.categories)
            id_to_idx = {mid: i for i, mid in enumerate(item_ids_order)}

    # 2) Content TF-IDF (if item mode and tfidf is available)
    if mode == "TfidfVectorizer" and tfidf_sim is not None and selected_title in title_to_idx:
        idx = title_to_idx[selected_title]
        sims = tfidf_sim[idx]
        rec_idx, rec_scores = topn_from_scores(sims, exclude_idx=idx, topn=topN)
        dfres = pd.DataFrame({
            'title': df_movies.iloc[rec_idx]['title'].values,
            'score': rec_scores
        })
        if not dfres.empty:
            st.dataframe(dfres.head(topN), use_container_width=True)
    else:
        dfres = pd.DataFrame(columns=['title', 'score'])
        if not dfres.empty:
            st.dataframe(dfres.head(topN), use_container_width=True)

    # 3) Content Metadata
    if mode == "CountVectorizer" and meta_sim is not None and selected_title in title_to_idx:
        idx = title_to_idx[selected_title]
        sims = meta_sim[idx]
        rec_idx, rec_scores = topn_from_scores(sims, exclude_idx=idx, topn=topN)
        dfres = pd.DataFrame({
            'title': df_movies.iloc[rec_idx]['title'].values,
            'score': rec_scores
        })
        if not dfres.empty:
            st.dataframe(dfres.head(topN), use_container_width=True)
    else:
        dfres = pd.DataFrame(columns=['title', 'score'])
        if not dfres.empty:
            st.dataframe(dfres.head(topN), use_container_width=True)
