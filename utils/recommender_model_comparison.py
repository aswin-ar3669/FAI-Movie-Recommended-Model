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
    mode = st.radio("Recommendation mode", ["Item-to-Item (content)"])
    selected_title = None

    if mode == "Item-to-Item (content)":
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

    # Run per-model recommendations
    results = {}

    # 2) Content TF-IDF (if item mode and tfidf is available)
    if mode == "Item-to-Item (content)" and tfidf_sim is not None and selected_title in title_to_idx:
        idx = title_to_idx[selected_title]
        sims = tfidf_sim[idx]
        rec_idx, rec_scores = topn_from_scores(sims, exclude_idx=idx, topn=topN)
        results['Content TF-IDF TfidfVectorizer'] = pd.DataFrame({
            'title': df_movies.iloc[rec_idx]['title'].values,
            'score': rec_scores
        })
    else:
        results['Content TF-IDF TfidfVectorizer'] = pd.DataFrame(columns=['title', 'score'])

    # 3) Content Metadata
    if mode == "Item-to-Item (content)" and meta_sim is not None and selected_title in title_to_idx:
        idx = title_to_idx[selected_title]
        sims = meta_sim[idx]
        rec_idx, rec_scores = topn_from_scores(sims, exclude_idx=idx, topn=topN)
        results['Content Metadata CountVectorizer'] = pd.DataFrame({
            'title': df_movies.iloc[rec_idx]['title'].values,
            'score': rec_scores
        })
    else:
        results['Content Metadata CountVectorizer'] = pd.DataFrame(columns=['title', 'score'])


    # Display results
    st.subheader("📋 Top-N Results by Model")
    tabs = st.tabs(list(results.keys()))
    for i, (model_name, dfres) in enumerate(results.items()):
        with tabs[i]:
            if dfres is not None and not dfres.empty:
                st.dataframe(dfres.head(topN), use_container_width=True)
            else:
                st.info("No results available for this model and context.")


# ---------------------------
# Classification Lab Page
# ---------------------------

def page_classification_lab(df_movies):
    st.header("🧪 Classification Lab")
    st.markdown("Train quick text/tabular classifiers (e.g., predict hit vs flop, rating band, or genre multi-label).")

    # Target selection helpers
    # 1) If user provides labeled CSV
    label_col = st.text_input("Target column name (binary/multi-class)", value="hit_column")
    text_source = st.selectbox("Text feature", options=['overview', 'soup', 'title'], index=0)

    df = df_movies.copy()
    # Optionally build soup
    if text_source == 'soup':
        df = build_content_soup(df)

    # Simple demo target
    df['vote_average'] = pd.to_numeric(df.get('vote_average', np.nan), errors='coerce').fillna(0.0)
    labeled_df = df.dropna(subset=[text_source]).copy()
    labeled_df[label_col] = (labeled_df['vote_average'] >= 6.5).astype(int)

    st.write(f"Training rows available: {len(labeled_df)}")

    # Model selection
    model_choice = st.selectbox(
        "Classifier",
        [
            "Logistic Regression (TF‑IDF)",
            "Linear SVM (TF‑IDF)",
            "Multinomial Naive Bayes (TF‑IDF)",
            "Random Forest (TF‑IDF)"
        ],
        index=1
    )

    if st.button("Train & Evaluate"):
        X = labeled_df[text_source].fillna('').astype(str)
        y = labeled_df[label_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,
                                                            stratify=y if y.nunique() > 1 else None)

        # Build pipeline
        tfidf = TfidfVectorizer(stop_words='english', max_features=20000, ngram_range=(1, 2), min_df=1, max_df=0.95)
        if model_choice.startswith("Logistic"):
            clf = LogisticRegression(max_iter=200, n_jobs=None)
        elif model_choice.startswith("Linear SVM"):
            clf = LinearSVC()
        elif model_choice.startswith("Multinomial"):
            clf = MultinomialNB()
        else:
            clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)

        pipe = Pipeline([
            ("tfidf", tfidf),
            ("clf", clf)
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted" if y.nunique() > 2 else "binary")
        prec = precision_score(y_test, y_pred, average="weighted" if y.nunique() > 2 else "binary", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted" if y.nunique() > 2 else "binary", zero_division=0)

        st.subheader("📊 Evaluation")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{acc:.3f}")
        with col2:
            st.metric("F1 Score", f"{f1:.3f}")
        with col3:
            st.metric("Precision", f"{prec:.3f}")
        with col4:
            st.metric("Recall", f"{rec:.3f}")

        scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2")
        mse_scores = -scores
        col11, col12, = st.columns(2)
        with col11:
            st.metric("Mean",f"{mse_scores.mean():.3f}")
        with col12:
            st.metric("Standard Deviation",f"{mse_scores.std():.3f}")

        st.subheader("🧾 Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.subheader("🧩 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=[f"True {c}" for c in np.unique(y_test)],
                             columns=[f"Pred {c}" for c in np.unique(y_test)])
        st.dataframe(cm_df, use_container_width=True)

        # Summary statistics
        st.info(
            "Tip: For multi-label tasks (e.g., genres), convert genres to multi-hot and wrap a OneVsRestClassifier over the chosen base model.")
