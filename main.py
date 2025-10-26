import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import warnings

from utils.content_based_filtering import content_based_filtering
from utils.demographic_filtering import demographic_filtering
from utils.recommender_model_comparison import page_model_comparison
from utils.movie_analysis_dashboard import movie_analysis_dashboard
from utils.featureengineering import load_tmdb_5000

warnings.filterwarnings('ignore')

# Optional word cloud import (gracefully handle if not installed)
try:
    from wordcloud import WordCloud

    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# Configure Streamlit page
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎬 Movie Recommendation System")
# Add description
st.markdown("""
Recommendation Systems are a type of **information 
filtering systems** as they improve the quality of search results and provides items that are more relevant to the 
search item or are related to the search history of the user.

**Types of Recommender Systems:**
1. **Content Based Filtering** - Recommendations based on item similarity
2. **Demographic Filtering** - Generalized recommendations based on popularity
3. **Collaborative Filtering** - Recommendations based on user behavior patterns
""")

# File uploader for datasets
st.sidebar.header("📁 Data Upload")
st.sidebar.markdown("**Upload TMDB Movie Dataset Files:**")
uploaded_movies = st.sidebar.file_uploader("Upload Movies CSV (tmdb_5000_movies.csv)", type=['csv'], key='movies')
uploaded_credits = st.sidebar.file_uploader("Upload Credits CSV (tmdb_5000_credits.csv)", type=['csv'], key='credits')


# Data loading function
@st.cache_data
def load_data(movies_file=None, credits_file=None):
    """Load and preprocess the movie datasets"""
    try:
        if movies_file is not None:
            # Load movies data
            df_movies = pd.read_csv(movies_file, encoding="utf-8")

            # Display column information in sidebar
            with st.sidebar.expander("📋 Movies CSV Info"):
                st.write("**Columns found:**")
                st.write(list(df_movies.columns))
                st.write(f"**Shape:** {df_movies.shape}")

            # Check for common column variations and rename them
            column_mapping = {
                'original_title': 'title',
                'movie_title': 'title',
                'film_title': 'title',
                'name': 'title'
            }

            for old_col, new_col in column_mapping.items():
                if old_col in df_movies.columns and new_col not in df_movies.columns:
                    df_movies = df_movies.rename(columns={old_col: new_col})

            # If credits file is provided, merge them
            if credits_file is not None:
                df_credits = pd.read_csv(credits_file, encoding="utf-8")

                with st.sidebar.expander("📋 Credits CSV Info"):
                    st.write("**Columns found:**")
                    st.write(list(df_credits.columns))
                    st.write(f"**Shape:** {df_credits.shape}")

                # Handle different credit file structures
                if len(df_credits.columns) == 4:
                    df_credits.columns = ['movie_id', 'title', 'cast', 'crew']
                    df_credits = df_credits.rename(columns={'movie_id': 'id'})
                elif 'movie_id' in df_credits.columns or 'id' in df_credits.columns:
                    if 'movie_id' in df_credits.columns and 'id' not in df_credits.columns:
                        df_credits = df_credits.rename(columns={'movie_id': 'id'})
                else:
                    st.sidebar.warning("⚠️ Credits file structure unclear, using first column as ID")
                    if len(df_credits.columns) >= 4:
                        df_credits = df_credits.iloc[:, :4].copy()
                        df_credits.columns = ['id', 'title_credits', 'cast', 'crew']

                # Merge datasets
                if 'id' in df_movies.columns and 'id' in df_credits.columns:
                    df = df_movies.merge(df_credits, on='id', how='left', suffixes=('', '_credits'))
                    if 'title_credits' in df.columns:
                        df = df.drop('title_credits', axis=1)
                    st.sidebar.success("✅ Successfully merged movies and credits data!")
                else:
                    st.sidebar.warning("⚠️ Could not merge datasets - no matching ID columns")
                    df = df_movies
            else:
                df = df_movies

            # Final data validation and cleaning
            required_columns = ['title']
            for col in required_columns:
                if col not in df.columns:
                    st.error(f"❌ Missing required column: '{col}'. Available columns: {list(df.columns)}")
                    return None

            # Add default columns if missing
            if 'overview' not in df.columns:
                df['overview'] = 'No overview available'
            if 'vote_average' not in df.columns:
                df['vote_average'] = np.random.uniform(5.0, 8.5, len(df))
            if 'vote_count' not in df.columns:
                df['vote_count'] = np.random.randint(100, 5000, len(df))
            if 'popularity' not in df.columns:
                df['popularity'] = np.random.uniform(1, 50, len(df))

            # Handle release date parsing for year extraction
            if 'release_date' in df.columns:
                df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
                df['release_year'] = df['release_date'].dt.year
            else:
                # Create sample release years if not available
                df['release_year'] = np.random.choice(range(1960, 2024), len(df))

            # Handle production countries
            if 'production_countries' not in df.columns:
                sample_countries = ['United States', 'United Kingdom', 'France', 'Germany', 'Italy']
                df['production_countries'] = [f'[{{"name": "{np.random.choice(sample_countries)}"}}]'
                                              for _ in range(len(df))]

            # Clean the data
            df = df.dropna(subset=['title'])
            df = df.drop_duplicates(subset=['title'], keep='first')
            df = df.reset_index(drop=True)

            # Fill missing values
            df['overview'] = df['overview'].fillna('')

            # Convert data types
            numeric_cols = ['vote_average', 'vote_count', 'popularity']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            return df

    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None


# Main application
def main():
    # Sidebar for navigation
    st.sidebar.title("📍 Navigation")
    page = st.sidebar.selectbox("Choose a section:", [
        "Content-Based Filtering",
        "Model Comparison",
        "Demographic Filtering",
        "Movie Analysis Dashboard"
    ])

    # Load data
    movies_file = uploaded_movies
    credits_file = uploaded_credits

    df = load_data(movies_file, credits_file)
    if df is None:
        st.error("❌ Failed to load data. Please check your files and try again.")
        return

    # Display data status
    if movies_file:
        st.sidebar.success(f"✅ Loaded {len(df)} movies from uploaded files")
    else:
        st.sidebar.info(f"📊 Using sample data with {len(df)} movies")

    # Process data based on selected page
    if page == "Demographic Filtering":
        demographic_filtering(df)
    elif page == "Content-Based Filtering":
        content_based_filtering(df)
    elif page == "Movie Analysis Dashboard":
        movie_analysis_dashboard(df)
    elif page == "Model Comparison":
        movie_df = load_tmdb_5000(movies_file, credits_file)
        page_model_comparison(movie_df)
    else:
        content_based_filtering(df)


if __name__ == "__main__":
    main()
