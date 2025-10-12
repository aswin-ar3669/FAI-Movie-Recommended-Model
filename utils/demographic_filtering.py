import streamlit as st
import pandas as pd
import plotly.express as px

# Import your weighted rating function if it's in another file
# from utils.weighted_rating import weighted_rating

# Include all previous preprocessing functions (weighted_rating, get_director, etc.)
def weighted_rating(x, m, C):
    """Calculate IMDB weighted rating"""
    try:
        v = x['vote_count']
        R = x['vote_average']
        return (v/(v+m) * R) + (m/(m+v) * C)
    except:
        return 0

def demographic_filtering(df):
    """Demographic filtering with interactive controls"""
    st.header("📊 Demographic Filtering")

    # --- Interactive Controls ---
    col1, col2, col3 = st.columns(3)

    with col1:
        quantile_value = st.slider("📊 Vote Count Quantile", 0.1, 0.95, 0.6, 0.05)
        vote_threshold = df['vote_count'].quantile(quantile_value)
        st.info(f"📈 {quantile_value*100:.0f}th percentile = {int(vote_threshold):,} votes")

    with col2:
        min_rating = st.slider("⭐ Minimum Rating", 0.0, 10.0, 0.0, 0.1)

    with col3:
        num_movies = st.selectbox("🎬 Movies to Display", [10, 15, 20, 25, 30], index=1)

    # --- Calculate metrics ---
    C = df['vote_average'].mean()
    m = df['vote_count'].quantile(quantile_value)

    filtered_df = df.copy()
    if min_rating > 0:
        filtered_df = filtered_df[filtered_df['vote_average'] >= min_rating]

    # --- Display Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Average Rating", f"{C:.2f}/10")
    with col2:
        st.metric("📈 Vote Threshold", f"{int(m):,}")
    with col3:
        qualified_movies = len(filtered_df[filtered_df['vote_count'] >= m])
        st.metric("🎯 Qualified Movies", f"{qualified_movies}")
    with col4:
        if len(filtered_df) > 0:
            percentage = (qualified_movies / len(filtered_df)) * 100
            st.metric("📊 Qualification Rate", f"{percentage:.1f}%")

    # --- Filter and Display Results ---
    q_movies = filtered_df[filtered_df['vote_count'] >= m].copy()
    if len(q_movies) == 0:
        st.warning("⚠️ No movies meet the criteria. Try lowering the thresholds.")
        return

    q_movies['score'] = q_movies.apply(lambda x: weighted_rating(x, m, C), axis=1)
    q_movies = q_movies.sort_values('score', ascending=False)

    # --- Top Movies Table ---
    st.subheader(f"🏆 Top {num_movies} Movies (Weighted Score)")

    display_data = []
    for i, (_, movie) in enumerate(q_movies.head(num_movies).iterrows(), 1):
        display_data.append({
            'Rank': i,
            '🎬 Title': movie['title'],
            '⭐ Score': f"{movie['score']:.3f}",
            '📊 Rating': f"{movie['vote_average']:.1f}/10",
            '🗳️ Votes': f"{int(movie['vote_count']):,}",
            '📅 Year': int(movie.get('release_year', 0)) if pd.notna(movie.get('release_year')) else 'N/A'
        })

    st.dataframe(pd.DataFrame(display_data), use_container_width=True, hide_index=True)
    
    # --- Popularity Chart ---
    if 'popularity' in df.columns:
        st.subheader("📈 Most Popular Movies")
        pop_movies = df.nlargest(num_movies, 'popularity')

        fig = px.bar(
            pop_movies, 
            x='popularity', 
            y='title',
            orientation='h',
            title=f'Top {num_movies} Most Popular Movies',
            labels={'popularity': 'Popularity Score', 'title': 'Movie Title'},
            color='popularity',
            color_continuous_scale='viridis',
            text='popularity'
        )
        fig.update_layout(
            yaxis={'categoryorder':'total ascending'},
            height=500,
            showlegend=False
        )
        fig.update_traces(texttemplate='%{text:.1f}', textposition='inside')
        config = {"responsive": True, "displayModeBar": True}
        st.plotly_chart(fig, use_container_width=True, config=config)
