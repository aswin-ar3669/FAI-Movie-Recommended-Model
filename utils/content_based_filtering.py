import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval
import streamlit as st
import plotly.express as px
import warnings

def _count_valid_texts(series, min_len=3):
    texts = series.fillna('').astype(str).str.strip()
    texts = texts[texts.str.len() >= min_len]
    return texts

def content_based_filtering(df):
    """Content-based filtering recommendations"""
    st.header("🎯 Content-Based Filtering")
    st.markdown("""
    This system recommends movies based on their content similarity using:
    - **📄 Plot-based**: TF-IDF vectorization of movie overviews
    - **🎭 Metadata-based**: Cast, director, genres, and keywords using Count Vectorization
    """)

    # Check if we have required data
    if 'title' not in df.columns:
        st.error("❌ Dataset must contain 'title' column for content-based filtering.")
        return

    # Preprocess data for content-based filtering
    with st.spinner("🔄 Processing movie data for recommendations..."):
        try:
            df_processed = df.copy()

            # Handle missing overviews
            df_processed['overview'] = df_processed['overview'].fillna('')
            overview_texts = df_processed['overview'].fillna('').astype(str).str.strip()

            # Check for metadata features
            metadata_features = ['cast', 'crew', 'keywords', 'genres']
            has_metadata = any(col in df_processed.columns for col in metadata_features)

            if has_metadata:
                st.success("✅ Metadata features found - both recommendation types available!")

                # Process metadata features
                for feature in metadata_features:
                    if feature in df_processed.columns:
                        if feature == 'crew':
                            df_processed['director'] = df_processed[feature].apply(get_director)
                            df_processed['director'] = df_processed['director'].apply(clean_data)
                        else:
                            df_processed[feature] = df_processed[feature].apply(get_list)
                            df_processed[feature] = df_processed[feature].apply(clean_data)

                # Create metadata soup
                df_processed['soup'] = df_processed.apply(create_soup, axis=1)
            else:
                st.info("ℹ️ Only plot-based recommendations available (metadata columns not found)")

            # Calculate similarity matrices

            # 1. TF-IDF for overview (plot-based)
            tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 1), min_df=1, max_df=1.0)
            tfidf_matrix = tfidf.fit_transform(overview_texts)
            cosine_sim_plot = cosine_similarity(tfidf_matrix, tfidf_matrix)

            # 2. Count vectorizer for metadata (if available)
            cosine_sim_metadata = None
            if has_metadata and 'soup' in df_processed.columns:
                soup_texts = _count_valid_texts(df_processed['soup'])
                if len(soup_texts) >= 5:
                    count = CountVectorizer(stop_words='english', ngram_range=(1, 1), min_df=1, max_df=1.0)
                    count_matrix = count.fit_transform(df_processed['soup'].fillna('').astype(str))
                    cosine_sim_metadata = cosine_similarity(count_matrix, count_matrix)
                else:
                    st.info(
                        "ℹ️ Not enough non-empty metadata rows to build metadata-based similarity; using plot-based only.")
                    cosine_sim_metadata = None

        except Exception as e:
            st.error(f"❌ Error processing data: {str(e)}")
            return

    # Movie selection interface
    st.subheader("🎬 Select a Movie for Recommendations")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Search functionality
        search_term = st.text_input("🔍 Search for a movie:", placeholder="Type movie title...")

        if search_term:
            filtered_titles = df_processed[df_processed['title'].str.contains(
                search_term, case=False, na=False)]['title'].tolist()
            if filtered_titles:
                selected_movie = st.selectbox("📋 Select from search results:", filtered_titles)
            else:
                st.warning("⚠️ No movies found. Try a different search term.")
                # Show top movies as fallback
                top_movies = df_processed.nlargest(100, 'vote_average')['title'].tolist()
                selected_movie = st.selectbox("📋 Choose from top rated movies:", top_movies)
        else:
            # Show top movies for selection
            top_movies = df_processed.nlargest(100, 'vote_average')['title'].tolist()
            selected_movie = st.selectbox("📋 Choose from top rated movies:", top_movies)

    with col2:
        if has_metadata and cosine_sim_metadata is not None:
            recommendation_type = st.radio(
                "🎯 Recommendation Type:",
                ["📄 Plot-based", "🎭 Metadata-based"],
            )
        else:
            recommendation_type = "📄 Plot-based"
            st.info("ℹ️ Only plot-based available")

        num_recommendations = st.slider("📊 Number of recommendations:", 5, 20, 10)

    # Display selected movie info
    if selected_movie:
        try:
            movie_info = df_processed[df_processed['title'] == selected_movie].iloc[0]

            with st.expander("📋 Selected Movie Details", expanded=True):
                col1, col2 = st.columns([1, 2])
                with col1:
                    if 'vote_average' in movie_info:
                        st.write(f"⭐ **Rating:** {movie_info['vote_average']:.1f}/10")
                    if 'vote_count' in movie_info:
                        st.write(f"🗳️ **Votes:** {int(movie_info['vote_count']):,}")
                    if 'popularity' in movie_info:
                        st.write(f"📈 **Popularity:** {movie_info['popularity']:.1f}")
                    if 'release_date' in movie_info:
                        st.write(f"📅 **Release:** {str(movie_info['release_date'])[:4]}")
                with col2:
                    overview = movie_info.get('overview', 'No overview available')
                    if len(overview) > 300:
                        overview = overview[:300] + "..."
                    st.write(f"📄 **Overview:** {overview}")
        except Exception as e:
            st.warning(f"⚠️ Could not display movie details: {str(e)}")

    # Generate recommendations
    if st.button("🎯 Get Recommendations", type="primary", use_container_width=True):
        with st.spinner("🔄 Finding similar movies..."):
            try:
                if recommendation_type == "📄 Plot-based":
                    cosine_sim = cosine_sim_plot
                else:
                    cosine_sim = cosine_sim_metadata

                recommendations = get_recommendations(
                    selected_movie,
                    cosine_sim,
                    df_processed,
                    num_recommendations
                )

                if recommendations and len(recommendations) > 0:
                    st.subheader(f"🎯 Movies Similar to '{selected_movie}' ({recommendation_type})")

                    # Create tabs for different views
                    tab1, tab2 = st.tabs(["📋 Detailed List", "📊 Similarity Chart"])

                    with tab1:
                        # Display recommendations in cards
                        for i, (title, similarity) in enumerate(recommendations, 1):
                            try:
                                movie_data = df_processed[df_processed['title'] == title].iloc[0]

                                # Color code similarity score
                                if similarity > 0.5:
                                    similarity_emoji = "🟢"
                                    similarity_text = "High"
                                elif similarity > 0.3:
                                    similarity_emoji = "🟡"
                                    similarity_text = "Medium"
                                else:
                                    similarity_emoji = "🟠"
                                    similarity_text = "Low"

                                with st.expander(f"{i}. **{title}** {similarity_emoji} ({similarity_text} Similarity: {similarity:.3f})"):
                                    col1, col2 = st.columns([1, 2])
                                    with col1:
                                        if 'vote_average' in movie_data:
                                            st.write(f"⭐ **Rating:** {movie_data['vote_average']:.1f}/10")
                                        if 'vote_count' in movie_data:
                                            st.write(f"🗳️ **Votes:** {int(movie_data['vote_count']):,}")
                                        if 'popularity' in movie_data:
                                            st.write(f"📈 **Popularity:** {movie_data['popularity']:.1f}")
                                        if 'runtime' in movie_data:
                                            st.write(f"⏱️ **Runtime:** {int(movie_data['runtime'])} min")
                                    with col2:
                                        overview_text = movie_data.get('overview', 'No overview available')
                                        if len(overview_text) > 250:
                                            overview_text = overview_text[:250] + "..."
                                        st.write(f"📄 **Overview:** {overview_text}")
                            except Exception as e:
                                st.write(f"{i}. **{title}** (Similarity: {similarity:.3f})")

                    with tab2:
                        # Create similarity score chart
                        titles = [rec[0] for rec in recommendations]
                        similarities = [rec[1] for rec in recommendations]

                        fig = px.bar(
                            x=similarities,
                            y=titles,
                            orientation='h',
                            title=f'Similarity Scores for "{selected_movie}" ({recommendation_type})',
                            labels={'x': 'Similarity Score', 'y': 'Movie Title'},
                            color=similarities,
                            color_continuous_scale='viridis',
                            text=similarities
                        )
                        fig.update_layout(
                            yaxis={'categoryorder': 'total ascending'},
                            height=600,
                            showlegend=False
                        )
                        fig.update_traces(texttemplate='%{text:.3f}', textposition='inside')
                        config = {"responsive": True, "displayModeBar": True}
                        st.plotly_chart(fig, use_container_width=True,config=config)

                else:
                    st.error("❌ No recommendations found. Try selecting a different movie.")

            except Exception as e:
                st.error(f"❌ Error generating recommendations: {str(e)}")



def get_recommendations(title, cosine_sim, df, num_recommendations=10):
    """Get movie recommendations based on similarity"""
    try:
        # Check if title exists in dataframe
        if title not in df['title'].values:
            return None

        # Create reverse mapping of titles to indices
        indices = pd.Series(df.index, index=df['title']).drop_duplicates()

        # Get movie index
        idx = indices[title]

        # Get similarity scores
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:num_recommendations+1]  # Exclude self

        # Get movie indices and titles
        movie_indices = [i[0] for i in sim_scores]
        similarities = [i[1] for i in sim_scores]

        # Ensure indices are valid
        valid_indices = [idx for idx in movie_indices if idx < len(df)]
        valid_similarities = similarities[:len(valid_indices)]

        titles = df.iloc[valid_indices]['title'].values

        return list(zip(titles, valid_similarities))

    except Exception as e:
        st.error(f"❌ Error in recommendation function: {str(e)}")
        return None

def get_director(x):
    """Extract director from crew data"""
    try:
        if pd.isna(x) or x == '':
            return ''
        crew_list = literal_eval(x) if isinstance(x, str) else x
        if isinstance(crew_list, list):
            for i in crew_list:
                if isinstance(i, dict) and i.get('job') == 'Director':
                    return i.get('name', '')
    except:
        pass
    return ''

def get_list(x):
    """Extract top 3 elements from list"""
    try:
        if pd.isna(x) or x == '':
            return []
        if isinstance(x, str):
            x = literal_eval(x)
        if isinstance(x, list):
            names = [i.get('name', '') for i in x if isinstance(i, dict) and i.get('name')]
            return names[:3] if len(names) > 3 else names
    except:
        pass
    return []

def clean_data(x):
    """Clean and normalize text data"""
    try:
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "")) for i in x if i and isinstance(i, str)]
        elif isinstance(x, str) and x:
            return str.lower(x.replace(" ", ""))
    except:
        pass
    return ''

def create_soup(x):
    """Create metadata soup for content-based filtering"""
    try:
        soup_parts = []

        if 'keywords' in x and isinstance(x['keywords'], list):
            soup_parts.extend(x['keywords'])
        if 'cast' in x and isinstance(x['cast'], list):
            soup_parts.extend(x['cast'])
        if 'director' in x and isinstance(x['director'], str) and x['director']:
            soup_parts.append(x['director'])
        if 'genres' in x and isinstance(x['genres'], list):
            soup_parts.extend(x['genres'])

        return ' '.join(soup_parts).strip()
    except:
        return ''