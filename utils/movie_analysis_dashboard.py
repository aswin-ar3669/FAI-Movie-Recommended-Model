import warnings

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from ast import literal_eval

warnings.filterwarnings('ignore')

# Optional word cloud import (gracefully handle if not installed)
try:
    from wordcloud import WordCloud

    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False


def movie_analysis_dashboard(df):
    """Enhanced Movie analysis dashboard with requested visualizations"""
    st.header("📈 Movie Analysis Dashboard")
    st.markdown("Comprehensive analysis with temporal, geographical, and textual insights.")

    # Overview statistics
    st.subheader("📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🎬 Total Movies", f"{len(df):,}")
    with col2:
        if 'vote_average' in df.columns:
            st.metric("⭐ Avg Rating", f"{df['vote_average'].mean():.2f}/10")
    with col3:
        if 'release_year' in df.columns:
            year_range = f"{int(df['release_year'].min())}-{int(df['release_year'].max())}"
            st.metric("📅 Year Range", year_range)
    with col4:
        if 'budget' in df.columns:
            total_budget = df['budget'].sum()
            if total_budget > 1e9:
                st.metric("💰 Total Budget", f"${total_budget / 1e9:.1f}B")
            else:
                st.metric("💰 Total Budget", f"${total_budget / 1e6:.0f}M")

    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📅 Movie Analysis",
        "🌍 Geographic Analysis",
        "📄 Text Analysis",
        "📊 Distributions",
        "🔗 Correlations"
    ])

    with tab1:
        st.subheader("📅 Movie Analysis")

        # 1. Number of Movies Released Each Year
        if 'release_year' in df.columns:
            st.markdown("#### Number of Movies Released Each Year")

            # Group by year and count movies
            yearly_counts = df.groupby('release_year').size().reset_index(name='count')
            yearly_counts = yearly_counts.sort_values('release_year')

            # Create interactive line plot
            fig_yearly = px.line(
                yearly_counts,
                x='release_year',
                y='count',
                title='Number of Movies Released Each Year',
                labels={'release_year': 'Release Year', 'count': 'Number of Movies'},
                markers=True,
                line_shape='spline'
            )
            fig_yearly.update_layout(
                height=500,
                showlegend=False,
                xaxis_title="Release Year",
                yaxis_title="Number of Movies"
            )
            config = {"responsive": True, "displayModeBar": True}
            st.plotly_chart(fig_yearly, use_container_width=True, config=config)

            # Additional yearly statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                peak_year = yearly_counts.loc[yearly_counts['count'].idxmax(), 'release_year']
                peak_count = yearly_counts['count'].max()
                st.metric("📈 Peak Year", f"{int(peak_year)} ({peak_count} movies)")

            with col2:
                avg_per_year = yearly_counts['count'].mean()
                st.metric("📊 Average per Year", f"{avg_per_year:.1f} movies")

            with col3:
                recent_years = yearly_counts[yearly_counts['release_year'] >= 2010]['count'].mean()
                st.metric("🆕 Recent Average (2010+)", f"{recent_years:.1f} movies")

        # Decade analysis
        if 'release_year' in df.columns:
            st.markdown("#### Movies by Decade")
            df_decade = df.copy()
            df_decade['decade'] = (df_decade['release_year'] // 10) * 10
            decade_counts = df_decade.groupby('decade').size().reset_index(name='count')

            fig_decade = px.bar(
                decade_counts,
                x='decade',
                y='count',
                title='Movie Distribution by Decade',
                labels={'decade': 'Decade', 'count': 'Number of Movies'},
                color='count',
                color_continuous_scale='viridis'
            )
            fig_decade.update_layout(height=400)
            config = {"responsive": True, "displayModeBar": True}
            st.plotly_chart(fig_decade, use_container_width=True, config=config)

        # Helper to parse list-like columns with dicts having 'name'
        def _parse_list_col(x):
            try:
                if pd.isna(x) or x == '':
                    return []
                if isinstance(x, str):
                    x = literal_eval(x)
                if isinstance(x, list):
                    return [i.get('name', '') for i in x if isinstance(i, dict) and i.get('name')]
            except:
                return []
            return []

        # Movies count by genre
        if 'genres' in df.columns:
            st.markdown("#### Numbers of Movies by Genre")
            gdf = df.copy()
            gdf['genres_list'] = gdf['genres'].apply(_parse_list_col)
            exploded = gdf.explode('genres_list')
            genre_counts = exploded['genres_list'].dropna().value_counts().reset_index()
            genre_counts.columns = ['Genre', 'Count']
            if len(genre_counts) > 0:
                fig_genre_counts = px.bar(
                    genre_counts.head(20).sort_values('Count'),
                    x='Count', y='Genre', orientation='h',
                    title='Movie Count by Genre (Top 20)',
                    labels={'Count': 'Number of Movies', 'Genre': 'Genre'},
                    color='Count', color_continuous_scale='teal'
                )
                fig_genre_counts.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_genre_counts, use_container_width=True,
                                config={"responsive": True, "displayModeBar": True})

        # Highest rated movies by genre
        st.markdown("#### Highest Rated Movies by Genre")

        if {'genres', 'title', 'vote_average', 'vote_count'}.issubset(df.columns):
            # Parse genres
            def _parse_list_col(x):
                try:
                    if pd.isna(x) or x == '':
                        return []
                    if isinstance(x, str):
                        x = literal_eval(x)
                    if isinstance(x, list):
                        return [i.get('name', '') for i in x if isinstance(i, dict) and i.get('name')]
                except:
                    return []
                return []

            min_votes_genre = st.slider("Minimum votes per movie (genre ranking)", 0, int(df['vote_count'].max()), 100,
                                        step=50, key="min_votes_genre_sep")
            gdf = df.copy()
            gdf['genres_list'] = gdf['genres'].apply(_parse_list_col)
            ex = gdf.explode('genres_list').dropna(subset=['genres_list'])

            # Apply minimum votes and pick top per genre
            ex = ex[ex['vote_count'] >= min_votes_genre]
            ex = ex.sort_values(['genres_list', 'vote_average', 'vote_count'], ascending=[True, False, False])
            top1 = ex.groupby('genres_list').head(1)[['genres_list', 'title', 'vote_average', 'vote_count']]

            if len(top1) > 0:
                st.markdown("Top movie per genre (min votes applied)")
                fig_top1 = px.bar(
                    top1.sort_values('vote_average'),
                    x='vote_average', y='genres_list', orientation='h',
                    title=f'Highest Rated Movie by Genre (min {min_votes_genre} votes)',
                    labels={'vote_average': 'Average Rating', 'genres_list': 'Genre'},
                    color='vote_average', color_continuous_scale='viridis',
                    text='title'
                )
                fig_top1.update_traces(textposition='outside')
                fig_top1.update_layout(height=650, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_top1, use_container_width=True, config={"responsive": True, "displayModeBar": True})
            else:
                st.info("💡 No qualifying movies for genre ranking.")
        else:
            st.info("💡 Required columns not found for genre ranking.")

        # Optional: show top-K per genre separately
        if {'genres', 'title', 'vote_average', 'vote_count'}.issubset(df.columns):
            st.markdown("#### Top-K Highest Rated Movies per Genre (separate)")
            top_k = st.slider("Top K per genre", 1, 10, 5, key="topk_genre_sep")

            def _parse_list_col(x):
                try:
                    if pd.isna(x) or x == '':
                        return []
                    if isinstance(x, str):
                        x = literal_eval(x)
                    if isinstance(x, list):
                        return [i.get('name', '') for i in x if isinstance(i, dict) and i.get('name')]
                except:
                    return []
                return []

            gdfK = df.copy()
            gdfK['genres_list'] = gdfK['genres'].apply(_parse_list_col)
            exK = gdfK.explode('genres_list').dropna(subset=['genres_list'])

            # Reuse minimum votes from above if created, else define a default
            min_votes_val = st.session_state.get("min_votes_genre_sep", 100)
            exK = exK[exK['vote_count'] >= min_votes_val]

            # Get available genres after filtering
            genres_available = exK['genres_list'].dropna().unique().tolist()
            selected_genres = st.multiselect(
                "Select genres to display",
                options=sorted(genres_available),
                default=sorted(genres_available)[:6]
            )

            # Create tabs per selected genre to keep charts separate
            if selected_genres:
                genre_tabs = st.tabs(selected_genres)
                for tg, tab in zip(selected_genres, genre_tabs):
                    with tab:
                        sub = exK[exK['genres_list'] == tg].sort_values(['vote_average', 'vote_count'],
                                                                        ascending=[False, False]).head(top_k)
                        if len(sub) == 0:
                            st.info(f"No movies for {tg} with min votes {min_votes_val}.")
                        else:
                            fig_sub = px.bar(
                                sub.sort_values('vote_average'),
                                x='vote_average', y='title', orientation='h',
                                title=f"{tg}: Top {len(sub)} by Rating (min {min_votes_val} votes)",
                                labels={'vote_average': 'Average Rating', 'title': 'Movie'},
                                color='vote_average', color_continuous_scale='plasma'
                            )
                            fig_sub.update_layout(height=max(350, 40 * len(sub)),
                                                  yaxis={'categoryorder': 'total ascending'})
                            st.plotly_chart(fig_sub, use_container_width=True,
                                            config={"responsive": True, "displayModeBar": True})


        # Actors with most appearances
        if 'cast' in df.columns:
            st.markdown("#### Actors with Most Appearances")
            cdf = df.copy()
            cdf['cast_list'] = cdf['cast'].apply(_parse_list_col)
            cast_ex = cdf.explode('cast_list')
            top_n_cast = st.slider("Top N actors", 5, 50, 20, key="top_cast")
            cast_counts = cast_ex['cast_list'].dropna().value_counts().head(top_n_cast).reset_index()
            cast_counts.columns = ['Actor', 'Appearances']
            if len(cast_counts) > 0:
                fig_cast = px.bar(
                    cast_counts.sort_values('Appearances'),
                    x='Appearances', y='Actor', orientation='h',
                    title=f'Top {len(cast_counts)} Actors by Appearances',
                    labels={'Appearances': 'Movie Appearances', 'Actor': 'Actor'},
                    color='Appearances', color_continuous_scale='plasma'
                )
                fig_cast.update_layout(height=650, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_cast, use_container_width=True, config={"responsive": True, "displayModeBar": True})

        # Directors with most movies
        def _get_director_from_crew(x):
            try:
                if pd.isna(x) or x == '':
                    return None
                crew = literal_eval(x) if isinstance(x, str) else x
                if isinstance(crew, list):
                    for d in crew:
                        if isinstance(d, dict) and d.get('job') == 'Director' and d.get('name'):
                            return d['name']
            except:
                return None
            return None

        if 'crew' in df.columns:
            st.markdown("#### Directors with Most Movies")
            ddf = df.copy()
            ddf['director'] = ddf['crew'].apply(_get_director_from_crew)
            dir_counts = ddf['director'].dropna().value_counts()
            top_n_dir = st.slider("Top N directors", 5, 50, 20, key="top_directors")
            dir_df = dir_counts.head(top_n_dir).reset_index()
            dir_df.columns = ['Director', 'Movies']
            if len(dir_df) > 0:
                fig_dir = px.bar(
                    dir_df.sort_values('Movies'),
                    x='Movies', y='Director', orientation='h',
                    title=f'Top {len(dir_df)} Directors by Movies',
                    labels={'Movies': 'Movies Directed', 'Director': 'Director'},
                    color='Movies', color_continuous_scale='sunset'
                )
                fig_dir.update_layout(height=650, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_dir, use_container_width=True, config={"responsive": True, "displayModeBar": True})


    with tab2:
        st.subheader("🌍 Geographic Analysis")

        # 2. Top Countries with Highest Number of Movies
        if 'production_countries' in df.columns:
            st.markdown("#### Top Countries with Highest Number of Movies")

            # Extract countries from the production_countries column
            df['countries'] = df['production_countries'].apply(extract_countries)

            # Create a list of all countries
            all_countries = []
            for countries_list in df['countries']:
                all_countries.extend(countries_list)

            # Count countries
            from collections import Counter
            country_counts = Counter(all_countries)

            # Convert to DataFrame for visualization
            country_df = pd.DataFrame(country_counts.most_common(15),
                                      columns=['Country', 'Number of Movies'])

            if len(country_df) > 0:
                # Create horizontal bar chart
                fig_countries = px.bar(
                    country_df,
                    x='Number of Movies',
                    y='Country',
                    orientation='h',
                    title='Top Countries with Highest Number of Movies',
                    labels={'Number of Movies': 'Number of Movies', 'Country': 'Country'},
                    color='Number of Movies',
                    color_continuous_scale='blues'
                )
                fig_countries.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    height=600
                )
                config = {"responsive": True, "displayModeBar": True}
                st.plotly_chart(fig_countries, use_container_width=True, config=config)

                # Country statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    top_country = country_df.iloc[0]['Country']
                    top_count = country_df.iloc[0]['Number of Movies']
                    st.metric("🥇 Top Country", f"{top_country} ({top_count})")

                with col2:
                    total_countries = len(country_df)
                    st.metric("🌍 Total Countries", total_countries)

                with col3:
                    avg_movies = country_df['Number of Movies'].mean()
                    st.metric("📊 Avg Movies/Country", f"{avg_movies:.1f}")
            else:
                st.info("💡 Country data not available in the current dataset")

        # Geographic distribution pie chart
        if 'production_countries' in df.columns and len(country_df) > 0:
            st.markdown("#### Geographic Distribution")
            fig_geo_pie = px.pie(
                country_df.head(10),
                values='Number of Movies',
                names='Country',
                title='Movie Production by Top 10 Countries'
            )
            fig_geo_pie.update_layout(height=500)
            config = {"responsive": True, "displayModeBar": True}
            st.plotly_chart(fig_geo_pie, use_container_width=True, config=config)

        # 3.6 Original Languages
        st.markdown("#### Original Languages")

        if 'original_language' in df.columns:
            # Optional mapping for common ISO 639-1 codes to readable names
            iso_map = {
                'en': 'English', 'hi': 'Hindi', 'ta': 'Tamil', 'te': 'Telugu', 'ml': 'Malayalam',
                'kn': 'Kannada', 'mr': 'Marathi', 'bn': 'Bengali', 'pa': 'Punjabi', 'gu': 'Gujarati',
                'ur': 'Urdu', 'fr': 'French', 'es': 'Spanish', 'de': 'German', 'it': 'Italian',
                'ja': 'Japanese', 'ko': 'Korean', 'zh': 'Chinese', 'ru': 'Russian', 'pt': 'Portuguese',
                'ar': 'Arabic', 'tr': 'Turkish', 'fa': 'Persian', 'nl': 'Dutch', 'sv': 'Swedish'
            }

            lang_series = df['original_language'].fillna('unknown').astype(str).str.lower().str.strip()
            # Count languages
            lang_counts = lang_series.value_counts(dropna=False).reset_index()
            lang_counts.columns = ['code', 'count']
            # Map to display name
            lang_counts['language'] = lang_counts['code'].map(iso_map).fillna(lang_counts['code'].str.upper())

            # Controls
            col_a, col_b = st.columns(2)
            with col_a:
                top_n_lang = st.slider("Top N languages", 5, 40, 15, key="top_n_languages")
            with col_b:
                group_others = st.checkbox("Group remaining as 'Others'", value=True, key="group_lang_others")

            if group_others and len(lang_counts) > top_n_lang:
                head = lang_counts.head(top_n_lang).copy()
                others_count = lang_counts['count'].iloc[top_n_lang:].sum()
                if others_count > 0:
                    head = pd.concat(
                        [head, pd.DataFrame([{'code': 'others', 'count': others_count, 'language': 'Others'}])],
                        ignore_index=True)
                plot_df = head
            else:
                plot_df = lang_counts.head(top_n_lang)

            # Horizontal bar chart
            fig_lang = px.bar(
                plot_df.sort_values('count'),
                x='count', y='language', orientation='h',
                title='Original Languages (Top)',
                labels={'count': 'Number of Movies', 'language': 'Language'},
                color='count', color_continuous_scale='Aggrnyl'
            )
            fig_lang.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_lang, use_container_width=True, config={"responsive": True, "displayModeBar": True})

            # Share as pie (optional)
            with st.expander("Show language share (pie)"):
                fig_lang_pie = px.pie(
                    plot_df, values='count', names='language',
                    title='Share of Original Languages'
                )
                fig_lang_pie.update_layout(height=500)
                st.plotly_chart(fig_lang_pie, use_container_width=True,
                                config={"responsive": True, "displayModeBar": True})

            # Quick metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🗣️ Unique Languages", f"{lang_counts.shape[0]}")
            with col2:
                top_lang = lang_counts.iloc[0]
                st.metric("🥇 Most Common",
                          f"{iso_map.get(top_lang['code'], top_lang['code'].upper())} ({int(top_lang['count'])})")
            with col3:
                pct_top5 = lang_counts['count'].head(5).sum() / lang_counts['count'].sum() * 100
                st.metric("Top 5 Coverage", f"{pct_top5:.1f}%")
        else:
            st.info("💡 original_language column not found in dataset")

    with tab3:
        st.subheader("📄 Text Analysis")

        # Most-used keywords
        if 'keywords' in df.columns:
            st.markdown("#### Most-used Keywords")
            kdf = df.copy()
            kdf['kw_list'] = kdf['keywords'].apply(_parse_list_col)
            kw_ex = kdf.explode('kw_list')
            top_n_kw = st.slider("Top N keywords", 10, 100, 30, step=5, key="top_keywords")
            kw_counts = kw_ex['kw_list'].dropna().value_counts().head(top_n_kw).reset_index()
            kw_counts.columns = ['Keyword', 'Count']
            if len(kw_counts) > 0:
                fig_kw = px.bar(
                    kw_counts.sort_values('Count'),
                    x='Count', y='Keyword', orientation='h',
                    title=f'Top {len(kw_counts)} Most-used Keywords',
                    labels={'Count': 'Occurrences', 'Keyword': 'Keyword'},
                    color='Count', color_continuous_scale='dense'
                )
                fig_kw.update_layout(height=700, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_kw, use_container_width=True, config={"responsive": True, "displayModeBar": True})


        # 3. Most Common Words (Word Clouds)
        if 'overview' in df.columns:
            st.markdown("#### Most Common Words in Movie Overviews")

            # Generate word frequency data
            word_freq_data = create_word_cloud_data(df['overview'], max_words=50)

            if word_freq_data:
                # Create word frequency bar chart (alternative to word cloud)
                words_df = pd.DataFrame(word_freq_data, columns=['Word', 'Frequency'])

                fig_words = px.bar(
                    words_df.head(20),
                    x='Frequency',
                    y='Word',
                    orientation='h',
                    title='Most Common Words in Movie Overviews (Top 20)',
                    labels={'Frequency': 'Frequency', 'Word': 'Word'},
                    color='Frequency',
                    color_continuous_scale='viridis'
                )
                fig_words.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    height=600
                )
                config = {"responsive": True, "displayModeBar": True}
                st.plotly_chart(fig_words, use_container_width=True, config=config)

                # Word cloud using matplotlib (if wordcloud is available)
                if WORDCLOUD_AVAILABLE:
                    st.markdown("#### Word Cloud Visualization")

                    # Create word frequency dictionary
                    word_dict = dict(word_freq_data)

                    # Generate word cloud
                    wordcloud = WordCloud(
                        width=800,
                        height=400,
                        background_color='white',
                        max_words=100,
                        colormap='viridis'
                    ).generate_from_frequencies(word_dict)

                    # Display word cloud
                    fig_wc, ax = plt.subplots(figsize=(12, 6))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    ax.set_title('Most Common Words in Movie Overviews', fontsize=16, pad=20)
                    st.pyplot(fig_wc)
                    plt.close()
                else:
                    st.info("💡 Install wordcloud package for enhanced word cloud visualization")

                # Text statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    most_common = words_df.iloc[0]
                    st.metric("🔤 Most Common Word", f"{most_common['Word']} ({most_common['Frequency']})")

                with col2:
                    unique_words = len(words_df)
                    st.metric("📚 Unique Words", unique_words)

                with col3:
                    total_words = sum(words_df['Frequency'])
                    st.metric("📊 Total Words", f"{total_words:,}")

        # Overview length analysis
        if 'overview' in df.columns:
            st.markdown("#### Overview Length Analysis")
            df['overview_length'] = df['overview'].str.len()

            fig_length = px.histogram(
                df,
                x='overview_length',
                nbins=30,
                title='Distribution of Overview Lengths',
                labels={'overview_length': 'Overview Length (characters)', 'count': 'Number of Movies'}
            )
            config = {"responsive": True, "displayModeBar": True}
            st.plotly_chart(fig_length, use_container_width=True, config=config)

    with tab4:
        st.subheader("📊 Statistical Distributions")

        # Votes histogram
        if 'vote_count' in df.columns:
            st.markdown("#### Histogram of Votes")
            fig_votes = px.histogram(
                df, x='vote_count', nbins=40,
                title='Distribution of Vote Counts',
                labels={'vote_count': 'Number of Votes', 'count': 'Number of Movies'},
                marginal="box"
            )
            fig_votes.add_vline(x=df['vote_count'].mean(), line_dash="dash",
                                annotation_text=f"Mean: {df['vote_count'].mean():.0f}")
            st.plotly_chart(fig_votes, use_container_width=True, config={"responsive": True, "displayModeBar": True})
        else:
            st.info("💡 vote_count column not found")

        # Top movies by number of votes
        if {'title', 'vote_count'}.issubset(df.columns):
            st.markdown("#### Movies with the Highest Number of Votes")
            top_n = st.slider("Top N (votes)", 5, 50, 15, key="top_votes_slider")
            top_votes_df = df[['title', 'vote_count']].dropna().sort_values('vote_count', ascending=False).head(top_n)
            fig_top_votes = px.bar(
                top_votes_df.sort_values('vote_count'),
                x='vote_count', y='title', orientation='h',
                title=f'Top {len(top_votes_df)} Movies by Vote Count',
                labels={'vote_count': 'Votes', 'title': 'Movie'},
                color='vote_count', color_continuous_scale='blues'
            )
            fig_top_votes.update_layout(
                yaxis={'categoryorder': 'array', 'categoryarray': top_votes_df.sort_values('vote_count')['title']})
            st.plotly_chart(fig_top_votes, use_container_width=True,
                            config={"responsive": True, "displayModeBar": True})

        # Histogram of vote_average
        if 'vote_average' in df.columns:
            st.markdown("#### Histogram of Vote Average")
            fig_va = px.histogram(
                df, x='vote_average', nbins=30,
                title='Distribution of Vote Averages',
                labels={'vote_average': 'Average Rating', 'count': 'Number of Movies'},
                marginal="box"
            )
            fig_va.add_vline(x=df['vote_average'].mean(), line_dash="dash",
                             annotation_text=f"Mean: {df['vote_average'].mean():.2f}")
            st.plotly_chart(fig_va, use_container_width=True, config={"responsive": True, "displayModeBar": True})
        else:
            st.info("💡 vote_average column not found")

        # Top movies by vote_average (with minimum votes filter)
        if {'title', 'vote_average', 'vote_count'}.issubset(df.columns):
            st.markdown("#### Movies with the Highest Vote Average")
            min_votes = st.slider("Minimum votes to consider", 0,
                                  int(df['vote_count'].max()) if 'vote_count' in df.columns else 10000, 100, step=50,
                                  key="min_votes_avg")
            top_n_avg = st.slider("Top N (rating)", 5, 50, 15, key="top_avg_slider")
            rated_df = df[df['vote_count'] >= min_votes][['title', 'vote_average', 'vote_count']].dropna()
            if len(rated_df) > 0:
                top_avg_df = rated_df.sort_values(['vote_average', 'vote_count'], ascending=[False, False]).head(
                    top_n_avg)
                fig_top_avg = px.bar(
                    top_avg_df.sort_values('vote_average'),
                    x='vote_average', y='title', orientation='h',
                    title=f'Top {len(top_avg_df)} Movies by Vote Average (min {min_votes} votes)',
                    labels={'vote_average': 'Average Rating', 'title': 'Movie'},
                    color='vote_average', color_continuous_scale='viridis'
                )
                fig_top_avg.update_layout(height=600, yaxis={'categoryorder': 'array',
                                                             'categoryarray': top_avg_df.sort_values('vote_average')[
                                                                 'title']})
                st.plotly_chart(fig_top_avg, use_container_width=True,
                                config={"responsive": True, "displayModeBar": True})
            else:
                st.info("💡 No movies meet the minimum vote threshold.")


        # Rating distribution
        if 'vote_average' in df.columns:
            fig_rating = px.histogram(
                df,
                x='vote_average',
                nbins=20,
                title='Distribution of Movie Ratings',
                labels={'vote_average': 'Average Rating', 'count': 'Number of Movies'},
                marginal="box"
            )
            fig_rating.add_vline(x=df['vote_average'].mean(), line_dash="dash",
                                 annotation_text=f"Mean: {df['vote_average'].mean():.2f}")
            config = {"responsive": True, "displayModeBar": True}
            st.plotly_chart(fig_rating, use_container_width=True, config=config)

        # Runtime distribution
        if 'runtime' in df.columns:
            fig_runtime = px.histogram(
                df,
                x='runtime',
                nbins=20,
                title='Distribution of Movie Runtimes',
                labels={'runtime': 'Runtime (minutes)', 'count': 'Number of Movies'}
            )
            fig_runtime.add_vline(x=df['runtime'].mean(), line_dash="dash",
                                  annotation_text=f"Mean: {df['runtime'].mean():.0f} min")
            config = {"responsive": True, "displayModeBar": True}
            st.plotly_chart(fig_runtime, use_container_width=True, config=config)

    with tab5:
        st.subheader("🔗 Feature Correlations")

        # Correlation analysis
        numeric_cols = ['vote_average', 'vote_count', 'popularity', 'runtime', 'budget', 'revenue']
        available_cols = [col for col in numeric_cols if col in df.columns]

        if len(available_cols) > 1:
            corr_matrix = df[available_cols].corr()

            fig_heatmap = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Correlation Matrix of Movie Features",
                color_continuous_scale='RdBu',
                color_continuous_midpoint=0
            )
            config = {"responsive": True, "displayModeBar": True}
            st.plotly_chart(fig_heatmap, use_container_width=True, config=config)

            # Summary statistics
            st.subheader("📊 Summary Statistics")
            st.dataframe(df[available_cols].describe().round(2), use_container_width=True)


def extract_countries(x):
    """Extract country names from production_countries column"""
    try:
        if pd.isna(x) or x == '':
            return []
        if isinstance(x, str):
            countries_list = literal_eval(x)
            if isinstance(countries_list, list):
                return [country.get('name', '') for country in countries_list if isinstance(country, dict)]
    except:
        pass
    return []


def create_word_cloud_data(text_series, max_words=100):
    """Create word frequency data for word cloud visualization"""
    # Combine all text
    all_text = ' '.join(text_series.fillna('').astype(str))

    # Simple word frequency analysis
    from collections import Counter
    import re

    # Clean text: remove punctuation, convert to lowercase
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', all_text.lower())

    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are',
                  'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                  'could', 'should', 'may', 'might', 'must', 'can', 'his', 'her', 'their', 'him', 'she', 'they', 'them',
                  'this', 'that', 'these', 'those', 'i', 'you', 'he', 'we', 'from', 'up', 'out', 'down', 'off', 'over',
                  'under', 'again', 'further', 'then', 'once'}

    # Split into words and count frequencies
    words = [word for word in cleaned_text.split() if len(word) > 2 and word not in stop_words]

    word_freq = Counter(words).most_common(max_words)

    return word_freq
