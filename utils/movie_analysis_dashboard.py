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

    with tab3:
        st.subheader("📄 Text Analysis")

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
