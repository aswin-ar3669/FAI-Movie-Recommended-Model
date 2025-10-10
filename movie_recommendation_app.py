
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
import streamlit as st
import plotly.express as px
import warnings

from utils.content_based_filtering import content_based_filtering
from utils.demographic_filtering import demographic_filtering

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
st.markdown("### The Age of Recommender Systems")

# Add description
st.markdown("""
The rapid growth of data collection has led to a new era of information. Data is being used to create more efficient 
systems and this is where Recommendation Systems come into play. Recommendation Systems are a type of **information 
filtering systems** as they improve the quality of search results and provides items that are more relevant to the 
search item or are related to the search history of the user.

**Types of Recommender Systems:**
1. **Demographic Filtering** - Generalized recommendations based on popularity
2. **Content Based Filtering** - Recommendations based on item similarity
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
            df_movies = pd.read_csv(movies_file)

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
                df_credits = pd.read_csv(credits_file)

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

        else:
            # Create enhanced sample data for better visualizations
            np.random.seed(42)

            sample_movies = [
                "The Dark Knight", "Inception", "The Matrix", "Pulp Fiction", "The Godfather",
                "Forrest Gump", "Fight Club", "The Lord of the Rings", "Star Wars", "Goodfellas",
                "The Shawshank Redemption", "Schindlers List", "Casablanca", "Gone with the Wind",
                "Lawrence of Arabia", "Citizen Kane", "The Wizard of Oz", "Vertigo", "Psycho",
                "Sunset Boulevard", "On the Waterfront", "Some Like It Hot", "Singin in the Rain",
                "The Bridge on the River Kwai", "All About Eve", "North by Northwest", "Doctor Zhivago",
                "Apocalypse Now", "The Treasure of the Sierra Madre", "The Best Years of Our Lives",
                "Taxi Driver", "Raging Bull", "The Deer Hunter", "Annie Hall", "Manhattan",
                "Chinatown", "The French Connection", "One Flew Over the Cuckoos Nest", "Rocky",
                "Jaws", "The Exorcist", "2001 A Space Odyssey", "Bonnie and Clyde", "The Graduate",
                "Midnight Cowboy", "Butch Cassidy and the Sundance Kid", "MASH", "Patton", "Love Story",
                "Terminator 2", "Alien", "Blade Runner", "Raiders of the Lost Ark", "E.T."
            ]

            # Enhanced sample data with more realistic distributions
            sample_countries = [
                'United States', 'United Kingdom', 'France', 'Germany', 'Italy', 'Spain', 'Japan', 'Canada',
                'Australia', 'India', 'China', 'South Korea', 'Brazil', 'Mexico', 'Russia', 'Sweden'
            ] * 4  # Repeat to have enough for all movies

            sample_years = np.random.choice(range(1960, 2024), len(sample_movies))

            sample_overviews = [
                "Batman begins his fight against crime with his greatest test facing the Joker in Gotham City.",
                "A thief who steals corporate secrets through dream-sharing technology explores subconscious minds.",
                "A computer hacker learns from mysterious rebels about the true nature of virtual reality.",
                "The lives of two mob hitmen, a boxer, and others intertwine in violent criminal ways.",
                "The aging patriarch of organized crime dynasty transfers control to his reluctant son.",
                "The presidencies of Kennedy and Johnson through the eyes of Alabama man with low IQ.",
                "An insomniac office worker and devil-may-care soapmaker form underground fight club.",
                "A meek Hobbit and eight companions set out to destroy the powerful One Ring.",
                "Luke Skywalker joins forces with rebels to save Princess Leia from evil Empire.",
                "A young man becomes a mobster in Italian-American crime family during the 1950s.",
                "Two imprisoned men bond over years, finding solace and eventual redemption through friendship.",
                "In German-occupied Poland, industrialist Oskar Schindler saves more than thousand Jews.",
                "A cynical nightclub owner protects his old flame and her husband from Nazi officials.",
                "A manipulative woman and her genteel husband live on plantation during Civil War era.",
                "The story of T.E. Lawrence, an English officer during the Arab Revolt against Turks.",
                "Following the life of newspaper tycoon Charles Foster Kane, from young heir to media mogul.",
                "Dorothy is swept away to magical land of Oz and must find her way home.",
                "A former police detective suffering from acrophobia investigates mysterious woman in San Francisco.",
                "A Phoenix secretary embezzles money from her employer and flees to remote Bates Motel.",
                "A faded movie star and young screenwriter develop complicated and dangerous relationship.",
                "Terry Malloy dreams of being prize fighter while working for corrupt union boss.",
                "Two musicians disguise themselves as women to escape from dangerous gangsters in Chicago.",
                "A silent movie production company struggles during Hollywood transition to sound era.",
                "British POWs are forced to build bridge for Japanese captors during World War.",
                "A seemingly successful aging actress reveals her insecurities to young ambitious playwright.",
                "An advertising executive is pursued by foreign spies after case of mistaken identity.",
                "The life of Russian physician during tumultuous period of Bolshevik Revolution and civil war.",
                "Captain Willard travels upriver through Vietnamese jungle to assassinate rogue Colonel Kurtz.",
                "Three prospectors seek gold in Sierra Madre mountains, but greed gets better of them.",
                "Three World War II veterans struggle to readjust to civilian life in post-war America.",
                "A mentally unstable taxi driver attempts to save young prostitute from dangerous pimp.",
                "Jake LaMotta rises from mean streets of Bronx to become middleweight boxing champion.",
                "A group of friends go on hunting trip that turns into nightmare during Vietnam War.",
                "A neurotic stand-up comedian falls in love with aspiring singer in New York City.",
                "A successful television writer explores complex relationships in Manhattan upper-class society.",
                "A private investigator uncovers corruption and murder in 1930s Los Angeles during water wars.",
                "A tough New York cop and his partner investigate international drug smuggling operation.",
                "A petty criminal pleads insanity after getting arrested for brutal murder of family.",
                "A small-time boxer from Philadelphia gets once-in-lifetime chance to fight heavyweight champion.",
                "A massive great white shark terrorizes New England beach town during busy summer season.",
                "A young girl becomes possessed by ancient demon and two Catholic priests try to save her.",
                "Humanity discovers mysterious monolith that appears to influence human evolution throughout history.",
                "A young couple goes on violent crime spree across American South during Great Depression.",
                "A college graduate returns home and has illicit affair with seductive older married woman.",
                "A naive male hustler travels to New York City to seek his fortune as gigolo.",
                "Two aging outlaws try to survive in rapidly changing American West of early 1900s.",
                "The staff of mobile army surgical hospital cope with Korean War through dark humor.",
                "General George Patton leads US Third Army through North African and European campaigns.",
                "A young man from working class and wealthy woman from high society fall in tragic love.",
                "A cyborg assassin sent from post-apocalyptic future attempts to kill mother of resistance leader.",
                "The crew of commercial space tug encounters deadly alien life form on distant planet.",
                "A blade runner must pursue and terminate four replicants who stole ship and returned to Earth.",
                "Archaeologist Indiana Jones races against Nazis to find Ark of Covenant before they do.",
                "A gentle alien botanist becomes stranded on Earth and befriends lonely young boy."
            ]

            # Create comprehensive sample data
            movies_data = {
                'id': range(1, len(sample_movies) + 1),
                'title': sample_movies,
                'overview': sample_overviews,
                'vote_average': np.random.uniform(6.0, 9.5, len(sample_movies)),
                'vote_count': np.random.randint(100, 25000, len(sample_movies)),
                'popularity': np.random.uniform(5, 150, len(sample_movies)),
                'runtime': np.random.randint(80, 220, len(sample_movies)),
                'budget': np.random.randint(1000000, 300000000, len(sample_movies)),
                'revenue': np.random.randint(2000000, 800000000, len(sample_movies)),
                'release_date': pd.to_datetime([f'{year}-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}' for year in sample_years]),
                'production_countries': [f'[{{"name": "{country}"}}]' for country in sample_countries[:len(sample_movies)]],
                'genres': ['[{"name": "Action"}, {"name": "Drama"}]'] * len(sample_movies),
                'keywords': ['[{"name": "hero"}, {"name": "adventure"}]'] * len(sample_movies),
                'cast': ['[{"name": "Actor One"}, {"name": "Actor Two"}]'] * len(sample_movies),
                'crew': ['[{"job": "Director", "name": "Director Name"}]'] * len(sample_movies)
            }

            df = pd.DataFrame(movies_data)

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
    stop_words = set([
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'his', 'her', 'their',
        'him', 'she', 'they', 'them', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'we',
        'from', 'up', 'out', 'down', 'off', 'over', 'under', 'again', 'further', 'then', 'once'
    ])

    # Split into words and count frequencies
    words = [word for word in cleaned_text.split() 
             if len(word) > 2 and word not in stop_words]

    word_freq = Counter(words).most_common(max_words)

    return word_freq

# Main application
def main():
    # Sidebar for navigation
    st.sidebar.title("📍 Navigation")
    page = st.sidebar.selectbox("Choose a section:", [
        "📊 Demographic Filtering",
        "🎯 Content-Based Filtering", 
        "📈 Movie Analysis Dashboard"
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
    if page == "📊 Demographic Filtering":
        demographic_filtering(df)
    elif page == "🎯 Content-Based Filtering":
        content_based_filtering(df)
    elif page == "📈 Movie Analysis Dashboard":
        movie_analysis_dashboard(df)


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
                st.metric("💰 Total Budget", f"${total_budget/1e9:.1f}B")
            else:
                st.metric("💰 Total Budget", f"${total_budget/1e6:.0f}M")

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
            st.plotly_chart(fig_yearly, use_container_width=True,config=config)

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
            st.plotly_chart(fig_decade, use_container_width=True,config=config)

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
                st.plotly_chart(fig_countries, use_container_width=True,config=config)

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
            st.plotly_chart(fig_geo_pie, use_container_width=True,config=config)

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
                st.plotly_chart(fig_words, use_container_width=True,config=config)

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
            st.plotly_chart(fig_length, use_container_width=True,config=config)

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
            st.plotly_chart(fig_rating, use_container_width=True,config=config)

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
            st.plotly_chart(fig_runtime, use_container_width=True,config=config)

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
            st.plotly_chart(fig_heatmap, use_container_width=True,config=config)

            # Summary statistics
            st.subheader("📊 Summary Statistics")
            st.dataframe(df[available_cols].describe().round(2), use_container_width=True)

if __name__ == "__main__":
    main()