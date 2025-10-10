# Movie Recommendation System - Streamlit App

This is a comprehensive movie recommendation system built with Streamlit, based on the TMDB 5000 Movie Dataset analysis.

## Features

### 1. Demographic Filtering
- Weighted rating system using IMDB formula
- Top-rated movies based on vote count and average rating
- Popular movies visualization

### 2. Content-Based Filtering
- **Plot-based recommendations**: Uses TF-IDF vectorization of movie overviews
- **Metadata-based recommendations**: Uses cast, director, genres, and keywords
- Interactive movie search and selection
- Similarity score visualizations

### 3. Movie Analysis Dashboard
- Comprehensive dataset statistics
- Interactive visualizations including:
  - Rating and runtime distributions
  - Budget vs Revenue analysis
  - ROI calculations
  - Correlation heatmaps
  - Various scatter plots and box plots

## Installation

1. Install required dependencies:
```bash
python -m venv venv
.\venv\Scripts\activate

pip install -r requirements.txt
```

2. Run the Streamlit application:
```bash
streamlit run movie_recommendation_app.py
```

## Usage

### Data Upload
- Upload your own TMDB credits and movies CSV files using the sidebar
- If no files are uploaded, the app uses high-quality sample data with classic movies

### Navigation
Use the sidebar to navigate between different sections:
- **Demographic Filtering**: General recommendations based on popularity
- **Content-Based Filtering**: Personalized recommendations based on movie similarity
- **Movie Analysis Dashboard**: Comprehensive data analysis and visualizations

### Getting Recommendations
1. Go to the Content-Based Filtering section
2. Search for or select a movie from the dropdown
3. Choose recommendation type (Plot-based or Metadata-based)
4. Adjust the number of recommendations
5. Click "Get Recommendations" to see similar movies

## Technical Details

### Algorithms Used
- **TF-IDF Vectorization**: For plot-based content filtering
- **Count Vectorization**: For metadata-based content filtering  
- **Cosine Similarity**: For measuring movie similarity
- **IMDB Weighted Rating**: For demographic filtering

### Data Processing
- Handles JSON-like string data for cast, crew, genres, and keywords
- Text preprocessing and normalization
- Feature extraction and similarity computation

## File Structure
```
├── movie_recommendation_app.py  # Main Streamlit application
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## Sample Data
The app includes sample data with 30 classic movies including:
- The Dark Knight, Inception, The Matrix
- Pulp Fiction, The Godfather, Forrest Gump
- And many more acclaimed films

## Future Enhancements
- Collaborative filtering implementation
- Movie poster integration
- User rating and review system
- Advanced filtering options
- Export recommendations functionality

## Credits
Based on the TMDB 5000 Movie Dataset analysis and recommendation system concepts.
