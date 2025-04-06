# Sentiment Analysis Dashboard

This directory contains the Streamlit dashboard for visualizing sentiment analysis of fire brigade incident reports.

## Dashboard Features

The dashboard provides the following visualizations and tools:

1. **Summary Metrics**:
   - Count and percentage of positive, neutral, and negative sentiment incidents
   - Visual indicators of sentiment distributions

2. **Sentiment Distribution**:
   - Pie chart visualization of incident sentiment distribution
   - Color-coded segments for easy interpretation

3. **Sentiment Trends**:
   - Time series chart showing sentiment trends over time
   - Adjustable time range for different analysis periods
   - Interactive elements for exploring specific dates

4. **Recent Incidents**:
   - List of recent incident reports with sentiment analysis
   - Color-coded for quick recognition of sentiment
   - Timestamp and full report text

5. **Sentiment Analysis Tool**:
   - Interactive tool for analyzing custom text
   - Real-time sentiment prediction
   - Confidence score visualization

## Features

- Real-time sentiment analysis for emergency services text data
- Interactive dashboard with sentiment distribution visualizations
- Historical sentiment trends over time
- Multi-model support with model selection and comparison
- Active learning feedback loop for model improvement
- Export functionality for data analysis

## Related Pages

### Social Media Data Analysis

The Social Media Data Analysis page provides dedicated insights for social media content:

- Sentiment summary statistics for social media posts
- Platform comparison of sentiment distribution 
- Daily sentiment chart showing day-by-day sentiment trends
- Social media post feed with sentiment analysis

Access the Social Media Data Analysis page through the navigation links in the dashboard.

### Daily Sentiment Chart (Social Media Page)

The Daily Sentiment Chart has been implemented in the Social Media Data Analysis page, providing a detailed view of sentiment trends for each day of a selected month. This visualization helps identify daily patterns and specific days with notable sentiment shifts.

**Features:**
- Line chart showing sentiment counts for each day of the month
- Separate lines for positive, neutral, and negative sentiments
- Interactive tooltips displaying exact counts
- Month and year selection controls
- Data download functionality
- Platform filtering capabilities

This component is specifically designed for social media data analysis and can be found in the Social Media Data Analysis page.

## Running the Dashboard

To run the dashboard locally:

```bash
streamlit run app/dashboard/dashboard.py
```

The dashboard will be available at http://localhost:8501 by default.

## Auto-Refresh

The dashboard includes an auto-refresh feature that periodically updates the visualizations with the latest data. This can be toggled on/off in the sidebar, and the refresh interval can be adjusted.

## Data Sources

The dashboard pulls data from:

1. **Database**: For historical incident reports and sentiment analysis
2. **Model API**: For real-time sentiment prediction of custom text

## Customization

The dashboard appearance and settings can be customized by modifying:

- `app/utils/config.py`: Dashboard title and default refresh interval
- `app/dashboard/dashboard.py`: Layout, charts, and visualization settings 