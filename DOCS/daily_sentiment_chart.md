# Daily Sentiment Chart Documentation

## Overview

The Daily Sentiment Chart is a new feature in the Sentiment Analysis Dashboard that provides a granular view of sentiment trends for each day of a selected month. This visualization helps users identify daily patterns and specific days with notable sentiment shifts across emergency services communications.

## Technical Implementation

### Data Processing

The daily sentiment data is processed through the following workflow:

1. **Data Retrieval**: The `get_sentiment_by_day_of_month()` function in `data_service.py` retrieves incident data from the database.
2. **Filtering**: Data is filtered to include only records from the selected month and year.
3. **Aggregation**: Sentiment counts are aggregated by day of month for each sentiment category (positive, neutral, negative).
4. **Data Preparation**: A complete dataset is created with all days of the month represented, even if no data exists for certain days.

### Visualization

The chart is implemented using Altair, providing:

- **Line Chart**: Shows trends over time with markers for each data point
- **Color Coding**: Green for positive, gray for neutral, red for negative sentiments
- **Interactive Features**: Tooltips, zooming, and panning capabilities

## User Interface

### Controls

The user interface provides several controls for the Daily Sentiment Chart:

1. **Month Selector**: Dropdown in the sidebar to select the month to display
2. **Year Selector**: Dropdown in the sidebar to select the year to display
3. **Interactive Chart**: Ability to hover, click, and drag for detailed exploration
4. **Download Button**: Option to download the raw data as a CSV file

### Data Interpretation

The chart helps users identify:

- **Daily Trends**: How sentiment changes from day to day
- **Weekend vs. Weekday Patterns**: Potential differences in sentiment based on day of week
- **Event Correlation**: Spikes or drops in sentiment that may correlate with known events
- **Periodic Patterns**: Recurring patterns within months (e.g., beginning/end of month effects)

## Use Cases

### Operational Analysis

- **Response Planning**: Identify days of the month with historically negative sentiment to prepare additional resources
- **Performance Tracking**: Monitor how daily sentiment trends correlate with operational changes

### Communication Strategy

- **Message Timing**: Determine optimal days for positive communications
- **Impact Assessment**: Measure the daily impact of communication campaigns on public sentiment

### Incident Analysis

- **Post-Incident Tracking**: Monitor sentiment recovery after negative incidents
- **Pattern Recognition**: Identify recurring patterns that may indicate systemic issues

## Data Export and Further Analysis

The download feature allows users to:

1. Export daily sentiment data as CSV
2. Perform additional analysis in external tools
3. Create custom reports for specific time periods
4. Combine with other datasets for deeper insights

## Technical Notes

- The chart supports all months of the year, automatically adjusting for the correct number of days (28-31)
- Empty data points (days with no incidents) are shown as zero rather than omitted
- The visualization maintains consistent color coding with other dashboard elements
- If no data is available for the selected month/year, an appropriate message is displayed

## Future Enhancements

Potential future improvements for the Daily Sentiment Chart include:

1. Comparative view of multiple months or years
2. Day-of-week analysis (grouping by Monday, Tuesday, etc.)
3. Anomaly detection to highlight unusual daily patterns
4. Integration with external event calendars for correlation analysis 