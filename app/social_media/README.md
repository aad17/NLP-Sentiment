# Social Media Integration for Sentiment Analysis

This module provides integration with social media platforms for sentiment analysis of posts and content. It enables the application to fetch data from various social media platforms, analyze the sentiment of the content, and display the results in a dashboard.

## Features

- **Multi-platform support**: Twitter, Facebook, LinkedIn, and custom platforms
- **Secure credential management**: Store API keys securely for different platforms
- **Automated data retrieval**: Scheduled batch processing of social media data
- **Sentiment analysis**: Integration with existing NLP models for sentiment prediction
- **Interactive dashboard**: Visualize sentiment trends across platforms
- **Performance tracking**: Monitor model performance and sentiment distribution

## Components

The social media integration is structured into several components:

1. **Authentication UI**: Streamlit interface for entering social media credentials
   - `social_media_auth.py`: Streamlit page for credential management

2. **Data Connectors**: Platform-specific API connectors
   - `social_media_connector.py`: Classes for connecting to social media platforms

3. **Data Storage**: Database schema and storage procedures
   - `database.py`: Functions for storing social media posts and sentiment

4. **Batch Processing**: Scheduled data retrieval and analysis
   - `data_processor.py`: Batch processing and analysis logic

5. **Data Visualization**: Dashboard for displaying results
   - `social_media_data.py`: Streamlit page for data visualization

6. **Initialization**: Script to start all components
   - `init.py`: Launch all social media components

## Getting Started

### Prerequisites

- Python 3.7+
- Streamlit
- Flask
- Required Python packages: `requests`, `pandas`, `matplotlib`, `altair`, `schedule`

### Installation

1. Ensure all dependencies are installed:
```bash
pip install streamlit flask pandas matplotlib altair schedule requests
```

2. Create the credentials directory:
```bash
mkdir -p app/data/credentials
```

### Running the Components

You can use the initialization script to start all components:

```bash
python app/social_media/init.py
```

Or start individual components:

```bash
# Start only the API
python app/social_media/init.py --components api

# Start only the authentication UI
python app/social_media/init.py --components auth

# Start only the data visualization
python app/social_media/init.py --components data

# Start only the batch processor
python app/social_media/init.py --components processor
```

You can also specify custom ports:

```bash
python app/social_media/init.py --port-api 5000 --port-auth 8501 --port-data 8502
```

## Usage Flow

1. **Set up API credentials**:
   - Navigate to the authentication UI (default: http://localhost:8501)
   - Enter your API credentials for each platform
   - Test connections to ensure valid credentials

2. **Fetch social media data**:
   - Use the "Fetch New Data" button on the data visualization page
   - Or let the scheduler automatically fetch data periodically

3. **View sentiment analysis**:
   - Navigate to the data visualization UI (default: http://localhost:8502)
   - Explore sentiment trends and platform comparisons
   - Filter by platform, time range, etc.

4. **Monitor batch processing**:
   - Check the scheduler status using the API endpoint:
   ```
   curl http://localhost:5000/social_media/scheduler/status
   ```

## API Endpoints

The following API endpoints are available for integrating with the social media component:

### Social Media Data Endpoints

- `POST /social_media/fetch`: Fetch data from social media platforms
- `POST /social_media/process`: Process social media data with sentiment analysis
- `POST /social_media/batch_job`: Run a social media batch job
- `GET /social_media/incidents`: Get incidents enriched with social media context

### Scheduler Endpoints

- `POST /social_media/scheduler/start`: Start the scheduler
- `POST /social_media/scheduler/stop`: Stop the scheduler
- `GET /social_media/scheduler/status`: Get scheduler status

## Extending for New Platforms

To add support for a new social media platform:

1. Create a new connector class in `social_media_connector.py` that inherits from `SocialMediaConnector`
2. Implement the required methods: `connect()`, `get_posts()`, and `_format_post()`
3. Add a form for the new platform's credentials in `social_media_auth.py`
4. Update the platform filter options in `social_media_data.py`

## Troubleshooting

- **API Connection Issues**: Check that the credentials are correctly entered and that your API keys have the necessary permissions.
- **Missing Data**: Verify that the batch processor is running and check the logs for any errors.
- **Visualization Problems**: Ensure that there is data in the database for the selected time range and platform.

## Contributing

When contributing to this module, please ensure that you:

1. Follow the existing code style and structure
2. Add appropriate error handling
3. Update documentation for any new features
4. Add tests for new functionality

## License

This module is part of the main application and is covered by the same license.

## Credits

This module was created as part of the social media sentiment analysis workflow and integrates with the existing sentiment analysis models developed for the application. 