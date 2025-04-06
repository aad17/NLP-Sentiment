# Sentiment Analysis Application

A cost-effective sentiment analysis application for a fire brigade company to analyze incident reports and visualize sentiment trends.

## Project Overview

This application:
- Ingests incident reports from a (mock) API
- Analyzes sentiment (positive, neutral, negative) with multiple model options
- Visualizes sentiment data in an interactive dashboard
- Uses free-tier services for cost-effectiveness

## Features

- **Mock API**: Simulates a fire brigade incident reporting API
- **Multiple Sentiment Models**: Choose between domain-specific, social media-based, hybrid, or expanded approaches
- **Interactive Dashboard**: Visualization of sentiment trends and statistics
- **Database**: SQLite storage for incidents and analysis results
- **Deployment**: Ready for Render.com deployment

## Sentiment Models

The application includes multiple sentiment analysis models:

1. **Synthetic Model**: Trained on synthetic templates specific to fire brigade domain. Provides 80% accuracy on domain-specific text.

2. **Twitter Model**: Trained on real Twitter data with diverse contexts. Better for general expressions but less accurate (40%) on domain-specific content.

3. **Hybrid Model**: A combined approach that leverages both models based on text domain and confidence scores. Provides 77.8% accuracy on domain text with improved generalization.

4. **Domain-Aware Model**: Uses specialized classification for each emergency service domain, offering improved accuracy (85-92%) and better performance with neutral sentiments.

5. **Expanded Model**: Trained on multiple large datasets (160,000+ tweets) with 73.5% overall accuracy. Best for general-purpose sentiment analysis across multiple domains.

## Components

1. **Mock API**: Provides simulated incident data
2. **Data Ingestion**: Collects and stores incident reports
3. **Sentiment Models**: Multiple approaches to analyze text sentiment
4. **Model API**: Serves sentiment predictions with model selection
5. **Dashboard**: Visualizes trends and statistics with model switching capability

## Setup Instructions

### Quick Start

#### On Linux/Mac:

The easiest way to get started is to use the start script:

```bash
# Make the script executable if needed
chmod +x start_app.sh

# Start all components (Mock API, Model API, Dashboard)
./start_app.sh
```

To stop the application:

```bash
./stop_app.sh
```

#### On Windows:

Use the provided batch files to start and stop the application:

```
# Start all components (Mock API, Model API, Dashboard)
start_app.bat
```

To stop the application:

```
stop_app.bat
```

The batch files perform the same functions as their shell script counterparts and include additional checks for the expanded sentiment model.

### Manual Setup

For more control, you can set up and run each component separately:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Sentiment_Analysis.git
   cd Sentiment_Analysis
   ```

2. Create and activate a virtual environment:
   ```bash
   # On Linux/Mac:
   python -m venv venv
   source venv/bin/activate
   
   # On Windows:
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Setup environment variables:
   ```bash
   # On Linux/Mac:
   cp .env.example .env
   
   # On Windows:
   copy .env.example .env
   ```
   Edit the `.env` file with your configuration.

5. Initialize the application:
   ```bash
   python setup.py
   ```

6. Start each component manually:
   ```bash
   # In separate terminals:
   python app/api/mock_api.py
   python app/api/model_api.py
   streamlit run app/dashboard/dashboard.py
   ```

## Using the Dashboard

1. Access the dashboard at http://localhost:8501
2. Use the model selection dropdown in the sidebar to switch between sentiment models:
   - Synthetic (Fire Brigade Domain)
   - Twitter (General Social Media)
   - Hybrid (Combined Approach)
   - Domain-Aware (Enhanced Model)
   - Expanded (Multi-Dataset General Model)
3. Enter text in the "Sentiment Prediction Tool" to analyze sentiment
4. Check "Compare all models" to see predictions from all available models side by side

## API Endpoints

### Model API (http://localhost:5002)

- **GET /health**: Check API health and available models
- **GET /models**: List available sentiment models
- **POST /predict**: Predict sentiment for a text
  - Request body: `{"text": "Your text here", "model_type": "synthetic"}`
- **POST /compare**: Compare predictions from all models
  - Request body: `{"text": "Your text here"}`

### Mock API (http://localhost:5001)

- **GET /get_incidents**: Get simulated incident reports
- **GET /health**: Check API health

## Documentation

Additional documentation is available in the `DOCS` directory:
- [API Documentation](app/api/README.md)
- [Model Documentation](app/models/README.md)
- [Dashboard Documentation](app/dashboard/README.md)
- [Deployment Guide](DOCS/render_deployment.md)

## Model Comparison

| Model | Domain-Specific Accuracy | General Text | Best For |
|-------|--------------------------|--------------|----------|
| Synthetic | 80% | 33.3% | Fire brigade specific text |
| Twitter | 40% | 33.3% | Social media expressions |
| Hybrid | 77.8% | 25% | Balance of domain knowledge and real-world patterns |
| Domain-Aware | 85-92% | 60% | Emergency services domain with improved neutral detection |
| Expanded | 47.5% | 73.5% | General-purpose sentiment across multiple domains |

## Deployment

To deploy to Render.com, follow the instructions in the [Deployment Guide](DOCS/render_deployment.md).

## License

MIT

## Contributors

- Your Name <your.email@example.com>