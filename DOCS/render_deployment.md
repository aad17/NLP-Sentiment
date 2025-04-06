# Deploying Sentiment Analysis Application to Render

This guide explains how to deploy the Sentiment Analysis application to Render, a cloud platform for web applications.

## Prerequisites

- A Render account (sign up at [render.com](https://render.com))
- Git repository with your Sentiment Analysis application code

## Deployment Overview

The application now consists of seven components that need to be deployed separately:

1. **Mock API**: Provides simulated incident data
2. **Model API**: Serves sentiment predictions
3. **Main Dashboard**: Visualizes the sentiment analysis
4. **Social Media Batch Processor**: Processes social media data in batches
5. **Social Media Authentication UI**: Manages social media platform authentication
6. **Social Media Data Visualization**: Displays social media sentiment analysis
7. **CSV Data Upload Page**: Allows uploading social media data via CSV

## Step 1: Prepare Your Repository

Ensure your repository includes these required files:

- `requirements.txt`: Lists all dependencies
- `render.yaml`: Defines the Render Blueprint (see below)

Update your `requirements.txt` to include all necessary packages:

```
streamlit>=1.24.0
pandas>=1.5.3
numpy>=1.24.2
scikit-learn>=1.2.2
matplotlib>=3.7.1
altair>=5.0.1
requests>=2.28.2
sqlite3>=3.41.2
python-dotenv>=1.0.0
```

Create a `render.yaml` file in your root directory:

```yaml
services:
  # Mock API Service
  - name: sentiment-mock-api
    type: web
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python app/api/mock_api.py
    envVars:
      - key: API_PORT
        value: 10000
      - key: PYTHON_VERSION
        value: 3.9.0
      - key: DATABASE_URL
        value: postgresql://username:password@host/database_name

  # Model API Service
  - name: sentiment-model-api
    type: web
    env: python
    buildCommand: pip install -r requirements.txt && python app/models/train_model.py
    startCommand: python app/api/model_api.py
    envVars:
      - key: API_PORT
        value: 10001
      - key: PYTHON_VERSION
        value: 3.9.0
      - key: API_HOST
        fromService:
          name: sentiment-mock-api
          type: web
          property: host
      - key: DATABASE_URL
        value: postgresql://username:password@host/database_name

  # Main Dashboard Service
  - name: sentiment-dashboard
    type: web
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app/dashboard/dashboard.py
    envVars:
      - key: API_HOST
        fromService:
          name: sentiment-mock-api
          type: web
          property: host
      - key: MODEL_API_HOST
        fromService:
          name: sentiment-model-api
          type: web
          property: host
      - key: PYTHON_VERSION
        value: 3.9.0

  # Social Media Batch Processor
  - name: sentiment-social-processor
    type: worker
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python app/social_media/data_processor.py
    envVars:
      - key: PYTHON_VERSION
        value: 3.9.0
      - key: MODEL_API_HOST
        fromService:
          name: sentiment-model-api
          type: web
          property: host

  # Social Media Authentication UI
  - name: sentiment-social-auth
    type: web
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app/social_media/social_media_auth.py
    envVars:
      - key: PYTHON_VERSION
        value: 3.9.0
      - key: PORT
        value: 10002

  # Social Media Data Visualization
  - name: sentiment-social-data
    type: web
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app/social_media/social_media_data.py
    envVars:
      - key: PYTHON_VERSION
        value: 3.9.0
      - key: PORT
        value: 10003
      - key: MODEL_API_HOST
        fromService:
          name: sentiment-model-api
          type: web
          property: host

  # CSV Data Upload Page
  - name: sentiment-csv-upload
    type: web
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app/social_media/csv_data_upload.py
    envVars:
      - key: PYTHON_VERSION
        value: 3.9.0
      - key: PORT
        value: 10004
      - key: MODEL_API_HOST
        fromService:
          name: sentiment-model-api
          type: web
          property: host
```

## Step 2: Set Up a Shared Database

Since multiple services need to access the same databases, you have two options:

### Option 1: Create a New PostgreSQL Database on Render

1. Log in to your Render account
2. Go to the Dashboard and click "New"
3. Select "PostgreSQL"
4. Set a name (e.g., "sentiment-analysis-db")
5. Choose an appropriate plan
6. Click "Create Database"

Once created, note the following connection details:
- Internal Database URL
- External Database URL
- Database Name
- Username
- Password

Update your `render.yaml` to reference this database for all services using the `fromDatabase` property:

```yaml
# Add this to each service that needs database access
envVars:
  - key: DATABASE_URL
    fromDatabase:
      name: sentiment-analysis-db
      property: connectionString
```

### Option 2: Use an Existing PostgreSQL Database

If you already have a PostgreSQL database set up on Render or elsewhere:

1. Note the connection details:
   - Internal Database URL (for services within Render)
   - External Database URL (for external connections)
   - Database credentials

2. Update your `render.yaml` to use the database connection string directly:

```yaml
# Add this to each service that needs database access
envVars:
  - key: DATABASE_URL
    value: postgresql://username:password@host/database_name
```

Replace `username`, `password`, `host`, and `database_name` with your actual database credentials.

## Step 3: Deploy Using Render Blueprint

1. Log in to your Render account
2. Go to the Dashboard and click "New"
3. Select "Blueprint"
4. Connect your Git repository
5. Render will detect the `render.yaml` file and set up the services
6. Click "Apply" to create and deploy all services

## Step 4: Configure Environment Variables

After deployment, verify these environment variables in each service:

- `API_HOST`: URL of the Mock API service
- `API_PORT`: Port for each service
- `MODEL_API_URL`: Full URL of the Model API service
- `DATABASE_URL`: Connection string to your PostgreSQL database (should already be configured in render.yaml)

## Step 5: Initialize the Databases

Both the main database and social media database need to be initialized in your PostgreSQL database:

### Initialize Main Database

1. In your Render dashboard, go to the Model API service
2. Click "Shell" to open a terminal
3. Verify the DATABASE_URL environment variable is correctly set:
   ```
   echo $DATABASE_URL
   ```
4. Run: `python -c "from app.data.database import init_db; init_db()"`
5. Then run: `python app/data/data_ingestion.py --count 50`

### Initialize Social Media Database

1. Go to the Social Media Batch Processor service
2. Click "Shell" to open a terminal
3. Verify the DATABASE_URL environment variable:
   ```
   echo $DATABASE_URL
   ```
4. Run: `python -c "from app.social_media.database import init_db; init_db()"`

## Step 6: Set Up Database Migration

If you have existing data in SQLite databases that you want to migrate to your PostgreSQL database:

1. Add the migration script to your repository as `scripts/migrate_to_postgres.py` (see full script below)

2. Update all database connection functions to use PostgreSQL when deployed:
   - Modify `app/data/database.py` and `app/social_media/database.py` to check for `DATABASE_URL` environment variable
   - Use psycopg2 for PostgreSQL connection when available

3. Run the migration script in the shell of any service, ensuring the `DATABASE_URL` points to your PostgreSQL database:
   ```
   echo $DATABASE_URL  # Verify this is your PostgreSQL connection string
   python scripts/migrate_to_postgres.py
   ```

## Step 7: Verify Deployment

1. Check that all seven services are running (green status in Render dashboard)
2. Test each service by visiting its URL:
   - Mock API: `https://sentiment-mock-api.onrender.com/health`
   - Model API: `https://sentiment-model-api.onrender.com/health`
   - Main Dashboard: `https://sentiment-dashboard.onrender.com/`
   - Social Media Auth UI: `https://sentiment-social-auth.onrender.com/`
   - Social Media Data Visualization: `https://sentiment-social-data.onrender.com/`
   - CSV Data Upload: `https://sentiment-csv-upload.onrender.com/`

## Step 8: Set Up Navigation Links

Update the navigation links in each Streamlit application to point to the correct URLs:

1. In your Render dashboard, note the URLs of all deployed services
2. For each Streamlit app, add a `.streamlit/config.toml` file in its directory:

```toml
[browser]
serverAddress = "sentiment-dashboard.onrender.com"
serverPort = 443
```

3. Update navigation links in each app to use the full URLs of the deployed services

## Troubleshooting

- **Service Crashes**: Check the logs in the Render dashboard
- **Database Issues**: Verify the DATABASE_URL environment variable is correctly set
- **Connection Errors**: Check if services can communicate with each other
- **CSS/JS Loading Issues**: Ensure Streamlit base URL is configured correctly
- **Persistent Storage**: Note that Render's default disk storage is ephemeral; use the database for persistent data

## Maintenance

- Render automatically redeploys services when you push changes to your repository
- You can manually restart services from the Render dashboard
- Set up Render's built-in monitoring for each service
- For database backups, use Render's automatic backup feature for the PostgreSQL database 