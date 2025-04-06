"""
Mock API for Fire Brigade Incident Reports

This module provides a Flask API that returns random incident reports
with realistic content. It simulates a real API that would be used
to fetch incidents from the fire brigade's system.
"""
import os
import sys
import logging
from flask import Flask, jsonify, request
import time
import csv
from io import StringIO
import pandas as pd
import json

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the mock data generator
from app.utils.mock_data_generator import generate_incidents
from app.utils.config import MOCK_API_PORT

# Import social media processor for integration
try:
    from app.social_media.data_processor import (
        get_incidents_from_api, analyze_incidents_with_social_context,
        fetch_and_save_social_media_data, process_social_media_batch,
        run_batch_job, start_scheduler, stop_scheduler, get_scheduler_status
    )
    SOCIAL_MEDIA_AVAILABLE = True
except ImportError:
    SOCIAL_MEDIA_AVAILABLE = False
    logging.warning("Social media processor not available. Social media endpoints will be disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mock_api')

# Create Flask application
app = Flask(__name__)

@app.route('/get_incidents', methods=['GET'])
def get_incidents():
    """
    Endpoint to fetch incident reports.
    
    Query Parameters:
        count (int): Number of incidents to return (default: 5)
        
    Returns:
        JSON: List of incident reports
    """
    try:
        # Get count parameter with default value
        count = request.args.get('count', default=5, type=int)
        
        # Validate count parameter
        if count < 1:
            return jsonify({"error": "Count must be at least 1"}), 400
        if count > 100:
            return jsonify({"error": "Count cannot exceed 100"}), 400
        
        # Generate random incidents
        incidents = generate_incidents(count)
        
        # Add a small delay to simulate network latency
        time.sleep(0.1)
        
        logger.info(f"Generated {count} incidents")
        return jsonify(incidents)
    
    except Exception as e:
        logger.error(f"Error generating incidents: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Simple health check endpoint.
    
    Returns:
        JSON: Status of the API
    """
    return jsonify({"status": "healthy"})

@app.route('/social_media/fetch', methods=['POST'])
def fetch_social_media():
    """
    Endpoint to fetch social media data.
    
    Query Parameters:
        count (int): Number of posts to fetch per platform (default: 100)
        platform (str, optional): Specific platform to fetch from
        
    Returns:
        JSON: Result of the fetch operation
    """
    if not SOCIAL_MEDIA_AVAILABLE:
        return jsonify({"error": "Social media functionality is not available"}), 501
    
    try:
        count = request.args.get('count', default=100, type=int)
        platform = request.args.get('platform', default=None)
        
        # Validate parameters
        if count < 1:
            return jsonify({"error": "Count must be at least 1"}), 400
        if count > 500:
            return jsonify({"error": "Count cannot exceed 500"}), 400
        
        # Fetch and save social media data
        saved_count = fetch_and_save_social_media_data(count=count, platform=platform)
        
        return jsonify({
            "status": "success",
            "saved_count": saved_count,
            "platform": platform or "all"
        })
    
    except Exception as e:
        logger.error(f"Error fetching social media data: {str(e)}")
        return jsonify({"error": f"Error fetching social media data: {str(e)}"}), 500

@app.route('/social_media/process', methods=['POST'])
def process_social_media():
    """
    Endpoint to process social media data with sentiment analysis.
    
    Query Parameters:
        count (int): Number of posts to process (default: 50)
        model (str): Model to use for sentiment analysis (default: domain_aware)
        
    Returns:
        JSON: Processing statistics
    """
    if not SOCIAL_MEDIA_AVAILABLE:
        return jsonify({"error": "Social media functionality is not available"}), 501
    
    try:
        count = request.args.get('count', default=50, type=int)
        model = request.args.get('model', default="domain_aware")
        
        # Validate parameters
        if count < 1:
            return jsonify({"error": "Count must be at least 1"}), 400
        if count > 1000:
            return jsonify({"error": "Count cannot exceed 1000"}), 400
        
        # Process social media data
        stats = process_social_media_batch(count=count, model_name=model)
        
        return jsonify({
            "status": "success",
            "model": model,
            "stats": stats
        })
    
    except Exception as e:
        logger.error(f"Error processing social media data: {str(e)}")
        return jsonify({"error": f"Error processing social media data: {str(e)}"}), 500

@app.route('/social_media/batch_job', methods=['POST'])
def start_batch_job():
    """
    Endpoint to run a social media batch job.
    
    Query Parameters:
        job_type (str): Type of job to run (fetch, process, all)
        count (int): Number of items to process (default: 100)
        model (str): Model to use for sentiment analysis (default: domain_aware)
        
    Returns:
        JSON: Job results
    """
    if not SOCIAL_MEDIA_AVAILABLE:
        return jsonify({"error": "Social media functionality is not available"}), 501
    
    try:
        job_type = request.args.get('job_type', default="all")
        count = request.args.get('count', default=100, type=int)
        model = request.args.get('model', default="domain_aware")
        
        # Validate parameters
        if job_type not in ["fetch", "process", "all"]:
            return jsonify({"error": "Invalid job type"}), 400
        
        if count < 1:
            return jsonify({"error": "Count must be at least 1"}), 400
        if count > 1000:
            return jsonify({"error": "Count cannot exceed 1000"}), 400
        
        # Run batch job
        results = run_batch_job(job_type=job_type, count=count, model_name=model)
        
        return jsonify({
            "status": "success",
            "job_type": job_type,
            "results": results
        })
    
    except Exception as e:
        logger.error(f"Error running batch job: {str(e)}")
        return jsonify({"error": f"Error running batch job: {str(e)}"}), 500

@app.route('/social_media/scheduler/start', methods=['POST'])
def start_scheduler_endpoint():
    """
    Endpoint to start the scheduler for regular batch jobs.
    
    Returns:
        JSON: Result of the operation
    """
    if not SOCIAL_MEDIA_AVAILABLE:
        return jsonify({"error": "Social media functionality is not available"}), 501
    
    try:
        success = start_scheduler()
        
        if success:
            return jsonify({
                "status": "success",
                "message": "Scheduler started successfully",
                "scheduler_status": get_scheduler_status()
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to start scheduler"
            }), 500
    
    except Exception as e:
        logger.error(f"Error starting scheduler: {str(e)}")
        return jsonify({"error": f"Error starting scheduler: {str(e)}"}), 500

@app.route('/social_media/scheduler/stop', methods=['POST'])
def stop_scheduler_endpoint():
    """
    Endpoint to stop the scheduler.
    
    Returns:
        JSON: Result of the operation
    """
    if not SOCIAL_MEDIA_AVAILABLE:
        return jsonify({"error": "Social media functionality is not available"}), 501
    
    try:
        success = stop_scheduler()
        
        return jsonify({
            "status": "success",
            "message": "Scheduler stopped successfully"
        })
    
    except Exception as e:
        logger.error(f"Error stopping scheduler: {str(e)}")
        return jsonify({"error": f"Error stopping scheduler: {str(e)}"}), 500

@app.route('/social_media/scheduler/status', methods=['GET'])
def scheduler_status_endpoint():
    """
    Endpoint to get the status of the scheduler.
    
    Returns:
        JSON: Scheduler status
    """
    if not SOCIAL_MEDIA_AVAILABLE:
        return jsonify({"error": "Social media functionality is not available"}), 501
    
    try:
        status = get_scheduler_status()
        
        return jsonify({
            "status": "success",
            "scheduler_status": status
        })
    
    except Exception as e:
        logger.error(f"Error getting scheduler status: {str(e)}")
        return jsonify({"error": f"Error getting scheduler status: {str(e)}"}), 500

@app.route('/social_media/incidents', methods=['GET'])
def get_enriched_incidents():
    """
    Endpoint to fetch incidents enriched with social media context.
    
    Query Parameters:
        count (int): Number of incidents to return (default: 5)
        model (str): Model to use for sentiment analysis (default: domain_aware)
        
    Returns:
        JSON: List of enriched incident reports with social context
    """
    if not SOCIAL_MEDIA_AVAILABLE:
        return jsonify({"error": "Social media functionality is not available"}), 501
    
    try:
        count = request.args.get('count', default=5, type=int)
        model = request.args.get('model', default="domain_aware")
        
        # Validate parameters
        if count < 1:
            return jsonify({"error": "Count must be at least 1"}), 400
        if count > 50:
            return jsonify({"error": "Count cannot exceed 50"}), 400
        
        # Get enriched incidents
        enriched_incidents = analyze_incidents_with_social_context(count=count, model_name=model)
        
        return jsonify(enriched_incidents)
    
    except Exception as e:
        logger.error(f"Error getting enriched incidents: {str(e)}")
        return jsonify({"error": f"Error getting enriched incidents: {str(e)}"}), 500

@app.route('/social_media/mock_data', methods=['POST'])
def upload_mock_data():
    """
    Upload mock social media data from CSV
    ---
    tags:
      - Social Media
    consumes:
      - text/csv
    parameters:
      - name: csv_data
        in: body
        required: true
        type: string
        description: CSV data with social media posts
    responses:
      200:
        description: Data processed successfully
      400:
        description: Invalid format or missing required columns
    """
    if not SOCIAL_MEDIA_AVAILABLE:
        return jsonify({"status": "error", "message": "Social media module is not available"}), 400

    try:
        # Get CSV data from request
        csv_data = request.data.decode('utf-8')
        
        # Parse CSV data
        df = pd.read_csv(StringIO(csv_data))
        required_columns = ['platform', 'post_id', 'username', 'content', 'timestamp']
        
        for col in required_columns:
            if col not in df.columns:
                return jsonify({"status": "error", "message": f"Missing required column: {col}"}), 400
        
        # Add missing columns with default values if needed
        if 'likes' not in df.columns:
            df['likes'] = 0
        if 'shares' not in df.columns:
            df['shares'] = 0
        if 'comments' not in df.columns:
            df['comments'] = 0
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Prepare data for database
        posts = []
        for _, row in df.iterrows():
            post = {
                'platform': row['platform'],
                'post_id': str(row['post_id']),
                'author': row['username'],  # Map username to author field
                'content': row['content'],
                'timestamp': row['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'url': f"https://example.com/{row['platform']}/{row['post_id']}",  # Generate a mock URL
                'likes': int(row['likes']),
                'shares': int(row['shares']),
                'comments': int(row['comments']),
                'raw_data': json.dumps({'source': 'mock_api_upload'})
            }
            posts.append(post)
        
        # Save posts to database
        from app.social_media.database import save_posts
        saved_count = save_posts(posts)
        
        # Return success response
        return jsonify({
            "status": "success", 
            "message": f"Successfully processed CSV data", 
            "posts_saved": saved_count
        })
    
    except Exception as e:
        logger.error(f"Error processing CSV data: {e}")
        return jsonify({"status": "error", "message": f"Error processing CSV data: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('MOCK_API_PORT', MOCK_API_PORT))
    logger.info(f"Starting mock API on port {port}")
    
    # Start the scheduler if social media functionality is available
    if SOCIAL_MEDIA_AVAILABLE:
        start_scheduler()
        logger.info("Social media scheduler started")
    
    app.run(host='0.0.0.0', port=port) 