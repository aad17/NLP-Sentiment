"""
Social Media Data Processor

This module handles batch processing of social media data,
integrating with the existing sentiment analysis models.
"""
import os
import sys
import logging
import time
import json
import requests
from datetime import datetime, timedelta
import threading
import schedule
import sqlite3
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import project modules
from app.utils.config import MOCK_API_PORT
from app.social_media.social_media_connector import get_posts_from_all_platforms, get_connector
from app.social_media.database import save_posts, get_posts, save_sentiment, save_model_performance, row_to_dict

# Import model prediction
try:
    from app.sentiment.predict import predict_sentiment
    MODEL_PREDICT_AVAILABLE = True
except ImportError:
    MODEL_PREDICT_AVAILABLE = False
    logging.warning("Could not import sentiment prediction module. Using mock implementation.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('social_media_processor')

# Constants
API_BASE_URL = f"http://localhost:{MOCK_API_PORT}"
DEFAULT_MODEL = "domain_aware"
BATCH_SIZE = 50
SCHEDULER_ACTIVE = False

# Database path
DB_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DB_DIR / "social_media.db"

def fetch_social_media_data(count=100, platform=None):
    """
    Fetch data from social media platforms.
    
    Args:
        count (int): Number of posts to fetch per platform
        platform (str, optional): Specific platform to fetch from
        
    Returns:
        list: List of fetched posts
    """
    try:
        if platform:
            # Import here to avoid circular import
            connector = get_connector(platform)
            return connector.get_posts(count=count)
        else:
            return get_posts_from_all_platforms(count_per_platform=count)
    except Exception as e:
        logger.error(f"Error fetching social media data: {str(e)}")
        return []

def process_social_media_batch(posts=None, count=BATCH_SIZE, model_name=DEFAULT_MODEL):
    """
    Process a batch of social media posts.
    
    Args:
        posts (list, optional): List of posts to process
        count (int): Number of posts to retrieve if posts not provided
        model_name (str): Name of the model to use
        
    Returns:
        dict: Processing statistics
    """
    stats = {
        "processed": 0,
        "with_sentiment": 0,
        "positive": 0,
        "neutral": 0,
        "negative": 0,
        "errors": 0
    }
    
    # Fetch posts from database if not provided
    if posts is None:
        # Retrieve posts that don't have sentiment analysis yet
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = """
        SELECT p.id as db_id, p.post_id, p.platform, p.content, p.author, p.timestamp 
        FROM social_media_posts p
        LEFT JOIN social_media_sentiment s ON p.id = s.post_id
        WHERE s.id IS NULL
        LIMIT ?
        """
        
        cursor.execute(query, (count,))
        posts = []
        for row in cursor.fetchall():
            post_dict = row_to_dict(row)
            posts.append(post_dict)
        
        conn.close()
    
    if not posts:
        logger.warning("No posts available for processing")
        return stats
    
    logger.info(f"Processing {len(posts)} posts with model: {model_name}")
    
    # Process each post
    for post in posts:
        try:
            # Check if post is a tuple or a dict
            if isinstance(post, tuple) or hasattr(post, 'keys') == False:
                # Convert tuple to dictionary
                logger.info(f"Converting tuple to dictionary: {post}")
                post = row_to_dict(post)
            
            # Extract post content
            content = post.get('content', '')
            
            if not content:
                logger.warning(f"Empty content for post {post.get('post_id')}")
                stats["errors"] += 1
                continue
            
            # Get post ID either from db_id field or look it up
            post_id_db = post.get('db_id')
            
            if not post_id_db:
                # Look up the post ID using platform and post_id
                conn = sqlite3.connect(DB_PATH)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id FROM social_media_posts WHERE platform = ? AND post_id = ?",
                    (post["platform"], post["post_id"])
                )
                result = cursor.fetchone()
                result_dict = row_to_dict(result)
                conn.close()
                
                if result_dict:
                    post_id_db = result_dict["id"]
                else:
                    logger.warning(f"Could not find post in database: {post.get('post_id')}")
                    stats["errors"] += 1
                    continue
            
            # Predict sentiment
            if MODEL_PREDICT_AVAILABLE:
                result = predict_sentiment(content, model_type=model_name)
                sentiment = result.get('sentiment')
                confidence = result.get('confidence')
            else:
                # Mock implementation
                sentiment = mock_predict_sentiment(content)
                confidence = 0.85  # Mock confidence score
            
            # Save sentiment to database
            success = save_sentiment(post_id_db, sentiment, confidence, model_name)
            
            if success:
                stats["with_sentiment"] += 1
                if sentiment == 1:
                    stats["positive"] += 1
                elif sentiment == 0:
                    stats["neutral"] += 1
                elif sentiment == -1:
                    stats["negative"] += 1
            else:
                stats["errors"] += 1
            
            stats["processed"] += 1
            
        except Exception as e:
            logger.error(f"Error processing post {post.get('post_id')}: {str(e)}")
            stats["errors"] += 1
    
    logger.info(f"Processed {stats['processed']} posts: {stats['positive']} positive, {stats['neutral']} neutral, {stats['negative']} negative, {stats['errors']} errors")
    return stats

def mock_predict_sentiment(text):
    """
    Mock sentiment prediction for testing.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        int: Sentiment score (-1, 0, 1)
    """
    # Simple keyword-based sentiment analysis
    positive_words = ['good', 'great', 'excellent', 'happy', 'positive', 'safe', 'success', 'thank']
    negative_words = ['bad', 'terrible', 'sad', 'negative', 'danger', 'warning', 'emergency', 'alert', 'avoid', 'fire']
    
    text = text.lower()
    
    # Count occurrences of positive and negative words
    positive_count = sum(1 for word in positive_words if word in text)
    negative_count = sum(1 for word in negative_words if word in text)
    
    # Determine sentiment
    if negative_count > positive_count:
        return -1  # Negative
    elif positive_count > negative_count:
        return 1   # Positive
    else:
        return 0   # Neutral

def fetch_and_save_social_media_data(count=100, platform=None):
    """
    Fetch data from social media platforms and save to the database.
    
    Args:
        count (int): Number of posts to fetch per platform
        platform (str, optional): Specific platform to fetch from
        
    Returns:
        int: Number of posts saved
    """
    try:
        # Fetch posts
        posts = fetch_social_media_data(count=count, platform=platform)
        
        if not posts:
            logger.warning(f"No posts retrieved from social media")
            return 0
        
        # Save posts to database
        saved_count = save_posts(posts)
        logger.info(f"Saved {saved_count} new posts to database")
        
        return saved_count
    except Exception as e:
        logger.error(f"Error in fetch and save: {str(e)}")
        return 0

def run_batch_job(job_type="all", count=100, model_name=DEFAULT_MODEL):
    """
    Run a batch job to fetch and process social media data.
    
    Args:
        job_type (str): Type of job to run (fetch, process, all)
        count (int): Number of items to process
        model_name (str): Model to use for sentiment analysis
        
    Returns:
        dict: Job results
    """
    results = {
        "job_type": job_type,
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
        "fetch_count": 0,
        "process_stats": {}
    }
    
    try:
        # Fetch data
        if job_type in ["fetch", "all"]:
            results["fetch_count"] = fetch_and_save_social_media_data(count=count)
            
        # Process data
        if job_type in ["process", "all"]:
            results["process_stats"] = process_social_media_batch(count=count, model_name=model_name)
            
        logger.info(f"Batch job completed: {job_type}")
        return results
    
    except Exception as e:
        logger.error(f"Error running batch job: {str(e)}")
        results["status"] = "error"
        results["error"] = str(e)
        return results

def _schedule_worker():
    """Worker function for the scheduler thread"""
    while SCHEDULER_ACTIVE:
        schedule.run_pending()
        time.sleep(1)

def start_scheduler():
    """Start the scheduler for regular batch jobs"""
    global SCHEDULER_ACTIVE
    
    if SCHEDULER_ACTIVE:
        logger.warning("Scheduler already running")
        return False
    
    try:
        # Schedule jobs
        # Fetch new data every 6 hours
        schedule.every(6).hours.do(run_batch_job, job_type="fetch", count=100)
        
        # Process data every hour
        schedule.every(1).hours.do(run_batch_job, job_type="process", count=200)
        
        # Run full job once a day
        schedule.every().day.at("02:00").do(run_batch_job, job_type="all", count=500)
        
        # Start scheduler in a separate thread
        SCHEDULER_ACTIVE = True
        scheduler_thread = threading.Thread(target=_schedule_worker)
        scheduler_thread.daemon = True
        scheduler_thread.start()
        
        logger.info("Batch job scheduler started")
        return True
    
    except Exception as e:
        logger.error(f"Error starting scheduler: {str(e)}")
        SCHEDULER_ACTIVE = False
        return False

def stop_scheduler():
    """Stop the scheduler"""
    global SCHEDULER_ACTIVE
    SCHEDULER_ACTIVE = False
    logger.info("Batch job scheduler stopped")
    return True

def get_scheduler_status():
    """Get the status of the scheduler"""
    return {
        "active": SCHEDULER_ACTIVE,
        "jobs": [str(job) for job in schedule.get_jobs()]
    }

# API integration functions

def get_incidents_from_api(count=10):
    """
    Fetch incidents from the mock API.
    
    Args:
        count (int): Number of incidents to fetch
        
    Returns:
        list: List of incidents
    """
    try:
        response = requests.get(f"{API_BASE_URL}/get_incidents", params={"count": count})
        response.raise_for_status()
        
        incidents = response.json()
        logger.info(f"Retrieved {len(incidents)} incidents from API")
        return incidents
    
    except Exception as e:
        logger.error(f"Error retrieving incidents from API: {str(e)}")
        return []

def analyze_incidents_with_social_context(count=10, model_name=DEFAULT_MODEL):
    """
    Analyze incidents with social media context.
    
    Args:
        count (int): Number of incidents to analyze
        model_name (str): Model to use for sentiment analysis
        
    Returns:
        list: List of enriched incidents with social context
    """
    try:
        # Get incidents from API
        incidents = get_incidents_from_api(count=count)
        
        if not incidents:
            return []
        
        # Process each incident
        enriched_incidents = []
        for incident in incidents:
            # Extract incident text
            report_text = incident.get('report', '')
            
            if not report_text:
                continue
            
            # Get keywords from the report
            keywords = extract_keywords(report_text)
            
            # Find related social media posts
            social_posts = find_related_posts(keywords, limit=5)
            
            # Analyze sentiment of the incident
            if MODEL_PREDICT_AVAILABLE:
                sentiment_result = predict_sentiment(report_text, model_type=model_name)
                sentiment = sentiment_result.get('sentiment')
                confidence = sentiment_result.get('confidence')
            else:
                sentiment = mock_predict_sentiment(report_text)
                confidence = 0.85
            
            # Add sentiment and social context to the incident
            enriched_incident = {
                **incident,
                "sentiment": sentiment,
                "sentiment_confidence": confidence,
                "social_context": {
                    "keywords": keywords,
                    "related_posts": social_posts,
                    "social_sentiment": analyze_social_sentiment(social_posts)
                }
            }
            
            enriched_incidents.append(enriched_incident)
        
        logger.info(f"Enriched {len(enriched_incidents)} incidents with social context")
        return enriched_incidents
    
    except Exception as e:
        logger.error(f"Error analyzing incidents with social context: {str(e)}")
        return []

def extract_keywords(text, max_keywords=5):
    """
    Extract keywords from text.
    
    Args:
        text (str): Text to extract keywords from
        max_keywords (int): Maximum number of keywords to extract
        
    Returns:
        list: List of keywords
    """
    # This is a simple implementation
    # In a real system, you would use NLP techniques like TF-IDF, etc.
    
    # Common emergency keywords
    emergency_keywords = [
        "fire", "accident", "flood", "storm", "earthquake", "rescue",
        "emergency", "evacuation", "injury", "damage", "police", "ambulance",
        "hazard", "alert", "warning", "disaster", "victim", "trapped", "smoke",
        "explosion", "collapse", "traffic", "crash", "danger", "safety"
    ]
    
    # Simple keyword extraction
    text_lower = text.lower()
    found_keywords = []
    
    for keyword in emergency_keywords:
        if keyword in text_lower:
            found_keywords.append(keyword)
            if len(found_keywords) >= max_keywords:
                break
    
    return found_keywords

def find_related_posts(keywords, limit=5):
    """
    Find social media posts related to keywords.
    
    Args:
        keywords (list): List of keywords to search for
        limit (int): Maximum number of posts to return
        
    Returns:
        list: List of related posts
    """
    try:
        if not keywords:
            return []
        
        # Get posts from database
        posts = get_posts(limit=100)
        
        # Filter posts by keywords
        related_posts = []
        for post in posts:
            content = post.get('content', '').lower()
            
            # Check if any keyword is in the content
            if any(keyword in content for keyword in keywords):
                related_posts.append(post)
                
                if len(related_posts) >= limit:
                    break
        
        return related_posts
    
    except Exception as e:
        logger.error(f"Error finding related posts: {str(e)}")
        return []

def analyze_social_sentiment(posts):
    """
    Analyze sentiment distribution in a set of social media posts.
    
    Args:
        posts (list): List of posts with sentiment
        
    Returns:
        dict: Sentiment distribution
    """
    # Initialize counters
    stats = {
        "total": len(posts),
        "positive": 0,
        "neutral": 0,
        "negative": 0,
        "unknown": 0
    }
    
    # Count sentiments
    for post in posts:
        sentiment = post.get('sentiment')
        
        if sentiment == 1:
            stats["positive"] += 1
        elif sentiment == 0:
            stats["neutral"] += 1
        elif sentiment == -1:
            stats["negative"] += 1
        else:
            stats["unknown"] += 1
    
    # Add percentages
    if stats["total"] > 0:
        stats["positive_pct"] = stats["positive"] / stats["total"] * 100
        stats["neutral_pct"] = stats["neutral"] / stats["total"] * 100
        stats["negative_pct"] = stats["negative"] / stats["total"] * 100
    else:
        stats["positive_pct"] = 0
        stats["neutral_pct"] = 0
        stats["negative_pct"] = 0
    
    return stats

# Automatically start scheduler if this module is run directly
if __name__ == "__main__":
    if not start_scheduler():
        logger.error("Failed to start scheduler")
        sys.exit(1)
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        stop_scheduler()
        logger.info("Scheduler stopped by user") 