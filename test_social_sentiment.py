#!/usr/bin/env python3
"""
Simple test script to analyze a single post and store the sentiment
"""
import os
import sys
import sqlite3
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_social_sentiment')

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Database path
DB_DIR = Path(__file__).resolve().parent / "app" / "data"
DB_PATH = DB_DIR / "social_media.db"

def get_db_connection():
    """
    Create a connection to the SQLite database.
    
    Returns:
        sqlite3.Connection: Database connection
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # This enables column access by name
    return conn

def get_one_post():
    """Get one post from the database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get a post that doesn't have sentiment analysis yet
    query = """
    SELECT p.id as db_id, p.post_id, p.platform, p.content, p.author, p.timestamp 
    FROM social_media_posts p
    LEFT JOIN social_media_sentiment s ON p.id = s.post_id
    WHERE s.id IS NULL
    LIMIT 1
    """
    
    cursor.execute(query)
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            "db_id": row["db_id"],
            "post_id": row["post_id"],
            "platform": row["platform"],
            "content": row["content"],
            "author": row["author"],
            "timestamp": row["timestamp"]
        }
    return None

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

def save_sentiment(post_id_db, sentiment, confidence=None, model_name=None):
    """
    Save sentiment analysis results for a post.
    
    Args:
        post_id_db (int): Database ID of the post
        sentiment (int): Sentiment score (-1, 0, 1)
        confidence (float, optional): Confidence score
        model_name (str, optional): Name of the model used
        
    Returns:
        bool: True if successful, False otherwise
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # Check if sentiment already exists for this post
        cursor.execute(
            "SELECT id FROM social_media_sentiment WHERE post_id = ?",
            (post_id_db,)
        )
        
        existing_sentiment = cursor.fetchone()
        if existing_sentiment:
            # Update existing sentiment
            cursor.execute(
                """
                UPDATE social_media_sentiment 
                SET sentiment = ?, confidence = ?, model_name = ?, processed_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (sentiment, confidence, model_name, existing_sentiment["id"])
            )
        else:
            # Insert new sentiment
            cursor.execute(
                """
                INSERT INTO social_media_sentiment (post_id, sentiment, confidence, model_name)
                VALUES (?, ?, ?, ?)
                """,
                (post_id_db, sentiment, confidence, model_name)
            )
        
        conn.commit()
        logger.info(f"Saved sentiment for post {post_id_db}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving sentiment: {str(e)}")
        conn.rollback()
        return False
    finally:
        conn.close()

def get_unprocessed_posts(limit=10):
    """Get posts that don't have sentiment analysis yet"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get posts that don't have sentiment analysis yet
    query = """
    SELECT p.id as db_id, p.post_id, p.platform, p.content, p.author, p.timestamp 
    FROM social_media_posts p
    LEFT JOIN social_media_sentiment s ON p.id = s.post_id
    WHERE s.id IS NULL
    LIMIT ?
    """
    
    cursor.execute(query, (limit,))
    
    posts = []
    for row in cursor.fetchall():
        posts.append({
            "db_id": row["db_id"],
            "post_id": row["post_id"],
            "platform": row["platform"],
            "content": row["content"],
            "author": row["author"],
            "timestamp": row["timestamp"]
        })
    
    conn.close()
    return posts

def main():
    """Main function"""
    logger.info(f"DB_PATH={DB_PATH}, exists={DB_PATH.exists()}")
    
    # Process multiple posts
    count = 29  # Process 29 remaining posts
    posts = get_unprocessed_posts(count)
    
    logger.info(f"Found {len(posts)} posts to analyze")
    
    if not posts:
        logger.error("No posts found to analyze")
        return
    
    # Process each post
    for i, post in enumerate(posts):
        logger.info(f"Processing post {i+1}/{len(posts)}: {post['post_id']} from {post['platform']}")
        
        # Analyze sentiment
        sentiment = mock_predict_sentiment(post['content'])
        confidence = 0.85  # Mock confidence
        
        logger.info(f"Sentiment: {sentiment}, Confidence: {confidence}")
        
        # Save sentiment
        success = save_sentiment(post['db_id'], sentiment, confidence, "mock")
        
        if success:
            logger.info("Successfully saved sentiment analysis")
        else:
            logger.error("Failed to save sentiment analysis")
    
    logger.info(f"Completed sentiment analysis for {len(posts)} posts")

if __name__ == "__main__":
    main() 