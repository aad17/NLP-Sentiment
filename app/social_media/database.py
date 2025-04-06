"""
Social Media Database Module

This module handles database operations for storing social media posts
and credentials for sentiment analysis.
"""
import os
import sys
import json
import logging
import sqlite3
from pathlib import Path
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('social_media_db')

# Constants
DB_DIR = Path(__file__).resolve().parent.parent / "data"
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

def row_to_dict(row):
    """
    Convert a database row (tuple, sqlite3.Row, etc.) to a dictionary.
    
    Args:
        row: Database result row
        
    Returns:
        dict: Dictionary representation of the row
    """
    if row is None:
        return None
        
    # If it's already a dict
    if isinstance(row, dict):
        return row
        
    # If it's a sqlite3.Row
    if isinstance(row, sqlite3.Row):
        return dict(row)
        
    # If it's a tuple or list
    if isinstance(row, (tuple, list)):
        # Try to get column names from cursor description
        # If not available, use generic column names
        return {f"col{i}": value for i, value in enumerate(row)}
    
    # If it's some other type, try converting to dict
    try:
        return dict(row)
    except (TypeError, ValueError):
        logger.warning(f"Unable to convert {type(row)} to dictionary")
        return {}

def init_db():
    """
    Initialize the database by creating necessary tables if they don't exist.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # Create social_media_accounts table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS social_media_accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            platform TEXT NOT NULL,
            account_name TEXT NOT NULL,
            account_id TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            last_updated TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(platform, account_id)
        )
        ''')
        
        # Create social_media_posts table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS social_media_posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id TEXT NOT NULL,
            platform TEXT NOT NULL,
            account_id INTEGER,
            content TEXT NOT NULL,
            author TEXT,
            timestamp TEXT NOT NULL,
            url TEXT,
            engagement_count INTEGER DEFAULT 0,
            raw_data TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(platform, post_id),
            FOREIGN KEY(account_id) REFERENCES social_media_accounts(id)
        )
        ''')
        
        # Create social_media_sentiment table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS social_media_sentiment (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id INTEGER NOT NULL,
            sentiment INTEGER NOT NULL,
            confidence REAL,
            model_name TEXT,
            processed_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(post_id) REFERENCES social_media_posts(id)
        )
        ''')
        
        # Create table for model performance tracking
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            dataset_type TEXT NOT NULL,
            accuracy REAL,
            precision REAL,
            recall REAL,
            f1_score REAL,
            confusion_matrix TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        logger.info("Social media database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        conn.rollback()
    finally:
        conn.close()

def save_account(platform, account_name, account_id=None):
    """
    Save a social media account to the database.
    
    Args:
        platform (str): Platform name (e.g., twitter, facebook)
        account_name (str): Display name of the account
        account_id (str, optional): Platform-specific account ID
        
    Returns:
        int: Account ID in the database
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # Check if account already exists
        cursor.execute(
            "SELECT id FROM social_media_accounts WHERE platform = ? AND account_id = ?",
            (platform, account_id)
        )
        
        existing_account = cursor.fetchone()
        if existing_account:
            # Update existing account
            cursor.execute(
                """
                UPDATE social_media_accounts
                SET account_name = ?, last_updated = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (account_name, existing_account["id"])
            )
            account_id_db = existing_account["id"]
        else:
            # Insert new account
            cursor.execute(
                """
                INSERT INTO social_media_accounts (platform, account_name, account_id)
                VALUES (?, ?, ?)
                """,
                (platform, account_name, account_id)
            )
            account_id_db = cursor.lastrowid
        
        conn.commit()
        logger.info(f"Saved account {account_name} for platform {platform}")
        return account_id_db
    
    except Exception as e:
        logger.error(f"Error saving account: {str(e)}")
        conn.rollback()
        return None
    finally:
        conn.close()

def get_account_id(platform, account_id):
    """
    Get the database ID for a social media account.
    
    Args:
        platform (str): Platform name
        account_id (str): Platform-specific account ID
        
    Returns:
        int: Account ID in the database
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM social_media_accounts WHERE platform = ? AND account_id = ?",
            (platform, account_id)
        )
        account = cursor.fetchone()
        account_dict = row_to_dict(account)
        return account_dict["id"] if account_dict else None
    
    except Exception as e:
        logger.error(f"Error getting account ID: {str(e)}")
        return None
    finally:
        conn.close()

def save_post(post):
    """
    Save a social media post to the database.
    
    Args:
        post (dict): Post data
            - post_id: Platform-specific post ID
            - platform: Platform name
            - content: Post content/text
            - author: Author name
            - timestamp: Post timestamp
            - url: URL to the post
            - [optional] raw_data: Raw API response
            - [optional] likes, shares, comments, etc.
        
    Returns:
        int: Post ID in the database
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # Check if post already exists
        cursor.execute(
            "SELECT id FROM social_media_posts WHERE platform = ? AND post_id = ?",
            (post["platform"], post["post_id"])
        )
        
        existing_post = cursor.fetchone()
        if existing_post:
            # Post already exists, return its ID
            return existing_post["id"]
        
        # Calculate engagement count based on available metrics
        engagement_count = 0
        if "likes" in post:
            engagement_count += post.get("likes", 0)
        if "retweets" in post:
            engagement_count += post.get("retweets", 0)
        if "shares" in post:
            engagement_count += post.get("shares", 0)
        if "comments" in post:
            engagement_count += post.get("comments", 0)
        if "engagement" in post:
            engagement_count += post.get("engagement", 0)
        
        # Convert raw_data to JSON if it's a dict
        raw_data = json.dumps(post.get("raw_data")) if post.get("raw_data") else None
        
        # Insert new post
        cursor.execute(
            """
            INSERT INTO social_media_posts 
            (post_id, platform, content, author, timestamp, url, engagement_count, raw_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                post["post_id"], post["platform"], post["content"], post["author"],
                post["timestamp"], post.get("url", ""), engagement_count, raw_data
            )
        )
        
        post_id_db = cursor.lastrowid
        conn.commit()
        logger.info(f"Saved post {post['post_id']} from {post['platform']}")
        return post_id_db
    
    except Exception as e:
        logger.error(f"Error saving post: {str(e)}")
        conn.rollback()
        return None
    finally:
        conn.close()

def save_posts(posts):
    """
    Save multiple social media posts to the database.
    
    Args:
        posts (list): List of post dictionaries
        
    Returns:
        int: Number of posts saved
    """
    saved_count = 0
    for post in posts:
        if save_post(post):
            saved_count += 1
    
    return saved_count

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
        existing_sentiment_dict = row_to_dict(existing_sentiment)
        
        if existing_sentiment_dict:
            # Update existing sentiment
            cursor.execute(
                """
                UPDATE social_media_sentiment 
                SET sentiment = ?, confidence = ?, model_name = ?, processed_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (sentiment, confidence, model_name, existing_sentiment_dict["id"])
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

def get_posts(platform=None, limit=100, offset=0, with_sentiment=True):
    """
    Retrieve posts from the database.
    
    Args:
        platform (str, optional): Filter by platform
        limit (int): Maximum number of posts to retrieve
        offset (int): Number of posts to skip
        with_sentiment (bool): Include sentiment data
        
    Returns:
        list: List of post dictionaries
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # Construct the query - use different JOIN based on with_sentiment flag
        base_query = """
        SELECT 
            p.id, 
            p.post_id, 
            p.platform, 
            p.content, 
            p.author, 
            p.timestamp, 
            p.url, 
            p.engagement_count
        """
        
        # Add sentiment columns if requested
        if with_sentiment:
            base_query += """,
            s.sentiment, 
            s.confidence, 
            s.model_name
            """
            
        base_query += """
        FROM social_media_posts p
        """
        
        # Use appropriate JOIN based on whether we want sentiment data
        if with_sentiment:
            # Use INNER JOIN to only include posts with sentiment data
            base_query += """
            INNER JOIN social_media_sentiment s ON p.id = s.post_id
            """
        
        # Add platform filter if specified
        where_clause = ""
        params = []
        if platform:
            where_clause = " WHERE p.platform = ? "
            params.append(platform)
        
        # Complete the query
        query = base_query + where_clause + """
        ORDER BY p.timestamp DESC
        LIMIT ? OFFSET ?
        """
        
        params.extend([limit, offset])
        
        # Log the query for debugging
        logger.info(f"Getting posts with query: {query}, params: {params}")
        
        # Execute query
        cursor.execute(query, params)
        
        # Process results
        posts = []
        for row in cursor.fetchall():
            row_dict = row_to_dict(row)
            
            # Create post dictionary with base fields
            post_dict = {
                "db_id": row_dict["id"],
                "post_id": row_dict["post_id"],
                "platform": row_dict["platform"],
                "content": row_dict["content"],
                "author": row_dict["author"],
                "timestamp": row_dict["timestamp"],
                "url": row_dict.get("url", ""),
                "engagement_count": row_dict.get("engagement_count", 0)
            }
            
            # Add sentiment data if available
            if with_sentiment:
                try:
                    sentiment_value = row_dict["sentiment"]
                    post_dict["sentiment"] = int(sentiment_value)
                    post_dict["sentiment_confidence"] = float(row_dict["confidence"]) if row_dict["confidence"] is not None else None
                    post_dict["sentiment_model"] = row_dict["model_name"]
                    
                    # Log for debugging
                    logger.debug(f"Added sentiment {sentiment_value} to post {post_dict['post_id']}")
                except (ValueError, TypeError, KeyError) as e:
                    logger.error(f"Error adding sentiment data: {str(e)}")
            
            posts.append(post_dict)
        
        # Log post count and sample for debugging
        logger.info(f"Retrieved {len(posts)} posts")
        if len(posts) > 0:
            sample_post = posts[0]
            has_sentiment = 'sentiment' in sample_post
            logger.info(f"Sample post has sentiment: {has_sentiment}, value: {sample_post.get('sentiment', 'N/A')}")
        
        return posts
    
    except Exception as e:
        logger.error(f"Error retrieving posts: {str(e)}")
        return []
    finally:
        conn.close()

def get_sentiment_stats(platform=None, days=30):
    """
    Get statistics about sentiment distribution for social media posts.
    
    Args:
        platform (str, optional): Filter by platform
        days (int): Number of days to look back
        
    Returns:
        dict: Dictionary with sentiment statistics
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # First check if we have any posts with sentiment data (without time filter)
        sentiment_query = """
        SELECT COUNT(*) as count 
        FROM social_media_posts p
        INNER JOIN social_media_sentiment s ON p.id = s.post_id
        """
        
        params = []
        
        # Add platform filter if specified
        if platform:
            sentiment_query += " WHERE p.platform = ? "
            params.append(platform)
            
        # Execute the query without time filter to check if any data exists at all
        cursor.execute(sentiment_query, params)
        sentiment_count = cursor.fetchone()["count"]
        
        if sentiment_count == 0:
            return {"positive": 0, "neutral": 0, "negative": 0, "total": 0}
        
        # Construct query to get sentiment statistics - use INNER JOIN to count only posts with sentiment
        query = """
        SELECT 
            COUNT(CASE WHEN s.sentiment = 1 THEN 1 END) as positive,
            COUNT(CASE WHEN s.sentiment = 0 THEN 1 END) as neutral,
            COUNT(CASE WHEN s.sentiment = -1 THEN 1 END) as negative,
            COUNT(p.id) as total
        FROM social_media_posts p
        INNER JOIN social_media_sentiment s ON p.id = s.post_id
        """
        
        params = []
        where_added = False
        
        # Add platform filter if specified
        if platform:
            query += " WHERE p.platform = ?"
            params.append(platform)
            where_added = True
        
        # Add time filter ONLY if days is reasonable and data exists within that timeframe
        if days > 0:
            # Check if we actually have data in the time range
            time_check_query = """
            SELECT COUNT(*) as count 
            FROM social_media_posts p
            INNER JOIN social_media_sentiment s ON p.id = s.post_id
            WHERE p.timestamp >= datetime('now', ? || ' days')
            """
            if platform:
                time_check_query += " AND p.platform = ?"
                time_check_params = [f"-{days}", platform]
            else:
                time_check_params = [f"-{days}"]
                
            cursor.execute(time_check_query, time_check_params)
            time_filtered_count = cursor.fetchone()["count"]
            
            # Only add time filter if we have data within this timeframe
            if time_filtered_count > 0:
                if where_added:
                    query += " AND"
                else:
                    query += " WHERE"
                    where_added = True
                    
                query += " p.timestamp >= datetime('now', ? || ' days')"
                params.append(f"-{days}")
        
        # Execute query
        cursor.execute(query, params)
        
        row = cursor.fetchone()
        row_dict = row_to_dict(row)
        
        stats = {
            "positive": row_dict.get("positive", 0) or 0,
            "neutral": row_dict.get("neutral", 0) or 0,
            "negative": row_dict.get("negative", 0) or 0,
            "total": row_dict.get("total", 0) or 0
        }
        
        # Log for debugging
        logger.info(f"Sentiment stats (final): {stats}")
        
        return stats
    
    except Exception as e:
        logger.error(f"Error retrieving sentiment stats: {str(e)}")
        return {"positive": 0, "neutral": 0, "negative": 0, "total": 0}
    finally:
        conn.close()

def get_sentiment_by_platform(days=30):
    """
    Get sentiment breakdown by platform.
    
    Args:
        days (int): Number of days to look back
        
    Returns:
        list: List of platform sentiment statistics
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # Check if we have any posts with sentiment data (without time filter)
        sentiment_query = """
        SELECT COUNT(*) as count 
        FROM social_media_posts p
        INNER JOIN social_media_sentiment s ON p.id = s.post_id
        """
            
        cursor.execute(sentiment_query)
        sentiment_count = cursor.fetchone()["count"]
        
        if sentiment_count == 0:
            return []
        
        # Get all platforms that have sentiment data (without time filter)
        platform_query = """
        SELECT DISTINCT p.platform 
        FROM social_media_posts p
        INNER JOIN social_media_sentiment s ON p.id = s.post_id
        """
            
        cursor.execute(platform_query)
        platforms = [row_to_dict(row)["platform"] for row in cursor.fetchall()]
        
        if not platforms:
            return []
            
        # Construct query with INNER JOIN to only include posts with sentiment
        query = """
        SELECT 
            p.platform,
            COUNT(CASE WHEN s.sentiment = 1 THEN 1 END) as positive,
            COUNT(CASE WHEN s.sentiment = 0 THEN 1 END) as neutral,
            COUNT(CASE WHEN s.sentiment = -1 THEN 1 END) as negative,
            COUNT(p.id) as total
        FROM social_media_posts p
        INNER JOIN social_media_sentiment s ON p.id = s.post_id
        """
        
        params = []
        
        # Add time filter ONLY if days is reasonable and data exists within that timeframe
        if days > 0:
            # Check if we actually have data in the time range
            time_check_query = """
            SELECT COUNT(*) as count 
            FROM social_media_posts p
            INNER JOIN social_media_sentiment s ON p.id = s.post_id
            WHERE p.timestamp >= datetime('now', ? || ' days')
            """
            cursor.execute(time_check_query, [f"-{days}"])
            time_filtered_count = cursor.fetchone()["count"]
            
            # Only add time filter if we have data within this timeframe
            if time_filtered_count > 0:
                query += " WHERE p.timestamp >= datetime('now', ? || ' days')"
                params.append(f"-{days}")
        
        # Group by platform
        query += " GROUP BY p.platform"
        
        # Execute query
        cursor.execute(query, params)
        
        results = []
        for row in cursor.fetchall():
            row_dict = row_to_dict(row)
            results.append({
                "platform": row_dict["platform"],
                "positive": row_dict.get("positive", 0) or 0,
                "neutral": row_dict.get("neutral", 0) or 0,
                "negative": row_dict.get("negative", 0) or 0,
                "total": row_dict.get("total", 0) or 0
            })
        
        # Log for debugging
        logger.info(f"Platform sentiment stats (final): {results}")
        
        return results
    
    except Exception as e:
        logger.error(f"Error retrieving platform sentiment: {str(e)}")
        return []
    finally:
        conn.close()

def save_model_performance(model_name, dataset_type, metrics):
    """
    Save model performance metrics to the database.
    
    Args:
        model_name (str): Name of the model
        dataset_type (str): Type of dataset (e.g., twitter, facebook)
        metrics (dict): Performance metrics
            - accuracy: Overall accuracy
            - precision: Precision score
            - recall: Recall score
            - f1_score: F1 score
            - confusion_matrix: Confusion matrix as JSON string
        
    Returns:
        bool: True if successful, False otherwise
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # Convert confusion matrix to JSON if it's not already a string
        confusion_matrix = metrics.get("confusion_matrix")
        if confusion_matrix and not isinstance(confusion_matrix, str):
            confusion_matrix = json.dumps(confusion_matrix)
        
        # Insert performance metrics
        cursor.execute(
            """
            INSERT INTO model_performance 
            (model_name, dataset_type, accuracy, precision, recall, f1_score, confusion_matrix)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_name, dataset_type, 
                metrics.get("accuracy"), metrics.get("precision"),
                metrics.get("recall"), metrics.get("f1_score"),
                confusion_matrix
            )
        )
        
        conn.commit()
        logger.info(f"Saved performance metrics for model {model_name} on {dataset_type}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving model performance: {str(e)}")
        conn.rollback()
        return False
    finally:
        conn.close()

def get_model_performance(model_name=None, limit=10):
    """
    Get model performance metrics from the database.
    
    Args:
        model_name (str, optional): Filter by model name
        limit (int): Maximum number of records to retrieve
        
    Returns:
        list: List of performance metric dictionaries
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        query = """
        SELECT 
            model_name, dataset_type, accuracy, precision, recall, 
            f1_score, confusion_matrix, timestamp
        FROM model_performance
        """
        
        params = []
        
        # Add model filter if specified
        if model_name:
            query += " WHERE model_name = ?"
            params.append(model_name)
        
        # Add ordering and limit
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        # Execute query
        cursor.execute(query, params)
        
        results = []
        for row in cursor.fetchall():
            # Parse confusion matrix if it's a JSON string
            confusion_matrix = row["confusion_matrix"]
            if confusion_matrix:
                try:
                    confusion_matrix = json.loads(confusion_matrix)
                except:
                    pass
            
            results.append({
                "model_name": row["model_name"],
                "dataset_type": row["dataset_type"],
                "accuracy": row["accuracy"],
                "precision": row["precision"],
                "recall": row["recall"],
                "f1_score": row["f1_score"],
                "confusion_matrix": confusion_matrix,
                "timestamp": row["timestamp"]
            })
        
        return results
    
    except Exception as e:
        logger.error(f"Error retrieving model performance: {str(e)}")
        return []
    finally:
        conn.close()

# Initialize the database when this module is imported
init_db() 