"""
Database module for the Sentiment Analysis application.

This module handles database interactions for storing and retrieving incident reports.
"""
import os
import sqlite3
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('database')

# Get the database path
current_dir = Path(__file__).resolve().parent
DB_PATH = os.path.join(current_dir, 'incidents.db')

def get_db_connection():
    """
    Create a connection to the SQLite database.
    
    Returns:
        sqlite3.Connection: Database connection
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # This enables column access by name
    return conn

def init_db():
    """
    Initialize the database by creating necessary tables if they don't exist.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # Create incidents table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS incidents (
            id TEXT PRIMARY KEY,
            report TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create sentiment table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sentiment (
            incident_id TEXT PRIMARY KEY,
            sentiment INTEGER NOT NULL,
            confidence REAL,
            processed_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (incident_id) REFERENCES incidents (id)
        )
        ''')
        
        conn.commit()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        conn.rollback()
    finally:
        conn.close()

def save_incidents(incidents):
    """
    Save incident reports to the database.
    
    Args:
        incidents (list): List of incident dictionaries
        
    Returns:
        int: Number of incidents saved
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        count = 0
        
        for incident in incidents:
            # Check if incident already exists
            cursor.execute("SELECT id FROM incidents WHERE id = ?", (incident['incident_id'],))
            if cursor.fetchone() is None:
                # Insert new incident
                cursor.execute(
                    "INSERT INTO incidents (id, report, timestamp) VALUES (?, ?, ?)",
                    (incident['incident_id'], incident['report'], incident['timestamp'])
                )
                count += 1
        
        conn.commit()
        logger.info(f"Saved {count} new incidents to database")
        return count
    except Exception as e:
        logger.error(f"Error saving incidents: {str(e)}")
        conn.rollback()
        return 0
    finally:
        conn.close()

def get_incidents(limit=100, offset=0):
    """
    Retrieve incidents from the database.
    
    Args:
        limit (int): Maximum number of incidents to retrieve
        offset (int): Number of incidents to skip
        
    Returns:
        list: List of incident dictionaries
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT i.id, i.report, i.timestamp, s.sentiment
            FROM incidents i
            LEFT JOIN sentiment s ON i.id = s.incident_id
            ORDER BY i.timestamp DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset)
        )
        
        incidents = []
        for row in cursor.fetchall():
            incidents.append({
                'incident_id': row['id'],
                'report': row['report'],
                'timestamp': row['timestamp'],
                'sentiment': row['sentiment']
            })
        
        return incidents
    except Exception as e:
        logger.error(f"Error retrieving incidents: {str(e)}")
        return []
    finally:
        conn.close()

def save_sentiment(incident_id, sentiment, confidence=None):
    """
    Save sentiment analysis results for an incident.
    
    Args:
        incident_id (str): Incident ID
        sentiment (int): Sentiment score (-1, 0, 1)
        confidence (float, optional): Confidence score
        
    Returns:
        bool: True if successful, False otherwise
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # Check if sentiment already exists
        cursor.execute("SELECT incident_id FROM sentiment WHERE incident_id = ?", (incident_id,))
        if cursor.fetchone() is None:
            # Insert new sentiment
            cursor.execute(
                "INSERT INTO sentiment (incident_id, sentiment, confidence) VALUES (?, ?, ?)",
                (incident_id, sentiment, confidence)
            )
        else:
            # Update existing sentiment
            cursor.execute(
                "UPDATE sentiment SET sentiment = ?, confidence = ?, processed_at = CURRENT_TIMESTAMP WHERE incident_id = ?",
                (sentiment, confidence, incident_id)
            )
        
        conn.commit()
        logger.info(f"Saved sentiment for incident {incident_id}")
        return True
    except Exception as e:
        logger.error(f"Error saving sentiment: {str(e)}")
        conn.rollback()
        return False
    finally:
        conn.close()

def get_sentiment_stats():
    """
    Get statistics about sentiment distribution.
    
    Returns:
        dict: Dictionary with sentiment statistics
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT 
                COUNT(CASE WHEN sentiment = 1 THEN 1 END) as positive,
                COUNT(CASE WHEN sentiment = 0 THEN 1 END) as neutral,
                COUNT(CASE WHEN sentiment = -1 THEN 1 END) as negative,
                COUNT(*) as total
            FROM sentiment
            """
        )
        
        row = cursor.fetchone()
        stats = {
            'positive': row['positive'],
            'neutral': row['neutral'],
            'negative': row['negative'],
            'total': row['total']
        }
        
        return stats
    except Exception as e:
        logger.error(f"Error retrieving sentiment stats: {str(e)}")
        return {'positive': 0, 'neutral': 0, 'negative': 0, 'total': 0}
    finally:
        conn.close()

# Initialize the database when this module is imported
init_db() 