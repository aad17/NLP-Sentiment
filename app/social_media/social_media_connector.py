"""
Social Media Connector Module

This module handles connections to various social media platforms
and provides functionality to retrieve data for sentiment analysis.
"""
import os
import sys
import json
import logging
import requests
from pathlib import Path
from datetime import datetime, timedelta
import time
import random  # For mock data generation

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('social_media_connector')

# Constants
CREDENTIALS_DIR = Path(__file__).resolve().parent.parent / "data" / "credentials"
CREDENTIALS_FILE = CREDENTIALS_DIR / "social_media_credentials.json"

def load_credentials(platform=None):
    """
    Load credentials for the specified platform or all platforms.
    
    Args:
        platform (str, optional): Platform name to load credentials for
        
    Returns:
        dict: Credentials for the specified platform or all platforms
    """
    if not CREDENTIALS_FILE.exists():
        logger.warning(f"Credentials file does not exist: {CREDENTIALS_FILE}")
        return {} if platform is None else None
    
    try:
        with open(CREDENTIALS_FILE, 'r') as f:
            all_credentials = json.load(f)
        
        if platform is None:
            return all_credentials
        
        return all_credentials.get(platform)
    except Exception as e:
        logger.error(f"Error loading credentials: {str(e)}")
        return {} if platform is None else None

class SocialMediaConnector:
    """Base class for social media connectors"""
    
    def __init__(self, platform_name):
        """
        Initialize the connector.
        
        Args:
            platform_name (str): Name of the social media platform
        """
        self.platform_name = platform_name
        self.credentials = load_credentials(platform_name)
        self.is_connected = False
        
        if not self.credentials:
            logger.warning(f"No credentials found for {platform_name}")
        
    def connect(self):
        """
        Establish connection to the platform.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        raise NotImplementedError("Subclasses must implement connect()")
    
    def get_posts(self, count=10, since=None, until=None, **kwargs):
        """
        Retrieve posts from the platform.
        
        Args:
            count (int): Number of posts to retrieve
            since (datetime): Retrieve posts after this date
            until (datetime): Retrieve posts before this date
            **kwargs: Additional platform-specific parameters
            
        Returns:
            list: List of post dictionaries
        """
        raise NotImplementedError("Subclasses must implement get_posts()")
    
    def _format_post(self, post_data, **kwargs):
        """
        Format raw post data into a standardized format.
        
        Args:
            post_data: Raw post data from the platform API
            **kwargs: Additional formatting parameters
            
        Returns:
            dict: Formatted post
        """
        raise NotImplementedError("Subclasses must implement _format_post()")

class TwitterConnector(SocialMediaConnector):
    """Connector for Twitter API"""
    
    def __init__(self):
        """Initialize the Twitter connector"""
        super().__init__("twitter")
        self.api = None
    
    def connect(self):
        """
        Establish connection to Twitter API.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        if not self.credentials:
            logger.error("Twitter credentials not found")
            return False
        
        try:
            # In a real implementation, you would use the Twitter API library (e.g., tweepy)
            # For now, we'll simulate a connection
            
            # Validate required credentials
            required_keys = ["api_key", "api_secret", "access_token", "access_token_secret"]
            if not all(key in self.credentials for key in required_keys):
                missing = [key for key in required_keys if key not in self.credentials]
                logger.error(f"Missing Twitter credentials: {', '.join(missing)}")
                return False
            
            # Simulate API connection
            logger.info("Connected to Twitter API")
            self.is_connected = True
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to Twitter API: {str(e)}")
            self.is_connected = False
            return False
    
    def get_posts(self, count=10, since=None, until=None, search_term=None, **kwargs):
        """
        Retrieve tweets from Twitter API.
        
        Args:
            count (int): Number of tweets to retrieve
            since (datetime): Retrieve tweets after this date
            until (datetime): Retrieve tweets before this date
            search_term (str): Search term to filter tweets
            **kwargs: Additional Twitter-specific parameters
            
        Returns:
            list: List of tweet dictionaries
        """
        if not self.is_connected and not self.connect():
            return []
        
        try:
            # In a real implementation, you would use the Twitter API
            # For now, we'll generate mock data
            
            # Mock data generation
            tweets = []
            for _ in range(count):
                tweet = self._generate_mock_tweet(search_term)
                tweets.append(self._format_post(tweet))
            
            logger.info(f"Retrieved {len(tweets)} tweets")
            return tweets
            
        except Exception as e:
            logger.error(f"Error retrieving tweets: {str(e)}")
            return []
    
    def _format_post(self, post_data, **kwargs):
        """
        Format raw tweet data into a standardized format.
        
        Args:
            post_data: Raw tweet data
            **kwargs: Additional formatting parameters
            
        Returns:
            dict: Formatted tweet
        """
        return {
            "post_id": post_data.get("id"),
            "platform": "twitter",
            "content": post_data.get("text"),
            "author": post_data.get("username"),
            "timestamp": post_data.get("created_at"),
            "likes": post_data.get("likes", 0),
            "retweets": post_data.get("retweets", 0),
            "url": f"https://twitter.com/{post_data.get('username')}/status/{post_data.get('id')}",
            "raw_data": post_data
        }
    
    def _generate_mock_tweet(self, search_term=None):
        """
        Generate a mock tweet for testing.
        
        Args:
            search_term (str): Search term to include in the tweet
            
        Returns:
            dict: Mock tweet data
        """
        tweet_id = f"{random.randint(1000000000000000000, 9999999999999999999)}"
        usernames = ["fireservice", "emergency_alerts", "fire_rescue", "police_updates", "ambulance_service"]
        username = random.choice(usernames)
        
        # Generate content based on search term
        if search_term:
            content = f"Important update about {search_term}! Please be aware of the situation. #emergency #alert"
        else:
            messages = [
                "Firefighters responding to a building fire on Main Street. Please avoid the area. #emergency",
                "Road closed due to accident. Seek alternative routes. #traffic #alert",
                "Weather warning: Heavy rain expected in the next 24 hours. Possible flooding in low areas. #weather",
                "Community meeting about emergency preparedness scheduled for tomorrow at 6 PM. All welcome! #community",
                "Reminder: Test your smoke alarms today. They save lives. #safety #tips"
            ]
            content = random.choice(messages)
        
        # Generate random timestamp within the last week
        days_ago = random.randint(0, 7)
        hours_ago = random.randint(0, 23)
        minutes_ago = random.randint(0, 59)
        created_at = (datetime.now() - timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)).isoformat()
        
        return {
            "id": tweet_id,
            "text": content,
            "username": username,
            "created_at": created_at,
            "likes": random.randint(0, 500),
            "retweets": random.randint(0, 100)
        }

class FacebookConnector(SocialMediaConnector):
    """Connector for Facebook API"""
    
    def __init__(self):
        """Initialize the Facebook connector"""
        super().__init__("facebook")
        self.api = None
    
    def connect(self):
        """
        Establish connection to Facebook API.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        if not self.credentials:
            logger.error("Facebook credentials not found")
            return False
        
        try:
            # In a real implementation, you would use the Facebook API library
            # For now, we'll simulate a connection
            
            # Validate required credentials
            required_keys = ["app_id", "app_secret", "access_token"]
            if not all(key in self.credentials for key in required_keys):
                missing = [key for key in required_keys if key not in self.credentials]
                logger.error(f"Missing Facebook credentials: {', '.join(missing)}")
                return False
            
            # Simulate API connection
            logger.info("Connected to Facebook API")
            self.is_connected = True
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to Facebook API: {str(e)}")
            self.is_connected = False
            return False
    
    def get_posts(self, count=10, since=None, until=None, page_id=None, **kwargs):
        """
        Retrieve posts from Facebook API.
        
        Args:
            count (int): Number of posts to retrieve
            since (datetime): Retrieve posts after this date
            until (datetime): Retrieve posts before this date
            page_id (str): Facebook page ID to retrieve posts from
            **kwargs: Additional Facebook-specific parameters
            
        Returns:
            list: List of post dictionaries
        """
        if not self.is_connected and not self.connect():
            return []
        
        try:
            # Use page_id from credentials if not provided
            page_id = page_id or self.credentials.get("page_id")
            
            if not page_id:
                logger.warning("No page ID provided")
            
            # In a real implementation, you would use the Facebook API
            # For now, we'll generate mock data
            
            # Mock data generation
            posts = []
            for _ in range(count):
                post = self._generate_mock_post(page_id)
                posts.append(self._format_post(post))
            
            logger.info(f"Retrieved {len(posts)} Facebook posts")
            return posts
            
        except Exception as e:
            logger.error(f"Error retrieving Facebook posts: {str(e)}")
            return []
    
    def _format_post(self, post_data, **kwargs):
        """
        Format raw Facebook post data into a standardized format.
        
        Args:
            post_data: Raw post data
            **kwargs: Additional formatting parameters
            
        Returns:
            dict: Formatted post
        """
        return {
            "post_id": post_data.get("id"),
            "platform": "facebook",
            "content": post_data.get("message"),
            "author": post_data.get("page_name"),
            "timestamp": post_data.get("created_time"),
            "likes": post_data.get("likes", 0),
            "shares": post_data.get("shares", 0),
            "comments": post_data.get("comments", 0),
            "url": f"https://facebook.com/{post_data.get('id')}",
            "raw_data": post_data
        }
    
    def _generate_mock_post(self, page_id=None):
        """
        Generate a mock Facebook post for testing.
        
        Args:
            page_id (str): Facebook page ID
            
        Returns:
            dict: Mock post data
        """
        post_id = f"{page_id or 'page'}_post_{random.randint(1000000000, 9999999999)}"
        page_names = ["City Fire Department", "Emergency Services", "Police Department", "Community Safety"]
        page_name = random.choice(page_names)
        
        messages = [
            "COMMUNITY ALERT: Fire crews responding to a structure fire near downtown. Please avoid the area to allow emergency vehicles to pass.",
            "Weather update: Severe thunderstorm warning in effect until 8 PM. Take necessary precautions and stay indoors if possible.",
            "Safety reminder: As temperatures drop, please check on elderly neighbors and ensure heating systems are functioning properly.",
            "UPDATE: The road closure on Highway 101 has been cleared. Traffic flow has returned to normal. Thank you for your patience.",
            "Join us this Saturday for our annual Fire Safety Open House! Fun activities for kids and important safety information for everyone. Event runs from 10 AM to 3 PM."
        ]
        
        # Generate random timestamp within the last week
        days_ago = random.randint(0, 7)
        hours_ago = random.randint(0, 23)
        created_time = (datetime.now() - timedelta(days=days_ago, hours=hours_ago)).isoformat()
        
        return {
            "id": post_id,
            "message": random.choice(messages),
            "page_name": page_name,
            "created_time": created_time,
            "likes": random.randint(5, 2000),
            "shares": random.randint(0, 200),
            "comments": random.randint(0, 50)
        }

class LinkedInConnector(SocialMediaConnector):
    """Connector for LinkedIn API"""
    
    def __init__(self):
        """Initialize the LinkedIn connector"""
        super().__init__("linkedin")
        self.api = None
    
    def connect(self):
        """
        Establish connection to LinkedIn API.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        if not self.credentials:
            logger.error("LinkedIn credentials not found")
            return False
        
        try:
            # Validate required credentials
            required_keys = ["client_id", "client_secret", "access_token"]
            if not all(key in self.credentials for key in required_keys):
                missing = [key for key in required_keys if key not in self.credentials]
                logger.error(f"Missing LinkedIn credentials: {', '.join(missing)}")
                return False
            
            # Simulate API connection
            logger.info("Connected to LinkedIn API")
            self.is_connected = True
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to LinkedIn API: {str(e)}")
            self.is_connected = False
            return False
    
    def get_posts(self, count=10, since=None, until=None, **kwargs):
        """
        Retrieve posts from LinkedIn API.
        
        Args:
            count (int): Number of posts to retrieve
            since (datetime): Retrieve posts after this date
            until (datetime): Retrieve posts before this date
            **kwargs: Additional LinkedIn-specific parameters
            
        Returns:
            list: List of post dictionaries
        """
        if not self.is_connected and not self.connect():
            return []
        
        try:
            # In a real implementation, you would use the LinkedIn API
            # For now, we'll generate mock data
            
            # Mock data generation
            posts = []
            for _ in range(count):
                post = self._generate_mock_post()
                posts.append(self._format_post(post))
            
            logger.info(f"Retrieved {len(posts)} LinkedIn posts")
            return posts
            
        except Exception as e:
            logger.error(f"Error retrieving LinkedIn posts: {str(e)}")
            return []
    
    def _format_post(self, post_data, **kwargs):
        """
        Format raw LinkedIn post data into a standardized format.
        
        Args:
            post_data: Raw post data
            **kwargs: Additional formatting parameters
            
        Returns:
            dict: Formatted post
        """
        return {
            "post_id": post_data.get("id"),
            "platform": "linkedin",
            "content": post_data.get("text"),
            "author": post_data.get("author_name"),
            "author_title": post_data.get("author_title"),
            "timestamp": post_data.get("created_at"),
            "likes": post_data.get("likes_count", 0),
            "comments": post_data.get("comments_count", 0),
            "url": f"https://linkedin.com/feed/update/{post_data.get('id')}",
            "raw_data": post_data
        }
    
    def _generate_mock_post(self):
        """
        Generate a mock LinkedIn post for testing.
        
        Returns:
            dict: Mock post data
        """
        post_id = f"linkedin_post_{random.randint(1000000000, 9999999999)}"
        
        authors = [
            {"name": "John Smith", "title": "Fire Chief at City Fire Department"},
            {"name": "Sarah Johnson", "title": "Emergency Services Coordinator"},
            {"name": "Robert Williams", "title": "Director of Public Safety"},
            {"name": "Emily Davis", "title": "Community Relations Manager at Emergency Response Team"}
        ]
        
        author = random.choice(authors)
        
        messages = [
            "Proud to announce our department's new emergency response protocol that has reduced response time by 15%. #EmergencyServices #Innovation",
            "Yesterday, our team conducted a joint training exercise with neighboring departments. Communication and coordination are key to effective emergency response.",
            "Excited to share that our community safety workshops have reached over 5,000 residents this year. Prevention is always better than response!",
            "Honored to receive the Distinguished Service Award for our department's work during last year's wildfire season. #Grateful #TeamEffort",
            "New study shows the effectiveness of our public awareness campaigns. The number of preventable incidents has decreased by 23% over the past year. #DataDriven #PublicSafety"
        ]
        
        # Generate random timestamp within the last week
        days_ago = random.randint(0, 14)
        hours_ago = random.randint(0, 23)
        created_at = (datetime.now() - timedelta(days=days_ago, hours=hours_ago)).isoformat()
        
        return {
            "id": post_id,
            "text": random.choice(messages),
            "author_name": author["name"],
            "author_title": author["title"],
            "created_at": created_at,
            "likes_count": random.randint(10, 500),
            "comments_count": random.randint(0, 50)
        }

class CustomPlatformConnector(SocialMediaConnector):
    """Connector for custom platforms"""
    
    def __init__(self, platform_name):
        """
        Initialize the custom platform connector.
        
        Args:
            platform_name (str): Name of the custom platform
        """
        super().__init__(platform_name)
    
    def connect(self):
        """
        Establish connection to the custom platform API.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        if not self.credentials:
            logger.error(f"{self.platform_name} credentials not found")
            return False
        
        try:
            # Check for minimal credentials
            if not self.credentials.get("api_key"):
                logger.error(f"API key is required for {self.platform_name}")
                return False
            
            # Simulate API connection
            logger.info(f"Connected to {self.platform_name} API")
            self.is_connected = True
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to {self.platform_name} API: {str(e)}")
            self.is_connected = False
            return False
    
    def get_posts(self, count=10, **kwargs):
        """
        Retrieve posts from the custom platform API.
        
        Args:
            count (int): Number of posts to retrieve
            **kwargs: Additional platform-specific parameters
            
        Returns:
            list: List of post dictionaries
        """
        if not self.is_connected and not self.connect():
            return []
        
        try:
            # In a real implementation, you would use the specific API
            # For now, we'll generate mock data
            
            # Mock data generation
            posts = []
            for _ in range(count):
                post = self._generate_mock_post()
                posts.append(self._format_post(post))
            
            logger.info(f"Retrieved {len(posts)} posts from {self.platform_name}")
            return posts
            
        except Exception as e:
            logger.error(f"Error retrieving posts from {self.platform_name}: {str(e)}")
            return []
    
    def _format_post(self, post_data, **kwargs):
        """
        Format raw post data into a standardized format.
        
        Args:
            post_data: Raw post data
            **kwargs: Additional formatting parameters
            
        Returns:
            dict: Formatted post
        """
        return {
            "post_id": post_data.get("id"),
            "platform": self.platform_name,
            "content": post_data.get("content"),
            "author": post_data.get("author"),
            "timestamp": post_data.get("created_at"),
            "engagement": post_data.get("engagement", 0),
            "url": post_data.get("url", ""),
            "raw_data": post_data
        }
    
    def _generate_mock_post(self):
        """
        Generate a mock post for testing.
        
        Returns:
            dict: Mock post data
        """
        post_id = f"post_{random.randint(1000000000, 9999999999)}"
        
        authors = ["Emergency Alert System", "Community Watch", "Public Safety Office", "Alert Network"]
        
        messages = [
            "Alert: Minor flooding reported in downtown area. Exercise caution when driving.",
            "Update on yesterday's incident: All affected residents have been contacted and support services provided.",
            "Reminder: Community emergency preparedness meeting tonight at 7 PM at the Community Center.",
            "Important: Road closure on Highway 20 due to fallen trees. Detour information available at [link]",
            "Safety Tip: Keep emergency supplies in your car during winter months including blankets and water."
        ]
        
        # Generate random timestamp within the last week
        days_ago = random.randint(0, 10)
        hours_ago = random.randint(0, 23)
        created_at = (datetime.now() - timedelta(days=days_ago, hours=hours_ago)).isoformat()
        
        return {
            "id": post_id,
            "content": random.choice(messages),
            "author": random.choice(authors),
            "created_at": created_at,
            "engagement": random.randint(5, 300),
            "url": f"https://{self.platform_name}.example.com/post/{post_id}"
        }

def get_connector(platform):
    """
    Get the appropriate connector for the specified platform.
    
    Args:
        platform (str): Platform name
        
    Returns:
        SocialMediaConnector: Connector instance
    """
    if platform == "twitter":
        return TwitterConnector()
    elif platform == "facebook":
        return FacebookConnector()
    elif platform == "linkedin":
        return LinkedInConnector()
    else:
        return CustomPlatformConnector(platform)

def get_all_connectors():
    """
    Get connectors for all platforms with saved credentials.
    
    Returns:
        dict: Dictionary of platform names to connector instances
    """
    credentials = load_credentials()
    connectors = {}
    
    for platform in credentials.keys():
        connectors[platform] = get_connector(platform)
    
    return connectors

def get_posts_from_all_platforms(count_per_platform=10, **kwargs):
    """
    Retrieve posts from all configured platforms.
    
    Args:
        count_per_platform (int): Number of posts to retrieve per platform
        **kwargs: Additional parameters to pass to get_posts()
        
    Returns:
        list: List of posts from all platforms
    """
    connectors = get_all_connectors()
    all_posts = []
    
    for platform, connector in connectors.items():
        try:
            platform_posts = connector.get_posts(count=count_per_platform, **kwargs)
            all_posts.extend(platform_posts)
            logger.info(f"Retrieved {len(platform_posts)} posts from {platform}")
        except Exception as e:
            logger.error(f"Error retrieving posts from {platform}: {str(e)}")
    
    # Sort by timestamp (newest first)
    all_posts.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    return all_posts 