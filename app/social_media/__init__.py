"""
Social Media Integration Package

This package provides integration with social media platforms for sentiment analysis.
"""

__version__ = "1.0.0"

# Export key classes and functions for easier imports
from app.social_media.social_media_connector import (
    SocialMediaConnector, TwitterConnector, FacebookConnector, 
    LinkedInConnector, CustomPlatformConnector,
    get_connector, get_all_connectors, get_posts_from_all_platforms
)

from app.social_media.database import (
    init_db, save_account, get_account_id, save_post, save_posts,
    save_sentiment, get_posts, get_sentiment_stats, get_sentiment_by_platform,
    save_model_performance, get_model_performance
)

from app.social_media.data_processor import (
    fetch_social_media_data, process_social_media_batch,
    fetch_and_save_social_media_data, run_batch_job,
    start_scheduler, stop_scheduler, get_scheduler_status,
    analyze_incidents_with_social_context
) 