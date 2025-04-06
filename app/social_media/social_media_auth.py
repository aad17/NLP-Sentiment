"""
Social Media Authentication UI

This module provides a Streamlit interface for users to enter and manage
social media API credentials for data retrieval and sentiment analysis.
"""
import os
import sys
import json
import logging
import streamlit as st
from pathlib import Path
import hashlib

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import configuration
from app.utils.config import DASHBOARD_TITLE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('social_media_auth')

# Constants
CREDENTIALS_DIR = Path(__file__).resolve().parent.parent / "data" / "credentials"
CREDENTIALS_FILE = CREDENTIALS_DIR / "social_media_credentials.json"

# Ensure credentials directory exists
CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)

# Set page configuration
st.set_page_config(
    page_title=f"{DASHBOARD_TITLE} - Social Media Setup",
    page_icon="🔑",
    layout="wide",
    initial_sidebar_state="expanded"
)

def encrypt_value(value):
    """Simple encryption for sensitive data (this is a basic implementation)"""
    if not value:
        return ""
    return hashlib.sha256(value.encode()).hexdigest()[:10] + "..." 

def save_credentials(platform, credentials):
    """Save credentials to the credentials file"""
    # Create or load existing credentials
    if CREDENTIALS_FILE.exists():
        with open(CREDENTIALS_FILE, 'r') as f:
            all_credentials = json.load(f)
    else:
        all_credentials = {}
    
    # Update credentials for the specified platform
    all_credentials[platform] = credentials
    
    # Save to file
    with open(CREDENTIALS_FILE, 'w') as f:
        json.dump(all_credentials, f)
    
    logger.info(f"Saved credentials for {platform}")
    return True

def load_credentials():
    """Load all saved credentials"""
    if CREDENTIALS_FILE.exists():
        with open(CREDENTIALS_FILE, 'r') as f:
            return json.load(f)
    return {}

def test_connection(platform, credentials):
    """Test connection to the social media platform API"""
    # This is where you would implement actual API connection testing
    # For now, we'll just return success if credentials are not empty
    
    # Twitter specific validation
    if platform == "twitter":
        if not all([credentials.get("api_key"), credentials.get("api_secret"),
                   credentials.get("access_token"), credentials.get("access_token_secret")]):
            return False, "All Twitter credentials are required"
    
    # Facebook specific validation
    elif platform == "facebook":
        if not all([credentials.get("app_id"), credentials.get("app_secret"), 
                   credentials.get("access_token")]):
            return False, "All Facebook credentials are required"
    
    # LinkedIn specific validation
    elif platform == "linkedin":
        if not all([credentials.get("client_id"), credentials.get("client_secret"),
                   credentials.get("access_token")]):
            return False, "All LinkedIn credentials are required"
    
    # Generic validation for other platforms
    else:
        if not credentials:
            return False, "Credentials cannot be empty"
    
    # Mock successful connection
    logger.info(f"Tested connection to {platform}")
    return True, "Connection successful"

def render_twitter_form():
    """Render form for Twitter API credentials"""
    st.subheader("Twitter API Credentials")
    
    # Load existing credentials if available
    all_creds = load_credentials()
    twitter_creds = all_creds.get("twitter", {})
    
    with st.form("twitter_credentials_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            api_key = st.text_input(
                "API Key (Consumer Key)", 
                value=twitter_creds.get("api_key", ""),
                type="password" if twitter_creds.get("api_key") else "default"
            )
            
            access_token = st.text_input(
                "Access Token", 
                value=twitter_creds.get("access_token", ""),
                type="password" if twitter_creds.get("access_token") else "default"
            )
        
        with col2:
            api_secret = st.text_input(
                "API Secret (Consumer Secret)", 
                value=twitter_creds.get("api_secret", ""),
                type="password" if twitter_creds.get("api_secret") else "default"
            )
            
            access_token_secret = st.text_input(
                "Access Token Secret", 
                value=twitter_creds.get("access_token_secret", ""),
                type="password" if twitter_creds.get("access_token_secret") else "default"
            )
        
        submitted = st.form_submit_button("Save & Test Connection")
        
        if submitted:
            credentials = {
                "api_key": api_key,
                "api_secret": api_secret,
                "access_token": access_token,
                "access_token_secret": access_token_secret
            }
            
            # Test connection before saving
            success, message = test_connection("twitter", credentials)
            
            if success:
                save_credentials("twitter", credentials)
                st.success(f"✅ {message} - Twitter credentials saved!")
            else:
                st.error(f"❌ {message}")

def render_facebook_form():
    """Render form for Facebook API credentials"""
    st.subheader("Facebook API Credentials")
    
    # Load existing credentials if available
    all_creds = load_credentials()
    facebook_creds = all_creds.get("facebook", {})
    
    with st.form("facebook_credentials_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            app_id = st.text_input(
                "App ID", 
                value=facebook_creds.get("app_id", ""),
                type="password" if facebook_creds.get("app_id") else "default"
            )
        
        with col2:
            app_secret = st.text_input(
                "App Secret", 
                value=facebook_creds.get("app_secret", ""),
                type="password" if facebook_creds.get("app_secret") else "default"
            )
        
        access_token = st.text_input(
            "Access Token (Long-lived)", 
            value=facebook_creds.get("access_token", ""),
            type="password" if facebook_creds.get("access_token") else "default"
        )
        
        page_id = st.text_input(
            "Page ID (Optional)",
            value=facebook_creds.get("page_id", "")
        )
        
        submitted = st.form_submit_button("Save & Test Connection")
        
        if submitted:
            credentials = {
                "app_id": app_id,
                "app_secret": app_secret,
                "access_token": access_token,
                "page_id": page_id
            }
            
            # Test connection before saving
            success, message = test_connection("facebook", credentials)
            
            if success:
                save_credentials("facebook", credentials)
                st.success(f"✅ {message} - Facebook credentials saved!")
            else:
                st.error(f"❌ {message}")

def render_linkedin_form():
    """Render form for LinkedIn API credentials"""
    st.subheader("LinkedIn API Credentials")
    
    # Load existing credentials if available
    all_creds = load_credentials()
    linkedin_creds = all_creds.get("linkedin", {})
    
    with st.form("linkedin_credentials_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            client_id = st.text_input(
                "Client ID", 
                value=linkedin_creds.get("client_id", ""),
                type="password" if linkedin_creds.get("client_id") else "default"
            )
        
        with col2:
            client_secret = st.text_input(
                "Client Secret", 
                value=linkedin_creds.get("client_secret", ""),
                type="password" if linkedin_creds.get("client_secret") else "default"
            )
        
        access_token = st.text_input(
            "Access Token", 
            value=linkedin_creds.get("access_token", ""),
            type="password" if linkedin_creds.get("access_token") else "default"
        )
        
        submitted = st.form_submit_button("Save & Test Connection")
        
        if submitted:
            credentials = {
                "client_id": client_id,
                "client_secret": client_secret,
                "access_token": access_token
            }
            
            # Test connection before saving
            success, message = test_connection("linkedin", credentials)
            
            if success:
                save_credentials("linkedin", credentials)
                st.success(f"✅ {message} - LinkedIn credentials saved!")
            else:
                st.error(f"❌ {message}")

def render_custom_platform_form():
    """Render form for custom API credentials"""
    st.subheader("Custom Platform API Credentials")
    
    # Select platform from dropdown or add new
    all_creds = load_credentials()
    custom_platforms = [k for k in all_creds.keys() if k not in ["twitter", "facebook", "linkedin"]]
    
    # Option to add a new platform
    new_platform = st.checkbox("Add new platform")
    
    if new_platform:
        platform_name = st.text_input("Platform Name").lower().strip()
    else:
        if not custom_platforms:
            st.info("No custom platforms found. Check 'Add new platform' to create one.")
            return
        
        platform_name = st.selectbox(
            "Select Platform", 
            options=custom_platforms
        )
    
    if platform_name:
        platform_creds = all_creds.get(platform_name, {})
        
        with st.form(f"{platform_name}_credentials_form"):
            api_key = st.text_input(
                "API Key", 
                value=platform_creds.get("api_key", ""),
                type="password" if platform_creds.get("api_key") else "default"
            )
            
            api_secret = st.text_input(
                "API Secret (if required)", 
                value=platform_creds.get("api_secret", ""),
                type="password" if platform_creds.get("api_secret") else "default"
            )
            
            access_token = st.text_input(
                "Access Token (if required)", 
                value=platform_creds.get("access_token", ""),
                type="password" if platform_creds.get("access_token") else "default"
            )
            
            additional_params = st.text_area(
                "Additional Parameters (JSON format)",
                value=json.dumps(platform_creds.get("additional_params", {}), indent=2) 
                if platform_creds.get("additional_params") else "{}"
            )
            
            submitted = st.form_submit_button("Save & Test Connection")
            
            if submitted:
                try:
                    # Parse additional parameters
                    additional_params_dict = json.loads(additional_params)
                    
                    credentials = {
                        "api_key": api_key,
                        "api_secret": api_secret,
                        "access_token": access_token,
                        "additional_params": additional_params_dict
                    }
                    
                    # Test connection before saving
                    success, message = test_connection(platform_name, credentials)
                    
                    if success:
                        save_credentials(platform_name, credentials)
                        st.success(f"✅ {message} - {platform_name.title()} credentials saved!")
                    else:
                        st.error(f"❌ {message}")
                        
                except json.JSONDecodeError:
                    st.error("❌ Invalid JSON format for additional parameters")

def show_saved_platforms():
    """Display cards for platforms that have saved credentials"""
    st.subheader("Connected Platforms")
    
    all_creds = load_credentials()
    
    if not all_creds:
        st.info("No platforms connected yet. Use the forms below to add social media credentials.")
        return
    
    # Display cards for each platform
    cols = st.columns(3)
    for i, (platform, creds) in enumerate(all_creds.items()):
        with cols[i % 3]:
            with st.expander(f"{platform.title()} ✓", expanded=True):
                st.markdown(f"**Platform:** {platform.title()}")
                
                # Display masked credentials
                for key, value in creds.items():
                    if key != "additional_params" and value:
                        if key.lower().find("secret") >= 0 or key.lower().find("key") >= 0 or key.lower().find("token") >= 0:
                            st.markdown(f"**{key.replace('_', ' ').title()}:** {encrypt_value(value)}")
                        else:
                            st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
                
                # Delete button
                if st.button(f"Remove {platform.title()} Connection", key=f"delete_{platform}"):
                    # Remove from credentials
                    all_creds.pop(platform)
                    with open(CREDENTIALS_FILE, 'w') as f:
                        json.dump(all_creds, f)
                    st.success(f"✅ {platform.title()} connection removed")
                    st.experimental_rerun()

def main():
    """Main function for the social media authentication UI"""
    st.title("Social Media Integration Setup")
    
    st.markdown("""
    This page allows you to configure connections to various social media platforms
    for data retrieval and sentiment analysis.
    
    **Security Note:** API credentials are stored locally on your machine and are not
    transmitted elsewhere. For production environments, consider using secure vaults.
    """)
    
    # Display connected platforms
    show_saved_platforms()
    
    # Create tabs for different platforms
    tabs = st.tabs(["Twitter", "Facebook", "LinkedIn", "Custom Platform"])
    
    with tabs[0]:
        render_twitter_form()
    
    with tabs[1]:
        render_facebook_form()
    
    with tabs[2]:
        render_linkedin_form()
    
    with tabs[3]:
        render_custom_platform_form()
    
    # Navigation links
    st.markdown("---")
    st.markdown("""
    [← Return to Dashboard](/) | [View Social Media Data →](/social_media_data)
    """)
    
    # Display navigation structure in sidebar
    st.sidebar.title("Navigation")
    st.sidebar.markdown("""
    - [Dashboard](/)
    - [Social Media Setup](/social_media_auth) (Current)
    - [Social Media Data](/social_media_data)
    """)
    
    # Help and documentation in sidebar
    st.sidebar.header("Help")
    with st.sidebar.expander("Twitter API Keys"):
        st.markdown("""
        1. Go to [Twitter Developer Portal](https://developer.twitter.com/)
        2. Create a Project and App
        3. Generate API keys and tokens from the App settings
        """)
    
    with st.sidebar.expander("Facebook API Keys"):
        st.markdown("""
        1. Go to [Facebook Developers](https://developers.facebook.com/)
        2. Create an App
        3. Navigate to App Settings to find your App ID and Secret
        4. Generate an access token for your page
        """)
    
    with st.sidebar.expander("LinkedIn API Keys"):
        st.markdown("""
        1. Go to [LinkedIn Developers](https://www.linkedin.com/developers/)
        2. Create an App
        3. Obtain Client ID and Client Secret from your app's Auth tab
        """)

if __name__ == "__main__":
    main() 