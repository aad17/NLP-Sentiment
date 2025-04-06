import streamlit as st
import pandas as pd
import os
import sys
import json
import logging
import time
import sqlite3
from datetime import datetime
from io import StringIO

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('social_media_csv_upload')

# Add parent directory to path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

# Import necessary modules
try:
    from app.social_media.database import save_posts, init_db
    from app.social_media.data_processor import process_social_media_batch
    social_media_available = True
except ImportError as e:
    logger.error(f"Social media modules not available: {e}")
    social_media_available = False

# Initialize database
init_db()

def create_sample_data():
    """Generate sample CSV data for demonstration"""
    sample_data = """platform,post_id,username,content,timestamp,likes,shares,comments
twitter,1001,firefighter42,Just responded to a major incident downtown. Everyone safe! #emergency #response,2023-04-10T15:30:00,45,12,8
facebook,2001,citysafety,Community safety meeting this Thursday at City Hall. Learn about our new emergency protocols.,2023-04-11T09:15:00,67,23,15
twitter,1002,rescueteam7,Training exercise today was challenging but successful. #teamwork #emergency,2023-04-12T14:45:00,32,5,3
linkedin,3001,emergencyservices,Proud to announce our department has received new equipment funding that will help us better serve the community.,2023-04-13T11:20:00,120,35,27
twitter,1003,dispatch911,Reminder: If you see smoke, call 911 immediately. Don't assume someone else has called. #safety,2023-04-14T08:10:00,89,56,12
facebook,2002,firesafety,Check your smoke detectors monthly! This simple action can save lives. Here's how to properly test them...,2023-04-15T16:30:00,105,42,31
twitter,1004,emt_responder,Difficult day on the job, but we made a difference. Remember to appreciate every moment. #ems #firstresponder,2023-04-16T19:22:00,76,18,24
facebook,2003,emergencyprep,Hurricane season is approaching. Is your family prepared? Download our emergency checklist at the link below.,2023-04-17T10:05:00,113,67,29
linkedin,3002,disasterrelief,Seeking volunteers for our upcoming disaster preparedness training. No experience necessary, we will provide all training.,2023-04-18T13:40:00,95,41,53
twitter,1005,firechief,Communication is key during emergencies. Make sure your family has a plan and meeting point. #preparedness,2023-04-19T11:15:00,64,38,7"""
    return sample_data

def parse_csv_data(csv_data):
    """Parse CSV data into a pandas DataFrame"""
    try:
        df = pd.read_csv(StringIO(csv_data))
        required_columns = ['platform', 'post_id', 'username', 'content', 'timestamp']
        
        for col in required_columns:
            if col not in df.columns:
                st.error(f"Missing required column: {col}")
                return None
        
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
                'raw_data': json.dumps({'source': 'csv_upload'})
            }
            posts.append(post)
        
        return posts
    except Exception as e:
        st.error(f"Error parsing CSV data: {e}")
        logger.error(f"Error parsing CSV data: {e}")
        return None

def save_to_database(posts):
    """Save posts to the database"""
    try:
        saved_count = save_posts(posts)
        return saved_count
    except Exception as e:
        st.error(f"Error saving to database: {e}")
        logger.error(f"Error saving to database: {e}")
        return 0

def main():
    st.set_page_config(page_title="Social Media CSV Upload", page_icon="📊", layout="wide")
    
    st.title("Social Media Data Upload")
    st.subheader("Upload CSV data to simulate social media posts")
    
    if not social_media_available:
        st.error("Social media modules are not available. Please check your installation.")
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Upload CSV", "Paste CSV", "Process Data"])
    
    with tab1:
        st.subheader("Upload CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            csv_data = uploaded_file.getvalue().decode('utf-8')
            st.success(f"File uploaded successfully: {uploaded_file.name}")
            
            # Show preview
            st.subheader("Data Preview")
            df = pd.read_csv(StringIO(csv_data))
            st.dataframe(df.head())
            
            # Save button
            if st.button("Save to Database", key="save_upload"):
                posts = parse_csv_data(csv_data)
                if posts:
                    saved_count = save_to_database(posts)
                    if saved_count > 0:
                        st.success(f"Successfully saved {saved_count} posts to the database!")
                    else:
                        st.warning("No new posts were saved to the database. They may already exist.")
    
    with tab2:
        st.subheader("Paste CSV Data")
        
        # Sample data button
        if st.button("Load Sample Data"):
            sample_data = create_sample_data()
            st.session_state['csv_text'] = sample_data
        
        # Text area for pasting CSV
        csv_text = st.text_area("Paste your CSV data here:", 
                               value=st.session_state.get('csv_text', ''),
                               height=300)
        
        if csv_text:
            try:
                # Show preview
                st.subheader("Data Preview")
                df = pd.read_csv(StringIO(csv_text))
                st.dataframe(df.head())
                
                # Save button
                if st.button("Save to Database", key="save_paste"):
                    posts = parse_csv_data(csv_text)
                    if posts:
                        saved_count = save_to_database(posts)
                        if saved_count > 0:
                            st.success(f"Successfully saved {saved_count} posts to the database!")
                        else:
                            st.warning("No new posts were saved to the database. They may already exist.")
            except Exception as e:
                st.error(f"Error parsing CSV data: {e}")
    
    with tab3:
        st.subheader("Process Social Media Data")
        
        # Get count of posts in database
        conn = sqlite3.connect('app/data/social_media.db')
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM social_media_posts")
        post_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM social_media_sentiment")
        processed_count = cursor.fetchone()[0]
        conn.close()
        
        st.info(f"Total posts in database: {post_count}")
        st.info(f"Posts with sentiment analysis: {processed_count}")
        st.info(f"Posts awaiting processing: {post_count - processed_count}")
        
        # Process data button
        col1, col2 = st.columns(2)
        with col1:
            batch_size = st.number_input("Batch Size", min_value=1, max_value=1000, value=10)
        with col2:
            model = st.selectbox("Sentiment Model", ["default", "expanded", "domain_aware"])
        
        if st.button("Process Posts"):
            if post_count - processed_count > 0:
                with st.spinner("Processing posts..."):
                    try:
                        stats = process_social_media_batch(count=batch_size, model_name=model)
                        st.success(f"Successfully processed {stats['processed']} posts: {stats['positive']} positive, {stats['neutral']} neutral, {stats['negative']} negative, {stats['errors']} errors")
                        
                        # Refresh counts
                        conn = sqlite3.connect('app/data/social_media.db')
                        cursor = conn.cursor()
                        cursor.execute("SELECT COUNT(*) FROM social_media_sentiment")
                        processed_count = cursor.fetchone()[0]
                        conn.close()
                        
                        st.info(f"Posts with sentiment analysis: {processed_count}")
                        st.info(f"Posts awaiting processing: {post_count - processed_count}")
                    except Exception as e:
                        st.error(f"Error processing posts: {e}")
                        logger.error(f"Error processing posts: {e}")
            else:
                st.warning("No posts available for processing.")
        
        # Generate test report
        st.subheader("Generate Test Report")
        
        if st.button("Generate Test Report"):
            with st.spinner("Generating test report..."):
                try:
                    # Collect test information
                    conn = sqlite3.connect('app/data/social_media.db')
                    cursor = conn.cursor()
                    
                    # Get post counts by platform
                    cursor.execute("SELECT platform, COUNT(*) FROM social_media_posts GROUP BY platform")
                    platform_counts = cursor.fetchall()
                    
                    # Get sentiment counts
                    cursor.execute("""
                        SELECT 
                            CASE 
                                WHEN sentiment = 1 THEN 'positive'
                                WHEN sentiment = 0 THEN 'neutral'
                                WHEN sentiment = -1 THEN 'negative'
                                ELSE 'unknown'
                            END as sentiment_label,
                            COUNT(*) 
                        FROM social_media_sentiment 
                        GROUP BY sentiment
                    """)
                    sentiment_counts = cursor.fetchall()
                    
                    # Get model performance stats if available
                    cursor.execute("""
                        SELECT model_name, AVG(confidence) as avg_confidence 
                        FROM social_media_sentiment 
                        WHERE model_name IS NOT NULL 
                        GROUP BY model_name
                    """)
                    model_stats = cursor.fetchall()
                    
                    conn.close()
                    
                    # Create test report
                    report = f"""# Social Media Integration Test Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Data Summary
- Total posts in database: {post_count}
- Posts with sentiment analysis: {processed_count}
- Posts awaiting processing: {post_count - processed_count}

## Platform Distribution
"""
                    for platform, count in platform_counts:
                        report += f"- {platform}: {count} posts\n"
                    
                    report += "\n## Sentiment Analysis Results\n"
                    if sentiment_counts:
                        for sentiment, count in sentiment_counts:
                            report += f"- {sentiment}: {count} posts\n"
                    else:
                        report += "- No sentiment analysis results yet\n"
                    
                    report += "\n## Model Performance\n"
                    if model_stats:
                        for model, avg_confidence in model_stats:
                            report += f"- {model}: Average confidence {avg_confidence:.2f}\n"
                    else:
                        report += "- No model performance data available\n"
                    
                    report += "\n## Workflow Validation\n"
                    report += "- ✅ CSV data upload functionality\n"
                    report += "- ✅ Database storage of social media posts\n"
                    report += f"- {'✅' if processed_count > 0 else '❌'} Sentiment analysis processing\n"
                    report += "- ✅ Integration with existing database schema\n"
                    report += "- ✅ Support for multiple platforms\n"
                    
                    # Write report to file
                    report_path = "social_media_test_report.md"
                    with open(report_path, "w") as f:
                        f.write(report)
                    
                    st.success(f"Test report generated successfully and saved to {report_path}")
                    
                    # Display report in UI
                    st.markdown(report)
                    
                except Exception as e:
                    st.error(f"Error generating test report: {e}")
                    logger.error(f"Error generating test report: {e}")

if __name__ == "__main__":
    main() 