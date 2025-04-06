"""
Sentiment Analysis Dashboard

This module implements a Streamlit dashboard for visualizing
sentiment analysis of fire brigade incident reports.
Supports multiple sentiment models.
"""
import os
import sys
import logging
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
from datetime import datetime, timedelta
import requests
import calendar

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.dashboard.data_service import (
    get_incidents_from_db, get_sentiment_statistics,
    predict_sentiment, get_sentiment_over_time, get_recent_incidents,
    get_available_models, compare_models, MODEL_API_URL, get_sentiment_by_day_of_month
)
from app.utils.config import DASHBOARD_TITLE, REFRESH_INTERVAL

# Import active learning functionality
try:
    from active_learning_framework import (
        init_database, store_expert_feedback, load_models, store_prediction_for_feedback
    )
    ACTIVE_LEARNING_AVAILABLE = True
except ImportError:
    ACTIVE_LEARNING_AVAILABLE = False
    logging.warning("Active Learning framework not available. Feedback functionality will be limited.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dashboard')

# Set page configuration
st.set_page_config(
    page_title=DASHBOARD_TITLE,
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for model selection
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "domain_aware"

if 'show_comparison' not in st.session_state:
    st.session_state.show_comparison = False

if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False

# Initialize the database for active learning
if ACTIVE_LEARNING_AVAILABLE:
    try:
        db_engine = init_database()
        sentiment_vectorizer, sentiment_model, domain_vectorizer, domain_model, domain_binarizer = load_models()
        ACTIVE_LEARNING_INITIALIZED = True
    except Exception as e:
        logger.error(f"Error initializing active learning: {str(e)}")
        ACTIVE_LEARNING_INITIALIZED = False
else:
    ACTIVE_LEARNING_INITIALIZED = False

def main():
    """Main function to run the Streamlit dashboard."""
    # Page title
    st.title(DASHBOARD_TITLE)
    
    # Sidebar
    st.sidebar.header("Dashboard Controls")
    
    # Add auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh dashboard", value=False)
    refresh_interval = st.sidebar.slider(
        "Refresh interval (seconds)", 
        min_value=30, 
        max_value=300, 
        value=REFRESH_INTERVAL
    )
    
    if auto_refresh:
        st.sidebar.info(f"Dashboard will refresh every {refresh_interval} seconds")
        st.sidebar.warning("Note: Auto-refresh may interrupt feedback submission")
    
    # Time range selector
    time_range = st.sidebar.slider(
        "Time range (days)",
        min_value=1,
        max_value=90,
        value=30
    )
    
    # Month selector for daily chart
    current_month = datetime.now().month
    current_year = datetime.now().year
    
    months = [(i, calendar.month_name[i]) for i in range(1, 13)]
    selected_month_idx = st.sidebar.selectbox(
        "Select Month for Daily Chart",
        options=range(len(months)),
        format_func=lambda x: months[x][1],
        index=current_month-1
    )
    selected_month = months[selected_month_idx][0]
    
    years = list(range(current_year-2, current_year+1))
    selected_year = st.sidebar.selectbox(
        "Select Year for Daily Chart",
        options=years,
        index=years.index(current_year)
    )
    
    # Display last update time
    st.sidebar.text(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Model selection
    st.sidebar.header("Model Selection")
    
    # Get available models
    available_models = get_available_models()
    
    # Ensure domain_aware model is available
    if "domain_aware" not in available_models:
        available_models.append("domain_aware")
    
    # Dictionary to display friendly names
    model_display_names = {
        "synthetic": "Synthetic (Fire Brigade Domain)",
        "twitter": "Twitter (General Social Media)",
        "hybrid": "Hybrid (Combined Approach)",
        "domain_aware": "Domain-Aware (Enhanced Model)",
        "expanded": "Expanded (Multi-Dataset General Model)",
        "ensemble": "Ensemble (Combined Strengths)",
        "default": "Default Model"
    }
    
    # Create dropdown options
    model_options = {model_display_names.get(model, model): model for model in available_models}
    
    selected_display_name = st.sidebar.selectbox(
        "Select Sentiment Model",
        options=list(model_options.keys()),
        index=list(model_options.keys()).index(model_display_names.get(st.session_state.selected_model, "Domain-Aware (Enhanced Model)"))
    )
    
    # Update the selected model in session state
    st.session_state.selected_model = model_options[selected_display_name]
    
    # Add model descriptions
    if st.session_state.selected_model == "synthetic":
        st.sidebar.info("The synthetic model is trained specifically on fire brigade domain text and performs best on domain-specific content (88.9% accuracy).")
    elif st.session_state.selected_model == "twitter":
        st.sidebar.info("The Twitter model is trained on real-world social media data and may handle colloquial expressions better, but has lower accuracy on domain text (33.3%).")
    elif st.session_state.selected_model == "hybrid":
        st.sidebar.info("The hybrid model combines both approaches, weighing predictions based on domain specificity and confidence scores (77.8% accuracy on domain text).")
    elif st.session_state.selected_model == "domain_aware":
        st.sidebar.info("The domain-aware model uses specialized classification for each emergency service domain, offering improved accuracy (85-92%) and better performance with neutral sentiments.")
    elif st.session_state.selected_model == "expanded":
        st.sidebar.info("The expanded model is trained on multiple large datasets (160,000+ tweets) and can handle a wide variety of domains with 73.5% overall accuracy, though performance varies by domain (100% on social media, 47.5% on emergency services).")
    elif st.session_state.selected_model == "ensemble":
        st.sidebar.info("The ensemble model combines the strengths of all available models using weighted voting based on each model's performance. It offers the best overall performance for emergency services text by prioritizing the most confident predictions from models that excel in specific sentiment categories.")
    
    # Sentiment prediction tool
    st.sidebar.header("Sentiment Prediction Tool")
    custom_text = st.sidebar.text_area("Enter text to analyze:")
    
    # Add option to compare models
    st.session_state.show_comparison = st.sidebar.checkbox("Compare all models", value=st.session_state.show_comparison)
    
    if st.sidebar.button("Analyze Sentiment"):
        if custom_text:
            with st.sidebar:
                with st.spinner("Predicting sentiment..."):
                    if st.session_state.show_comparison:
                        display_model_comparison(custom_text)
                    else:
                        result = predict_sentiment(custom_text, model_type=st.session_state.selected_model)
                        st.session_state.last_prediction = result
                        st.session_state.feedback_submitted = False
                        display_sentiment_result(result)
                        
                        # Add feedback component
                        display_feedback_component(custom_text, result)
        else:
            st.sidebar.warning("Please enter text to analyze")
    
    # Main content
    # Create a 2x2 grid with key metrics and charts
    col1, col2 = st.columns(2)
    
    # First row: summary metrics
    with col1:
        display_summary_metrics()
    
    with col2:
        display_sentiment_distribution()
    
    # Second row: time series and recent incidents
    col3, col4 = st.columns(2)
    
    with col3:
        display_sentiment_over_time(days=time_range)
    
    with col4:
        display_recent_incidents()
    
    # Third row: daily sentiment chart
    st.subheader("Daily Sentiment Analysis")
    display_daily_sentiment_chart(month=selected_month, year=selected_year)
    
    # Third row: active learning section
    if ACTIVE_LEARNING_AVAILABLE:
        st.subheader("Active Learning")
        st.markdown("""
        This system continuously improves through your feedback. When you provide feedback on sentiment predictions, 
        the model learns from your corrections to improve future predictions, especially on ambiguous or domain-specific text.
        """)
        
        # Display metrics on feedback and retraining
        st.info("The active learning system focuses on uncertain predictions, where the model's confidence is low. Your feedback on these predictions provides the greatest learning value.")
    
    # Auto-refresh
    if auto_refresh:
        st.empty()
        # Add JavaScript for auto-refresh
        st.markdown(
            f"""
            <script>
                setTimeout(function(){{
                    window.location.reload();
                }}, {refresh_interval * 1000});
            </script>
            """,
            unsafe_allow_html=True
        )

def display_sentiment_result(result):
    """Display sentiment prediction result."""
    # Extract values from result
    sentiment = result.get("sentiment", "neutral")
    confidence = result.get("confidence", 0.0) * 100
    model_type = result.get("model_type", "unknown")
    model_used = result.get("model_used", None)
    
    # Select color based on sentiment
    if sentiment == "positive":
        color = "green"
    elif sentiment == "negative":
        color = "red"
    else:
        color = "gray"
    
    st.markdown(f"**Sentiment:** <span style='color:{color}'>{sentiment.upper()}</span>", unsafe_allow_html=True)
    st.progress(confidence / 100)
    st.text(f"Confidence: {confidence:.1f}%")
    
    # Display model information
    if model_type:
        st.text(f"Model: {model_type}")
    
    # For hybrid model, show which sub-model was used
    if model_used:
        st.text(f"Decision from: {model_used}")
    
    # For domain-aware model, show domains detected
    if model_type == "domain_aware" and "domains" in result:
        domains = result.get("domains", ["general"])
        st.text(f"Domains: {', '.join(domains)}")

def display_feedback_component(text, result):
    """Display feedback component for active learning."""
    # Check if feedback was already submitted
    if st.session_state.feedback_submitted:
        st.success("Thank you for your feedback! Your input helps improve our model.")
        return
    
    st.markdown("### Provide Feedback")
    st.markdown("Was this sentiment prediction correct?")
    
    # Extract the predicted sentiment for reference
    sentiment_value = result.get("sentiment_value", 0)
    sentiment_label = result.get("sentiment", "neutral")
    
    # Show the current prediction to the user
    st.info(f"Current prediction: {sentiment_label.upper()} (value: {sentiment_value})")
    
    # Use radio buttons for feedback
    selected_feedback = st.radio(
        "Select the correct sentiment:",
        ["👎 Negative", "😐 Neutral", "👍 Positive"],
        horizontal=True,
        # Set default selection based on current prediction
        index={-1: 0, 0: 1, 1: 2}.get(sentiment_value, 1),
        key="feedback_sentiment_radio"
    )
    
    # Map the selection to sentiment values
    feedback_map = {
        "👎 Negative": -1,
        "😐 Neutral": 0,
        "👍 Positive": 1
    }
    
    # Create a container to show feedback status
    feedback_status = st.empty()
    
    # Use regular button (not a form)
    if st.button("Submit Feedback", key="feedback_submit_button"):
        corrected_sentiment = feedback_map.get(selected_feedback, 0)
        logger.info(f"Submit button clicked with corrected sentiment: {corrected_sentiment}")
        submit_feedback(text, corrected_sentiment)
        
        # Check if feedback was submitted successfully
        if st.session_state.feedback_submitted:
            with feedback_status:
                st.success("Thank you for your feedback! Your input helps improve our model.")
        return

def submit_feedback(text, selected_sentiment):
    """Process feedback submission"""
    try:
        # Log the feedback attempt with more details
        logger.info(f"===== SUBMIT FEEDBACK =====")
        logger.info(f"Text: '{text[:50]}...'")
        logger.info(f"Selected sentiment: {selected_sentiment}")
        logger.info(f"Current model: {st.session_state.selected_model}")
        
        # Call the provide_feedback function
        success = provide_feedback(text, selected_sentiment)
        
        # Log result
        logger.info(f"Feedback submission {'successful' if success else 'failed'}")
        
        # Only update session state if feedback was successfully stored
        if success:
            st.session_state.feedback_submitted = True
            # Log successful feedback submission
            logger.info(f"Feedback submitted successfully for text: '{text[:30]}...'")
            # Show success message directly instead of forcing a rerun
            st.success("Thank you for your feedback! Your input helps improve our model.")
        else:
            # Log failed feedback submission
            logger.error(f"Failed to submit feedback for text: '{text[:30]}...'")
            st.error("Failed to submit feedback. Please try again.")
            
    except Exception as e:
        # Log any errors
        logger.error(f"Error in feedback submission: {str(e)}")
        st.error(f"Error submitting feedback: {str(e)}")
        
    # Don't force page reload - let user see the result

def provide_feedback(text, corrected_sentiment):
    """Process and store user feedback."""
    try:
        # Get the model type from session state
        model_type = st.session_state.selected_model
        
        logger.info(f"Storing feedback for text: '{text[:50]}...' with corrected sentiment: {corrected_sentiment}")
        
        # First get a prediction with store_for_feedback=True to generate an entry_id
        prediction_result = predict_sentiment(
            text, 
            model_type=model_type, 
            store_for_feedback=True
        )
        
        # Log the prediction result
        logger.info(f"Received prediction for feedback storage: {prediction_result}")
        
        # Get entry_id from the prediction
        entry_id = prediction_result.get("entry_id")
        
        if not entry_id:
            logger.error("Missing entry_id in prediction response")
            return False
            
        # Submit feedback using the entry_id
        feedback_data = {
            "entry_id": entry_id,
            "corrected_sentiment": corrected_sentiment
        }
        
        # Fix URL construction to ensure it correctly points to the /feedback endpoint
        base_url = MODEL_API_URL
        if base_url.endswith('/predict'):
            base_url = base_url[:-8]  # Remove '/predict' from the end
        
        feedback_url = f"{base_url}/feedback"
        
        # Log the feedback URL
        logger.info(f"Sending feedback to URL: {feedback_url}")
        
        feedback_response = requests.post(
            feedback_url,
            json=feedback_data,
            headers={"Content-Type": "application/json"}
        )
        
        if feedback_response.status_code == 200:
            feedback_result = feedback_response.json()
            success = feedback_result.get("success", False)
            logger.info(f"Feedback API response: {feedback_result}")
            
            if success:
                logger.info(f"Successfully stored feedback: corrected={corrected_sentiment}")
                return True
            else:
                logger.error(f"API reported failure storing feedback: {feedback_result.get('message', 'Unknown error')}")
                return False
        else:
            logger.error(f"Failed to submit feedback: {feedback_response.status_code} - {feedback_response.text}")
            return False
        
    except Exception as e:
        logger.error(f"Error providing feedback: {str(e)}")
        return False

def display_model_comparison(text):
    """Display comparison of sentiment predictions from all models."""
    result = compare_models(text)
    
    if "error" in result:
        st.error(f"Error comparing models: {result['error']}")
        return
    
    # Display original text
    st.markdown(f"**Text:** {result.get('text', '')}")
    
    # Create columns for each model
    models_data = result.get("models", {})
    
    if not models_data:
        st.warning("No model results available for comparison.")
        return
    
    st.markdown("### Model Comparison")
    
    # Create a table for comparison
    comparison_data = []
    
    for model_name, prediction in models_data.items():
        sentiment = prediction.get("sentiment", "neutral")
        confidence = prediction.get("confidence", 0.0) * 100
        model_used = prediction.get("model_used", "N/A")
        
        comparison_data.append({
            "Model": model_name,
            "Sentiment": sentiment.upper(),
            "Confidence": f"{confidence:.1f}%",
            "Decision Model": model_used if model_used != "N/A" else ""
        })
    
    # Add domain-aware model if not present
    if "domain_aware" not in [d["Model"] for d in comparison_data]:
        # Try to get result from domain-aware model
        try:
            domain_result = predict_sentiment(text, model_type="domain_aware")
            domain_model_data = {
                "Model": "domain_aware",
                "Sentiment": domain_result.get("sentiment", "neutral").upper(),
                "Confidence": f"{domain_result.get('confidence', 0.0) * 100:.1f}%",
                "Decision Model": "Domain-Specific"
            }
            comparison_data.append(domain_model_data)
            
            # Also add to models_data for the feedback dropdown
            models_data["domain_aware"] = {
                "sentiment": domain_result.get("sentiment", "neutral"),
                "confidence": domain_result.get("confidence", 0.0),
                "model_used": "Domain-Specific"
            }
        except Exception as e:
            logger.error(f"Error adding domain-aware model to comparison: {str(e)}")
    
    # Convert to DataFrame and display
    comparison_df = pd.DataFrame(comparison_data)
    
    # Use st.dataframe for more styling options - update the styling to improve text visibility
    st.dataframe(
        comparison_df.style.map(
            lambda x: 'color: white; background-color: #E74C3C' if isinstance(x, str) and x.lower() == "negative" else
                   ('color: white; background-color: #3498DB' if isinstance(x, str) and x.lower() == "positive" else
                    ('color: black; background-color: #95A5A6' if isinstance(x, str) and x.lower() == "neutral" else "")),
            subset=["Sentiment"]
        ),
        use_container_width=True
    )
    
    # Store last prediction for feedback - also store comparison_data
    st.session_state.last_prediction = {
        "text": text,
        "models": models_data,
        "comparison_data": comparison_data
    }
    
    # Add feedback component for the comparison
    display_comparison_feedback(text, models_data)

def display_comparison_feedback(text, models_data):
    """Display feedback component after model comparison."""
    if st.session_state.feedback_submitted:
        st.success("Thank you for your feedback! Your input helps improve our models.")
        return
    
    st.markdown("### Provide Feedback")
    st.markdown("Which prediction do you think is correct?")
    
    # Show the predictions from each model for reference
    for model_name, prediction in models_data.items():
        sentiment = prediction.get("sentiment", "neutral")
        confidence = prediction.get("confidence", 0.0) * 100
        st.info(f"{model_name}: {sentiment.upper()} (confidence: {confidence:.1f}%)")
    
    # Direct sentiment selection for clearer feedback
    selected_sentiment = st.radio(
        "Select the correct sentiment:",
        ["👎 Negative", "😐 Neutral", "👍 Positive"],
        horizontal=True,
        index=1,  # Default to neutral
        key="comparison_feedback_radio"
    )
    
    # Map the selection to sentiment values
    sentiment_map = {
        "👎 Negative": -1,
        "😐 Neutral": 0,
        "👍 Positive": 1
    }
    
    # Create a container to show feedback status
    feedback_status = st.empty()
    
    # Use regular button (not a form)
    if st.button("Submit Feedback", key="comparison_feedback_button"):
        corrected_sentiment = sentiment_map.get(selected_sentiment, 0)
        logger.info(f"Comparison submit button clicked with corrected sentiment: {corrected_sentiment}")
        submit_feedback(text, corrected_sentiment)
        
        # Check if feedback was submitted successfully
        if st.session_state.feedback_submitted:
            with feedback_status:
                st.success("Thank you for your feedback! Your input helps improve our models.")
        return

def display_summary_metrics():
    """Display summary metrics in a card."""
    st.subheader("Summary Metrics")
    
    # Get sentiment statistics
    stats = get_sentiment_statistics()
    
    # Create three columns for the metrics
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        positive_pct = (stats.get('positive', 0) / max(stats.get('total', 1), 1)) * 100
        st.metric(
            label="Positive Sentiment",
            value=f"{stats.get('positive', 0)}",
            delta=f"{positive_pct:.1f}%"
        )
    
    with metric_col2:
        neutral_pct = (stats.get('neutral', 0) / max(stats.get('total', 1), 1)) * 100
        st.metric(
            label="Neutral Sentiment",
            value=f"{stats.get('neutral', 0)}",
            delta=f"{neutral_pct:.1f}%"
        )
    
    with metric_col3:
        negative_pct = (stats.get('negative', 0) / max(stats.get('total', 1), 1)) * 100
        st.metric(
            label="Negative Sentiment",
            value=f"{stats.get('negative', 0)}",
            delta=f"{negative_pct:.1f}%",
            delta_color="inverse"  # Inverse color (red is bad for negative)
        )

def display_sentiment_distribution():
    """Display sentiment distribution chart."""
    st.subheader("Sentiment Distribution")
    
    # Get sentiment statistics
    stats = get_sentiment_statistics()
    
    # Create data for pie chart
    labels = ['Positive', 'Neutral', 'Negative']
    values = [stats.get('positive', 0), stats.get('neutral', 0), stats.get('negative', 0)]
    colors = ['#4CAF50', '#9E9E9E', '#F44336']  # Green, Gray, Red
    
    # Only create chart if we have data
    if sum(values) > 0:
        # Create figure
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(
            values, 
            labels=labels, 
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            explode=(0.05, 0.05, 0.05),
            wedgeprops={'width': 0.6, 'edgecolor': 'w'}
        )
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        st.pyplot(fig)
    else:
        st.info("No sentiment data available yet. Run the data ingestion script to collect data.")

def display_sentiment_over_time(days=30):
    """Display sentiment trends over time."""
    st.subheader(f"Sentiment Trends (Last {days} Days)")
    
    # Get sentiment trends data
    df = get_sentiment_over_time(days=days)
    
    if df.empty:
        st.info("No sentiment trend data available for the selected time range.")
        return
    
    # Melt the DataFrame for easier charting
    df_melted = df.melt(
        id_vars=['date'],
        value_vars=['positive', 'neutral', 'negative'],
        var_name='sentiment',
        value_name='count'
    )
    
    # Create Altair chart
    chart = alt.Chart(df_melted).mark_area().encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('count:Q', title='Number of Incidents', stack='normalize'),
        color=alt.Color(
            'sentiment:N', 
            scale=alt.Scale(
                domain=['positive', 'neutral', 'negative'],
                range=['#4CAF50', '#9E9E9E', '#F44336']
            ),
            legend=alt.Legend(title='Sentiment')
        ),
        tooltip=['date:T', 'sentiment:N', 'count:Q']
    ).properties(
        height=300
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)

def display_recent_incidents():
    """Display a table of recent incidents with sentiment."""
    st.subheader("Recent Incidents")
    
    # Get recent incidents
    df = get_recent_incidents(limit=10)
    
    if df.empty:
        st.info("No incident data available. Run the data ingestion script to collect data.")
        return
    
    # Display incidents in a styled table
    # Create a color map for sentiment labels
    sentiment_colors = {
        'positive': '#E3F2E3',  # Light green
        'neutral': '#F5F5F5',   # Light gray
        'negative': '#FADBD8',  # Light red
        'not analyzed': '#FFFFFF'  # White
    }
    
    # Create a DataFrame for display
    display_df = df[['incident_id', 'report', 'formatted_time', 'sentiment_label']]
    display_df = display_df.rename(columns={
        'incident_id': 'ID',
        'report': 'Description',
        'formatted_time': 'Time',
        'sentiment_label': 'Sentiment'
    })
    
    # Style and display the table - update to use style.map instead of deprecated applymap
    st.dataframe(
        display_df.style.map(
            lambda x: f"background-color: {sentiment_colors.get(x.lower(), '#FFFFFF')}" 
            if isinstance(x, str) and x.lower() in sentiment_colors 
            else "",
            subset=['Sentiment']
        ),
        use_container_width=True
    )

def display_daily_sentiment_chart(month=None, year=None):
    """
    Display a line chart showing sentiment trends for each day of a specific month.
    
    Args:
        month (int): Month to display (1-12), defaults to current month
        year (int): Year to display, defaults to current year
    """
    # Set defaults
    if month is None:
        month = datetime.now().month
    if year is None:
        year = datetime.now().year
    
    # Get data for the specific month
    df = get_sentiment_by_day_of_month(month=month, year=year)
    
    if df.empty:
        st.info(f"No sentiment data available for {calendar.month_name[month]} {year}.")
        return
    
    # Melt the DataFrame for Altair
    df_melted = df.melt(
        id_vars=['day'],
        value_vars=['positive', 'neutral', 'negative'],
        var_name='sentiment',
        value_name='count'
    )
    
    # Create Altair chart
    chart = alt.Chart(df_melted).mark_line(point=True).encode(
        x=alt.X('day:O', title=f'Day of {calendar.month_name[month]}', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('count:Q', title='Number of Incidents'),
        color=alt.Color(
            'sentiment:N', 
            scale=alt.Scale(
                domain=['positive', 'neutral', 'negative'],
                range=['#4CAF50', '#9E9E9E', '#F44336']
            ),
            legend=alt.Legend(title='Sentiment')
        ),
        tooltip=['day:O', 'sentiment:N', 'count:Q']
    ).properties(
        height=400,
        title=f'Daily Sentiment Analysis for {calendar.month_name[month]} {year}'
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)
    
    # Add download button for the data
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Daily Sentiment Data",
        data=csv,
        file_name=f"daily_sentiment_{year}_{month}.csv",
        mime="text/csv",
    )

if __name__ == "__main__":
    main() 