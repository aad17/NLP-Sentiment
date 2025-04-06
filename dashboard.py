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
    
    # Display last update time
    st.sidebar.text(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
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
    
    # Third row: active learning section
    if ACTIVE_LEARNING_AVAILABLE:
        st.subheader("Active Learning")
        st.markdown("""
        This system continuously improves through your feedback. When you provide feedback on sentiment predictions, 
        the model learns from your corrections to improve future predictions, especially on ambiguous or domain-specific text.
        """)
        
        # Display metrics on feedback and retraining
        st.info("The active learning system focuses on uncertain predictions, where the model's confidence is low. Your feedback on these predictions provides the greatest learning value.") 