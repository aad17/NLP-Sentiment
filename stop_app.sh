#!/bin/bash

# Stop Sentiment Analysis Application
# This script stops all components of the Fire Brigade Sentiment Analysis application

# Function to print colored output
print_color() {
    COLOR=$1
    TEXT=$2
    case $COLOR in
        "green") printf "\033[0;32m%s\033[0m\n" "$TEXT" ;;
        "red") printf "\033[0;31m%s\033[0m\n" "$TEXT" ;;
        "yellow") printf "\033[0;33m%s\033[0m\n" "$TEXT" ;;
        "blue") printf "\033[0;34m%s\033[0m\n" "$TEXT" ;;
        *) echo "$TEXT" ;;
    esac
}

print_color "blue" "Stopping Fire Brigade Sentiment Analysis Application..."
print_color "yellow" "Note: All trained models (including the expanded sentiment model) will be preserved."

# Check if .app_pids file exists
if [ -f ".app_pids" ]; then
    print_color "blue" "Stopping application components..."
    PIDS=$(cat .app_pids)
    for PID in $PIDS; do
        if ps -p $PID > /dev/null; then
            print_color "blue" "Stopping process with PID $PID..."
            kill $PID
            print_color "green" "Stopped process with PID $PID."
        else
            print_color "yellow" "Process with PID $PID is not running."
        fi
    done
    rm -f .app_pids
    print_color "green" "All services stopped."
else
    # Try to find and kill any remaining processes
    print_color "yellow" "No PID file found. Attempting to find and stop processes..."
    
    # Find processes for mock API
    MOCK_PIDS=$(ps aux | grep "mock_api.py" | grep -v grep | awk '{print $2}')
    if [ -n "$MOCK_PIDS" ]; then
        for PID in $MOCK_PIDS; do
            print_color "blue" "Stopping Mock API with PID $PID..."
            kill $PID
        done
        print_color "green" "Mock API stopped."
    else
        print_color "yellow" "No Mock API process found."
    fi
    
    # Find processes for model API
    MODEL_PIDS=$(ps aux | grep "model_api.py" | grep -v grep | awk '{print $2}')
    if [ -n "$MODEL_PIDS" ]; then
        for PID in $MODEL_PIDS; do
            print_color "blue" "Stopping Model API with PID $PID..."
            kill $PID
        done
        print_color "green" "Model API stopped."
    else
        print_color "yellow" "No Model API process found."
    fi
    
    # Find processes for Streamlit dashboard
    STREAMLIT_PIDS=$(ps aux | grep "streamlit run" | grep "dashboard.py" | grep -v grep | awk '{print $2}')
    if [ -n "$STREAMLIT_PIDS" ]; then
        for PID in $STREAMLIT_PIDS; do
            print_color "blue" "Stopping Streamlit dashboard with PID $PID..."
            kill $PID
        done
        print_color "green" "Streamlit dashboard stopped."
    else
        print_color "yellow" "No Streamlit dashboard process found."
    fi
    
    # Find processes for Social Media Batch Processor
    PROCESSOR_PIDS=$(ps aux | grep "data_processor.py" | grep -v grep | awk '{print $2}')
    if [ -n "$PROCESSOR_PIDS" ]; then
        for PID in $PROCESSOR_PIDS; do
            print_color "blue" "Stopping Social Media Batch Processor with PID $PID..."
            kill $PID
        done
        print_color "green" "Social Media Batch Processor stopped."
    else
        print_color "yellow" "No Social Media Batch Processor process found."
    fi
    
    # Find processes for Social Media Auth UI
    AUTH_PIDS=$(ps aux | grep "streamlit run" | grep "social_media_auth.py" | grep -v grep | awk '{print $2}')
    if [ -n "$AUTH_PIDS" ]; then
        for PID in $AUTH_PIDS; do
            print_color "blue" "Stopping Social Media Auth UI with PID $PID..."
            kill $PID
        done
        print_color "green" "Social Media Auth UI stopped."
    else
        print_color "yellow" "No Social Media Auth UI process found."
    fi
    
    # Find processes for Social Media Data UI
    DATA_PIDS=$(ps aux | grep "streamlit run" | grep "social_media_data.py" | grep -v grep | awk '{print $2}')
    if [ -n "$DATA_PIDS" ]; then
        for PID in $DATA_PIDS; do
            print_color "blue" "Stopping Social Media Data UI with PID $PID..."
            kill $PID
        done
        print_color "green" "Social Media Data UI stopped."
    else
        print_color "yellow" "No Social Media Data UI process found."
    fi
    
    # Find processes for CSV Data Upload UI
    CSV_UPLOAD_PIDS=$(ps aux | grep "streamlit run" | grep "csv_data_upload.py" | grep -v grep | awk '{print $2}')
    if [ -n "$CSV_UPLOAD_PIDS" ]; then
        for PID in $CSV_UPLOAD_PIDS; do
            print_color "blue" "Stopping CSV Data Upload UI with PID $PID..."
            kill $PID
        done
        print_color "green" "CSV Data Upload UI stopped."
    else
        print_color "yellow" "No CSV Data Upload UI process found."
    fi
fi

print_color "green" "=========================================================="
print_color "green" "Fire Brigade Sentiment Analysis Application stopped."
print_color "blue" "All sentiment models (including the expanded model) are preserved."
print_color "blue" "Run ./start_app.sh to restart the application."
print_color "green" "==========================================================" 