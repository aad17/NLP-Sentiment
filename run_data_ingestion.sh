#!/bin/bash

# Script to run data ingestion for the Sentiment Analysis application
# This script can be called manually or by a cron job

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Virtual environment not found. Make sure you're in the correct directory."
    exit 1
fi

# Set up log directory
LOGS_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOGS_DIR"

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="$LOGS_DIR/data_ingestion_$TIMESTAMP.log"

echo "Starting data ingestion at $TIMESTAMP" | tee -a "$LOG_FILE"

# Run data ingestion script
python app/data/data_ingestion.py --count 20 2>&1 | tee -a "$LOG_FILE"

# Check if script succeeded
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "Data ingestion completed successfully" | tee -a "$LOG_FILE"
else
    echo "Error running data ingestion script" | tee -a "$LOG_FILE"
    exit 1
fi

# Deactivate virtual environment
deactivate

echo "Finished at $(date +"%Y-%m-%d %H:%M:%S")" | tee -a "$LOG_FILE" 