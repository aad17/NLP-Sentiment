"""
Job Monitoring Utility for Sentiment Analysis Application

This script checks the status of data ingestion jobs and provides monitoring capabilities.
"""
import os
import sys
import glob
import logging
import argparse
from datetime import datetime, timedelta
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('job_monitor')

def get_latest_log_file():
    """
    Find the most recent log file in the logs directory.
    
    Returns:
        str: Path to the latest log file, None if no logs found
    """
    # Get the project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    logs_dir = os.path.join(project_root, 'logs')
    
    if not os.path.exists(logs_dir):
        logger.warning(f"Logs directory not found: {logs_dir}")
        return None
    
    # Get all log files for data ingestion
    log_files = glob.glob(os.path.join(logs_dir, 'data_ingestion_*.log'))
    
    if not log_files:
        logger.warning("No log files found")
        return None
    
    # Return the most recent log file
    return max(log_files, key=os.path.getctime)

def check_job_status():
    """
    Check the status of the most recent data ingestion job.
    
    Returns:
        dict: Status dictionary with job information
    """
    latest_log = get_latest_log_file()
    
    if not latest_log:
        return {
            'status': 'unknown',
            'error': 'No log files found',
            'last_run': None,
            'details': None
        }
    
    # Get timestamp from filename
    log_filename = os.path.basename(latest_log)
    try:
        # Extract timestamp from filename format data_ingestion_YYYY-MM-DD_HH-MM-SS.log
        timestamp_str = log_filename.replace('data_ingestion_', '').replace('.log', '')
        last_run = datetime.strptime(timestamp_str, '%Y-%m-%d_%H-%M-%S')
    except Exception as e:
        logger.error(f"Error parsing timestamp from log filename: {str(e)}")
        last_run = datetime.fromtimestamp(os.path.getctime(latest_log))
    
    # Check if job was successful
    try:
        with open(latest_log, 'r') as f:
            log_content = f.read()
            
        if "Data ingestion completed successfully" in log_content:
            status = 'success'
            error = None
        elif "Error running data ingestion script" in log_content:
            status = 'failed'
            # Extract error details
            error_lines = [line for line in log_content.splitlines() if "ERROR" in line]
            error = error_lines[-1] if error_lines else "Unknown error"
        else:
            status = 'unknown'
            error = "Could not determine job status"
            
        # Extract details
        details = []
        for line in log_content.splitlines():
            if "Fetched" in line or "Processed" in line or "Saved" in line:
                details.append(line.strip())
                
        return {
            'status': status,
            'error': error,
            'last_run': last_run.isoformat(),
            'details': details
        }
    except Exception as e:
        logger.error(f"Error reading log file: {str(e)}")
        return {
            'status': 'unknown',
            'error': f"Error reading log file: {str(e)}",
            'last_run': last_run.isoformat() if last_run else None,
            'details': None
        }

def check_stale_jobs(hours=24):
    """
    Check if the most recent job is stale (older than specified hours).
    
    Args:
        hours (int): Number of hours to consider a job stale
        
    Returns:
        dict: Status information including staleness
    """
    status = check_job_status()
    
    if status['last_run'] is None:
        status['is_stale'] = True
        status['stale_hours'] = None
        return status
    
    last_run = datetime.fromisoformat(status['last_run'])
    time_since_last_run = datetime.now() - last_run
    
    status['is_stale'] = time_since_last_run > timedelta(hours=hours)
    status['stale_hours'] = time_since_last_run.total_seconds() / 3600
    
    return status

def main():
    """Main function to run the job monitor."""
    parser = argparse.ArgumentParser(description='Monitor data ingestion jobs')
    parser.add_argument('--check-stale', type=int, default=24, 
                       help='Check if jobs are stale (hours)')
    parser.add_argument('--json', action='store_true', 
                       help='Output in JSON format')
    args = parser.parse_args()
    
    if args.check_stale:
        status = check_stale_jobs(args.check_stale)
    else:
        status = check_job_status()
    
    if args.json:
        # Convert datetime to string for JSON serialization
        print(json.dumps(status, indent=2))
    else:
        # Pretty print the status
        print(f"Job Status: {status['status']}")
        print(f"Last Run: {status['last_run']}")
        
        if status.get('is_stale') is not None:
            print(f"Stale: {'Yes' if status['is_stale'] else 'No'}")
            if status['stale_hours'] is not None:
                print(f"Hours since last run: {status['stale_hours']:.1f}")
        
        if status['error']:
            print(f"Error: {status['error']}")
            
        if status['details']:
            print("\nDetails:")
            for detail in status['details']:
                print(f"  {detail}")

if __name__ == "__main__":
    main() 