# Setting Up Scheduled Execution with Cron

This document explains how to set up scheduled execution of the data ingestion script using cron jobs.

## Prerequisites

- Linux or macOS system with cron installed
- The Sentiment Analysis application properly set up

## Scheduling Options

### Hourly Execution

To run the data ingestion script every hour:

```bash
# Edit your crontab
crontab -e

# Add this line
0 * * * * /full/path/to/Sentiment_Analysis/run_data_ingestion.sh
```

### Daily Execution

To run the data ingestion script once a day at midnight:

```bash
# Edit your crontab
crontab -e

# Add this line
0 0 * * * /full/path/to/Sentiment_Analysis/run_data_ingestion.sh
```

### Custom Schedule

To run the data ingestion script every 6 hours:

```bash
# Edit your crontab
crontab -e

# Add this line
0 */6 * * * /full/path/to/Sentiment_Analysis/run_data_ingestion.sh
```

## Checking Cron Job Status

To check if your cron jobs are properly set up:

```bash
crontab -l
```

## Viewing Logs

All logs are stored in the `logs` directory:

```bash
ls -l /path/to/Sentiment_Analysis/logs/
```

The most recent log file will have the most recent timestamp in its filename.

## Troubleshooting

If your cron job doesn't seem to be running:

1. Check if cron is running: `systemctl status cron` (Linux) or `launchctl list | grep cron` (macOS)
2. Ensure the script has execute permissions: `chmod +x run_data_ingestion.sh`
3. Use absolute paths in your crontab entry
4. Verify the log files in the logs directory 