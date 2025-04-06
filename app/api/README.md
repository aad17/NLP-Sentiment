# Fire Brigade Incident API Documentation

This document describes the endpoints available in the Fire Brigade Incident Mock API.

## Base URL

When running locally: `http://localhost:5000`

## Endpoints

### Get Incidents

Fetches incident reports with sentiment information.

**URL**: `/get_incidents`

**Method**: `GET`

**Query Parameters**:

| Parameter | Type | Description | Required | Default | Constraints |
|-----------|------|-------------|----------|---------|-------------|
| count | integer | Number of incidents to return | No | 5 | 1-100 |

**Success Response**:

- **Code**: 200 OK
- **Content Example**:

```json
[
  {
    "incident_id": "a1b2c3d4",
    "report": "Successfully rescued family from burning building with no injuries.",
    "timestamp": "2023-04-01T14:30:00Z"
  },
  {
    "incident_id": "e5f6g7h8",
    "report": "Response time delayed due to traffic congestion, increasing property damage.",
    "timestamp": "2023-04-02T09:15:00Z"
  }
]
```

**Error Responses**:

- **Code**: 400 Bad Request
  - **Condition**: If count parameter is less than 1 or greater than 100
  - **Content Example**: `{"error": "Count must be at least 1"}`

- **Code**: 500 Internal Server Error
  - **Condition**: If there's a server error
  - **Content Example**: `{"error": "Internal server error"}`

### Health Check

Simple endpoint to check if the API is running.

**URL**: `/health`

**Method**: `GET`

**Success Response**:

- **Code**: 200 OK
- **Content Example**: `{"status": "healthy"}`

## Data Structure

Each incident has the following structure:

| Field | Type | Description |
|-------|------|-------------|
| incident_id | string | Unique identifier for the incident |
| report | string | Text description of the incident |
| timestamp | string | ISO 8601 formatted timestamp |

## Running the API

To run the API locally:

```bash
python app/api/mock_api.py
```

## Testing the API

Example curl command:

```bash
curl "http://localhost:5000/get_incidents?count=3"
``` 