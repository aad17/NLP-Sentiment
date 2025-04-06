# **Sentiment Analysis Application - Full Implementation Documentation**  

This documentation provides a step-by-step guide to building a **cost-effective sentiment analysis application** for a fire brigade company. It includes data ingestion, ML model processing, and dashboard visualization.  

---

## **Table of Contents**  
1. **Project Overview**  
2. **Tech Stack**  
3. **System Architecture**  
4. **Setup & Installation**  
5. **Implementation Guide**  
   - **Mock API for Incident Data**  
   - **Data Ingestion Script**  
   - **Sentiment Analysis Model**  
   - **Model Prediction API**  
   - **Dashboard (Streamlit)**  
6. **Deployment Guide**  
7. **Future Enhancements**  

---

## **1. Project Overview**  
The sentiment analysis application processes fire brigade incident reports, determines sentiment (positive, neutral, negative), and visualizes the data in a dashboard.  

### **Key Features**  
- **Batch data ingestion** from an API (mock for now).  
- **Sentiment analysis** using an ML model.  
- **Dashboard visualization** using Streamlit.  
- **Minimal cost setup** with free-tier deployment options.  

---

## **2. Tech Stack**  

| **Component** | **Technology** | **Why?** |
|--------------|---------------|----------|
| **Mock API** | Flask (Render Free Tier) | Lightweight and free to host. |
| **Data Storage** | SQLite / Google Sheets | Simple, free-tier solutions. |
| **Data Processing** | Python Script (Cron Job) | Automates batch ingestion without Airflow. |
| **ML Model** | Scikit-learn (TF-IDF + Logistic Regression) | Lightweight and fast. |
| **Backend** | Flask (Render Free Tier) | Serves predictions. |
| **Dashboard** | Streamlit (Free Hosting) | Quick and interactive visualization. |

---

## **3. System Architecture**  

```
+-------------------------------------------------+
|             User Interaction Layer             |
|  - Streamlit Dashboard (User Insights)        |
|  - API Calls to Fetch Predictions             |
+-------------------------------------------------+
              ⬆          ⬆
+-------------------------------------------------+
|          Backend (Model Inference)             |
|  - Flask API (Serves Predictions)             |
|  - ML Model (TF-IDF + Logistic Regression)    |
+-------------------------------------------------+
              ⬆          ⬆
+-------------------------------------------------+
|          Data Processing Layer                 |
|  - Python Script (Cron Job for Ingestion)     |
|  - SQLite / Google Sheets                     |
+-------------------------------------------------+
              ⬆          ⬆
+-------------------------------------------------+
|          Data Source (Mock API)                |
|  - Flask API (Simulated Incident Reports)     |
+-------------------------------------------------+
```

---

## **4. Setup & Installation**  

### **Step 1: Install Dependencies**  
Run the following command to install required packages:  
```bash
pip install flask fastapi requests sqlite3 scikit-learn pickle-mixin streamlit
```

---

## **5. Implementation Guide**  

### **5.1 Mock API for Incident Data**  
A Flask API that returns random fire brigade incident reports.  

#### **Implementation (`mock_api.py`)**  
```python
from flask import Flask, jsonify
import random

app = Flask(__name__)

INCIDENTS = [
    {"incident_id": "001", "report": "Firefighters saved a trapped child.", "timestamp": "2025-04-04T12:00:00Z"},
    {"incident_id": "002", "report": "Explosion reported, no casualties.", "timestamp": "2025-04-04T14:30:00Z"},
    {"incident_id": "003", "report": "Fire extinguished at warehouse.", "timestamp": "2025-04-04T16:00:00Z"},
]

@app.route("/get_incidents", methods=["GET"])
def get_incidents():
    return jsonify(random.choices(INCIDENTS, k=2))

if __name__ == "__main__":
    app.run(debug=True)
```

### **Deployment**
Deploy this API for free on [Render](https://render.com/).

---

### **5.2 Data Ingestion Script**  
A Python script that fetches data and stores it in **SQLite**.  

#### **Implementation (`data_ingestion.py`)**  
```python
import requests
import sqlite3

API_URL = "https://your-render-app.com/get_incidents"

# Fetch data from API
response = requests.get(API_URL)
incidents = response.json()

# Save to SQLite
conn = sqlite3.connect("incidents.db")
c = conn.cursor()
c.execute("CREATE TABLE IF NOT EXISTS incidents (id TEXT, report TEXT, timestamp TEXT)")
for incident in incidents:
    c.execute("INSERT INTO incidents VALUES (?, ?, ?)", (incident["incident_id"], incident["report"], incident["timestamp"]))

conn.commit()
conn.close()
print("Data saved successfully.")
```

---

### **5.3 Sentiment Analysis Model**  
Trains a logistic regression model and saves it.  

#### **Implementation (`train_model.py`)**  
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Training Data
texts = ["Firefighters saved lives", "Explosion caused casualties", "Routine fire drill"]
labels = [1, -1, 0]  # 1 = Positive, -1 = Negative, 0 = Neutral

# Train Model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
model = LogisticRegression()
model.fit(X, labels)

# Save Model
with open("sentiment_model.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)
```

---

### **5.4 Model Prediction API**  
A Flask API that serves sentiment predictions.  

#### **Implementation (`model_api.py`)**  
```python
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load Model
with open("sentiment_model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["incident_report"]
    X = vectorizer.transform([data])
    prediction = model.predict(X)[0]
    return jsonify({"sentiment": prediction})

if __name__ == "__main__":
    app.run(debug=True)
```

---

## **6. Deployment Guide**
| **Component** | **Deployment** |
|--------------|--------------|
| **Mock API** | Flask on Render (Free) |
| **Model API** | Flask on Render (Free) |
| **Dashboard** | Streamlit Sharing (Free) |

---

## **7. Future Enhancements**
- Replace mock API with **real API integration**.
- Upgrade storage to **PostgreSQL / BigQuery**.
- Deploy ML model using **Cloud Run / Kubernetes**.
- Replace Streamlit with **React-based frontend**.

---

