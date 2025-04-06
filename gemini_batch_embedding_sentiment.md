
# 🚀 Batch Processing with Gemini API for Sentiment Analysis

This guide integrates **Google Gemini's embedding-001 API** into your sentiment analysis app using **batch processing** for improved efficiency and reduced costs.

---

## ✅ Why Use Batch Processing?

- **Reduces API calls**: Less latency, fewer network hits.
- **Cost-effective**: Cheaper than sending individual requests.
- **Efficient**: Speeds up embedding process for large datasets.

---

## 🔧 Setup

### 1. Install Gemini SDK

```bash
pip install google-generativeai
```

### 2. Authenticate

```python
import google.generativeai as genai

genai.configure(api_key="your-api-key")
```

---

## 🧠 Batch Embedding Function

```python
def get_embeddings_batch(texts: list, batch_size: int = 50):
    model = genai.get_model('embedding-001')
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = model.batch_embed_contents([
            {"content": txt, "task_type": "CLASSIFICATION"} for txt in batch
        ])
        all_embeddings.extend(response['embedding'])
    
    return all_embeddings
```

---

## 📘 Usage Example

```python
# Example tweets
tweets = [
    "Fire services arrived quickly and handled the situation well.",
    "No ambulance arrived even after multiple calls.",
    "Emergency responders were conducting a routine drill.",
    # Add more...
]

# Get embeddings in batches
embeddings = get_embeddings_batch(tweets)

# Proceed to train or predict using these embeddings
```

---

## 📈 Next Steps

- Train a classifier using these embeddings (`LogisticRegression`, `RandomForest`, etc.)
- Cache embeddings to avoid re-processing.
- Optionally: store embeddings in a vector DB for semantic search.

---

## 📌 Notes

- **Batch size limit**: Respect Gemini API limits (start with 50–100).
- **Token limits**: Each tweet should be under the token limit (~8k).

---

## 🔗 References

- [Gemini API Documentation](https://ai.google.dev/gemini-api/docs/embeddings)
- [Python SDK Reference](https://github.com/google/generative-ai-python)
