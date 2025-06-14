# local_embedder.py

import requests

def get_local_embedding(text, model="llama3", endpoint="http://localhost:11434/api/embeddings"):
    """
    Send a text to your local LLM's embedding endpoint.
    Returns the embedding vector.
    """
    payload = {
        "model": model,
        "prompt": text
    }

    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        return response.json().get("embedding", [])
    except Exception as e:
        print(f"❌ Error: {e}")
        return []
